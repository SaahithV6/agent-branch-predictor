"""Drive prefetchd with real trajectories chained into long workflows, checking every answer.

Usage: python3 -I bench/workload.py <tau-bench> --calls N [--chain C] [--clients P] [--threads T]
                                    [--think S] [--shards K] [--port 9100] [--reset] [--out run.json]

Every recorded tau2 conversation (retail, airline, telecom) is a ticket. A workflow is C tickets
from the same customer fold back to back in one session, so a session runs hundreds of tool calls,
far past the ~7-20 the predictor was trained on. Each ticket gets a fresh environment from its
task's own starting state, and the predictor's memory is cleared between tickets; with --blind it
is never told where one ticket ends (the harder case: a single unbroken workflow). Passes over the
corpus repeat in new orders until N tool calls have run.

Every call is checked: the client keeps its own shadow environment, runs the same call directly,
and compares. A result served from a prefetch must equal the direct one, always.
"""

import argparse
import collections
import http.client
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realtools as RT  # noqa: E402
import traces as TR  # noqa: E402

BUCKETS = [20, 100, 500, 2000, 10 ** 9]
ROUTER = 0  # set from --router; read by Client in the forked client processes
FOREIGN = []  # (server, tool) pairs from --foreign-tools  # position of the call within its session


def workflows(args, models):
    out, total, p = [], 0, 0
    by = collections.defaultdict(list)  # (domain, fold) -> tickets
    for d in args.domains.split(","):
        fold_of = json.load(open(os.path.join(models, f"gpred_{d}_folds.json")))["fold_of_group"]
        for t in RT.load(d):
            by[(d, fold_of[t["group"]])].append({"task": t["task"], "steps": [{k: s[k] for k in ("name", "args", "text")} for s in t["steps"]]})
    while total < args.calls:
        for (d, f), ts in sorted(by.items()):
            ts = ts[:]
            random.Random(p * 1000 + f).shuffle(ts)
            for i in range(0, len(ts), args.chain):
                chunk = ts[i:i + args.chain]
                out.append({"domain": d, "fold": f, "tickets": chunk, "id": f"{p}:{d}:{f}:{i}"})
                total += sum(len(t["steps"]) for t in chunk)
        p += 1
    random.Random(0).shuffle(out)
    return out, total, p


class Client:
    def __init__(self, port, shards):
        self.port, self.shards, self.conns = port, shards, {}

    def post(self, shard, path, body):
        if ROUTER:
            shard = -1  # one router port; the session id says which shard
        for attempt in range(3):
            c = self.conns.get(shard) or http.client.HTTPConnection("127.0.0.1", ROUTER or self.port + shard, timeout=120)
            self.conns[shard] = c
            try:
                c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = c.getresponse()
                out = json.loads(r.read())
                if r.status != 200:
                    raise RuntimeError(f"{path} {r.status} {out}")
                return out
            except (ConnectionError, http.client.HTTPException, TimeoutError):
                c.close()
                self.conns.pop(shard, None)
                if attempt == 2:
                    raise


def bucket(pos):
    return next(i for i, b in enumerate(BUCKETS) if pos < b)


def run_workflow(w, cl, a, st, lat):
    shard = zlib.crc32(w["id"].encode()) % a.shards
    tickets = w["tickets"]
    sid = cl.post(shard, "/session", {"domain": w["domain"], "fold": w["fold"], "task": tickets[0]["task"], "servers": [w["domain"]]})["sid"]
    pos = 0
    for i, t in enumerate(tickets):
        if i:
            cl.post(shard, "/task", {"sid": sid, "task": t["task"], "reset": not a.blind})
        shadow, said = RT.make_env(w["domain"], t["task"]), ""
        for s in t["steps"]:
            if s["text"] != said:
                cl.post(shard, "/text", {"sid": sid, "text": s["text"][len(said):] if s["text"].startswith(said) else s["text"]})
                said = s["text"]
                time.sleep(a.think)
            t0 = time.perf_counter()
            r = cl.post(shard, "/call", {"sid": sid, "name": s["name"], "args": s["args"]})
            lat.append((time.perf_counter() - t0) * 1000)
            want, _ = RT.call(shadow, s["name"], s["args"])
            b = bucket(pos)
            read = s["name"] in TR.READ[w["domain"]]
            with st["lock"]:
                c = st["n"]
                c["calls"] += 1
                c[f"source_{r['source']}"] += 1
                if r["result"] != want:
                    c["mismatch_" + r["source"]] += 1
                if read:
                    c[f"reads_b{b}"] += 1
                    c[f"prefetched_b{b}"] += r["source"] in ("cache", "inflight")  # memo: a repeat of the agent's own read
                    c[f"attached_b{b}"] += bool(r["attached"])
            pos += 1
            if FOREIGN and random.random() < a.foreign_rate:  # a catalog tool on another server, through the router
                server, name = random.choice(FOREIGN)
                t0 = time.perf_counter()
                r = cl.post(shard, "/call", {"sid": sid, "name": name, "server": server, "args": {"q": pos}})
                ms = (time.perf_counter() - t0) * 1000
                ok = json.loads(r["result"]) == {"tool": name, "args": {"q": pos}, "upstream": "stub"}
                with st["lock"]:
                    st["n"]["foreign_calls"] += 1
                    st["n"]["foreign_mismatch"] += not ok
                    st["n"]["foreign_ms_x1000"] += int(ms * 1000)
            time.sleep(a.think)
    cl.post(shard, "/close", {"sid": sid})


def client_proc(idx):
    """Runs in a forked process: the workflows, args and progress counter are inherited."""
    ws, a, progress = WS[idx::A.clients], A, PROGRESS
    st = {"lock": threading.Lock(), "n": collections.Counter()}
    lat, errors = [], []
    it = iter(ws)
    it_lock = threading.Lock()

    def worker():
        cl = Client(a.port, a.shards)
        while True:
            with it_lock:
                w = next(it, None)
            if w is None:
                return
            try:
                run_workflow(w, cl, a, st, lat)
            except Exception as e:
                errors.append(f"{w['id']}: {e!r}")
            with progress.get_lock():
                progress.value += sum(len(t["steps"]) for t in w["tickets"])

    th = [threading.Thread(target=worker) for _ in range(a.threads)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return dict(st["n"]), random.Random(idx).sample(lat, min(len(lat), 50000)), errors


def metrics(a):
    out = []
    for i in range(a.shards):
        c = http.client.HTTPConnection("127.0.0.1", a.port + i, timeout=30)
        c.request("GET", "/metrics")
        out.append(json.loads(c.getresponse().read()))
    return out


def summarize(a, n, lat, shards, wall, total_planned, errors):
    cnt = collections.Counter()
    for s in shards:
        cnt.update(s["counts"])
    by_pos = []
    for i, b in enumerate(BUCKETS):
        r = n.get(f"reads_b{i}", 0)
        if r:
            by_pos.append({"calls_into_session": f"<{b}" if b < 10 ** 9 else f">={BUCKETS[i - 1]}", "reads": r,
                           "ready_ahead": round(n.get(f"prefetched_b{i}", 0) / r, 4), "attached": round(n.get(f"attached_b{i}", 0) / r, 4)})
    reads = sum(n.get(f"reads_b{i}", 0) for i in range(len(BUCKETS)))
    pooled = lambda k: {q: round(float(np.median([s["ms"][k][str(q)] for s in shards if k in s["ms"]])), 3) for q in (50, 90, 99)}  # noqa: E731
    return {
        "config": {k: v for k, v in vars(a).items()},
        "env": {k: os.environ.get(k) for k in ("WINDOW", "TEXT_CHARS", "MAX_PREFETCH", "CACHE_MAX") if os.environ.get(k)},
        "tool_calls": n.get("calls", 0), "wall_s": round(wall, 1), "calls_per_s": round(n.get("calls", 0) / wall, 1),
        "workflow_errors": len(errors), "error_examples": errors[:5],
        "served": {k[7:]: v for k, v in n.items() if k.startswith("source_")},
        "foreign": {"calls": n.get("foreign_calls", 0), "mismatches": n.get("foreign_mismatch", 0),
                    "mean_ms": round(n.get("foreign_ms_x1000", 0) / 1000 / max(n.get("foreign_calls", 0), 1), 3)},
        "mismatches": {k[9:]: v for k, v in n.items() if k.startswith("mismatch_")},
        "reads": reads, "ready_ahead": round(sum(n.get(f"prefetched_b{i}", 0) for i in range(len(BUCKETS))) / max(reads, 1), 4),
        "attached": round(sum(n.get(f"attached_b{i}", 0) for i in range(len(BUCKETS))) / max(reads, 1), 4),
        "attach_precision": round(cnt["attached_hit"] / max(cnt["attached"], 1), 4),
        "prefetch_precision": round(cnt["prefetch_used"] / max(cnt["prefetch_done"], 1), 4),
        "by_position": by_pos,
        "client_call_ms": {q: round(float(np.percentile(lat, q)), 3) for q in (50, 90, 99)} if lat else {},
        "server_ms_median_of_shards": {k: pooled(k) for k in ("call", "predict", "queue_lag")},
        "server_counts": dict(cnt),
        "rss_mb_per_shard_max": max(s["rss_mb"] for s in shards),
        "planned_calls": total_planned,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tau")
    ap.add_argument("--calls", type=int, default=20000)
    ap.add_argument("--chain", type=int, default=50)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--think", type=float, default=0.2)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--blind", action="store_true", help="never tell the predictor where a ticket ends")
    ap.add_argument("--domains", default=",".join(RT.DOMAINS))
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "live"))
    ap.add_argument("--out")
    ap.add_argument("--router", type=int, default=0, help="send everything through bench/router.py on this port")
    ap.add_argument("--foreign-tools", help="catalog jsonl: with --foreign-rate, also call random catalog tools via the router")
    ap.add_argument("--foreign-rate", type=float, default=0.0)
    a = ap.parse_args()
    ROUTER = a.router
    if a.foreign_tools:
        FOREIGN = [(t["server"], t["name"]) for t in map(json.loads, open(a.foreign_tools)) if t["read"]]
    WS, planned, passes = workflows(a, a.models)
    A, PROGRESS = a, mp.Value("q", 0)
    progress = PROGRESS
    print(f"{len(WS)} workflows, {planned} tool calls planned ({passes} passes over the corpus)", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.clients) as pool:
        res = pool.map_async(client_proc, range(a.clients))
        while not res.ready():
            res.wait(30)
            el = time.time() - t0
            print(f"[{el:7.0f}s] {progress.value:>9} / {planned} calls  ({progress.value / max(el, 1):.0f}/s)", flush=True)
        res = res.get()
    wall = time.time() - t0
    n, lat, errors = collections.Counter(), [], []
    for c, l_, e in res:
        n.update(c)
        lat += l_
        errors += e
    rep = summarize(a, n, lat, metrics(a), wall, planned, errors)
    print(json.dumps(rep, indent=1), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
