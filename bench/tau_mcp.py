"""Stdio MCP server exposing tau-bench's retail tools over the benchmark's own data.

Usage (spawned by bench/tau_live.py): python3 -I tau_mcp.py <tau-bench checkout> <trace.jsonl> [predictor.json]

Reads return what tau-bench's tools return; writes are checked loosely and return "ok" without
changing the data (each task gets a fresh process, so nothing leaks between tasks). Every call is
appended to the trace file as {"name", "args", "seq", "t", "ms"}: arrival time (epoch s) and
execution time in ms. Prefetched rows also carry "after" (seq of the read that triggered them)
and "pred_ms" (time spent predicting).

TOOL_LATENCY=0.8 gives every call a simulated network delay (lognormal, median 0.8 s, same delay
for the same call in every arm). Predicted reads then run in parallel; whatever finishes within
PREFETCH_WAIT seconds is attached to the result, the rest wait in a cache for the agent's call.

With predictor.json (rules + config + the task's request text, written by tau_live.py), each read
result also carries the predicted follow-up reads, already executed and marked "prefetched", so
the agent can skip those calls. Prefetches are logged to the trace with "prefetched": true.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

os.environ.setdefault("TAU_ROOT", sys.argv[1])
trace = sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pred = None
if len(sys.argv) > 3:
    import ast
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tau_replay as T
    pred = json.load(open(sys.argv[3]))
    if "vocab" in pred:  # learned predictor (lpred.py)
        import lpred
        lp = lpred.Predictor(pred, pred["model_file"])
    else:
        lp, keep = None, {ast.literal_eval(k): tuple(v) for k, v in pred["keep"].items()}
    task = {"instruction": pred["instruction"], "lower": pred["instruction"].lower()}
    seen, results = set(), []
from tau_tools import READ, TOOLS, WAIT, execute, key  # noqa: E402

KEEP = set(json.load(open(os.environ["PRUNE"]))) if os.environ.get("PRUNE") else None  # learned field list (prune.py)


def trim(name, result):
    """Attached records carry only fields agents use later (prune.py, ~20-40 us); the agent's own calls are never trimmed."""
    if KEEP is None or not isinstance(result, (dict, list)):
        return result
    import prune
    return prune.project(name, result, KEEP) or result

seq = 0
ATTACH_MIN = float(os.environ.get("ATTACH_MIN", "0"))  # 0/99 reproduce the original runs (attach every finished guess)
ATTACH_MAX = int(os.environ.get("ATTACH_MAX", "99"))  # the rest stay in the cache, never shown to the agent unasked
pool, cache = ThreadPoolExecutor(8), {}  # cache: call key -> Future of execute()


def log(row):
    global seq
    seq += 1
    with open(trace, "a") as f:
        f.write(json.dumps({**row, "seq": seq}) + "\n")
    return seq


def reply(id_, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    method, id_ = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(id_, {"protocolVersion": msg["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "tau", "version": "1"}})
    elif method == "tools/list":
        reply(id_, {"tools": [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p, "required": list(p)}} for n, (d, p, _) in TOOLS.items()]})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        k0 = key(name, args)
        if k0 in cache:  # prefetched earlier: return it, waiting only for whatever is still running
            tw = time.time()
            out = cache.pop(k0).result()[0]
            me = log({"name": name, "args": args, "t": round(tw, 3), "ms": round((time.time() - tw) * 1000, 2), "cached": True})
        else:
            out, t0, ms = execute(name, args)
            me = log({"name": name, "args": args, "t": round(t0, 3), "ms": ms})
        if name not in READ:
            cache.clear()  # never serve a read from before a write
        text = out if isinstance(out, str) else json.dumps(out)
        if pred and name in READ and not text.startswith("Error"):
            seen.add(k0)
            results.append((name, out))
            extra, deadline = [], time.time() + WAIT
            for _ in range(3):  # chain: user -> orders -> products
                tp = time.time()
                new = (lp.predict(task, results, seen, scores=True) if lp else [(k, 1.0) for k in sorted(T.predict(keep, task, results, pred["cfg"]) - seen)])[:8]
                pred_ms = round((time.time() - tp) * 1000, 2)
                if not new:
                    break
                futs, attach = {}, set()
                for k, p in new:  # all predicted reads run in parallel
                    n, a = k[:k.index("{")], json.loads(k[k.index("{"):])
                    seen.add(k)
                    futs[k] = cache[k] = pool.submit(execute, n, a)
                    if p >= ATTACH_MIN and len(attach) < ATTACH_MAX - len(extra):
                        attach.add(k)
                wait([futs[k] for k in attach], timeout=max(deadline - time.time(), 0))  # wait only for what will be shown
                got = 0
                for k, f in futs.items():
                    n, a = k[:k.index("{")], json.loads(k[k.index("{"):])
                    gate = "inject" if k in attach else "cache"
                    if k not in attach or not f.done():  # stays in the cache, served instantly if the agent asks
                        log({"name": n, "args": a, "prefetched": True, "pending": not f.done(), "gate": gate, "after": me, "pred_ms": pred_ms})
                        continue
                    r, t1, ms = f.result()
                    if isinstance(r, str) and r.startswith("Error"):
                        continue
                    results.append((n, r))
                    extra.append(f"{n}({json.dumps(a, separators=(',', ':'))}) -> {json.dumps(trim(n, r), separators=(',', ':'))}")
                    got += 1
                    log({"name": n, "args": a, "prefetched": True, "gate": gate, "after": me, "t": round(t1, 3), "ms": ms, "pred_ms": pred_ms})
                if not got or len(extra) >= ATTACH_MAX:
                    break
            if extra:
                text += ("\n\n[prefetched lookups: read-only results fetched ahead of time, so you do not need to call these tools again. "
                         "Nothing has been changed yet: still carry out every action the request needs with the tools.]\n" + "\n".join(extra))
        reply(id_, {"content": [{"type": "text", "text": text}]})
    elif id_ is not None:
        reply(id_, {})
