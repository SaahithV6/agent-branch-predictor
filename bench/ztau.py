"""The zero-shot 1M-tool predictor (zpred) on tau-bench retail, post-trained on a few of the deployment's own sessions.

tau-bench's retail tools join the 1M catalog as one more server. The models are zpred's catalog-setting models,
trained on ToolBench + ToolACE only: they have never seen a tau-bench call. Post-training (adapt) uses H sessions of
the same agent deployment (its no-prediction runs, bench/results/ab3_base.jsonl), cross-fit over two folds of
customers: a customer is only ever predicted by a model post-trained on the other fold's customers.

    python3 -I bench/ztau.py offline [--hist 0,10,50] [--gates 0,0.5]   attached / precision / ready per gate
    python3 -I bench/ztau.py serve --hist 10 --fold 0     HTTP predictor for tau_mcp.py on port 8765 + fold
"""
import argparse
import json
import os
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zadapt as A  # noqa: E402
import zpred as Z  # noqa: E402
from tau_tools import READ, TOOLS, execute  # noqa: E402

CAT_FILE = Z.ROOT / "data/catalogs/tools_tau_retail.jsonl"
RUNS = Z.ROOT / "bench/results/ab3_base.jsonl"  # the deployment's own sessions: history for one fold, eval for the other
SPEC = int(os.environ.get("ZSPEC", "3"))  # new calls per prediction; 3 as in zpred's evaluation, tau fans out over every order


def tid(name):
    return f"tau/retail/{name}"


def setup():
    with open(CAT_FILE, "w") as f:
        for n, (d, props, _) in TOOLS.items():
            f.write(json.dumps({"source": "tau", "server": "retail", "name": n, "read": n in READ, "desc": d, "id": tid(n),
                                "args": [{"name": k, "type": v["type"], "required": True} for k, v in props.items()]}) + "\n")
    Z.CAT = Z.Catalog(paths=[*Z.CATALOGS, CAT_FILE], emb=False)
    Z.SLOT = lgb.Booster(model_file=str(Z.OUT / "slots.txt"))
    Z.INC = lgb.Booster(model_file=str(Z.OUT / "include.txt"))
    A.RANKERS[False] = lgb.Booster(model_file=str(Z.OUT / "ranker_catalog.txt"))
    A.SESSION, A.ALPHA = False, 1.0  # catalog setting; alpha as picked on dev users (adapt.json)
    Z.TEMP = json.load(open(Z.OUT / "gate_catalog.json"))["temp"]
    Z.SPEC = SPEC
    Z.VALS = max(3, SPEC)


def fold(user):
    return zlib.crc32(user.encode()) % 2


def runs(f=None, other=False):
    """The deployment's sessions; with f, those of fold f's customers (other: of the other fold's)."""
    rs = [r for r in map(json.loads, open(RUNS)) if r["actions"]]
    return rs if f is None else [r for r in rs if (fold(r["user"]) == f) != other]


def steps(hist):
    """Sessions as training steps, one list per session: request, every earlier call with its real result, the call made."""
    out = []
    for r in hist:
        acts = [(n, a) for n, a in r["actions"] if n in TOOLS]
        prior, conv = [], []
        for j, (n, a) in enumerate(acts):
            conv.append({"query": r["instruction"], "prior": list(prior), "session": [], "tool": tid(n), "args": a,
                         "future": [(tid(m), b) for m, b in acts[j:]]})
            prior.append(rec(n, a))
        out.append(conv)
    return out


def adapt(hist):
    return A.post_train(steps(hist))


def predict(st, query, prior, seen, n=SPEC):
    """The n most likely read calls not made or fetched yet: [(name, args, P)]."""
    pool = A.predict({"query": query, "prior": prior, "session": []}, st) or []
    out = []
    for lp, t, args in pool:
        name = t.removeprefix("tau/retail/")
        k = name + json.dumps(args, sort_keys=True)
        if t.startswith("tau/retail/") and name in READ and k not in seen and len(out) < n:
            seen.add(k)
            out.append((name, args, float(np.exp(lp))))
    return out


def rec(name, args):
    r = execute(name, args)[0]
    return {"tool": tid(name), "args": args, "result": r if isinstance(r, str) else json.dumps(r)}


def offline(hists, gates):
    """Replay each fold's sessions, post-trained on the first h of the other fold's, the way tau_mcp serves them.
    After each of the agent's reads: predict up to SPEC new reads, prefetch them all, attach those with P >= gate to
    the result, chain on the attached ones up to 3 times. Per gate, over the agent's reads after its first call:
    attached (already in its context when it asked), precision (attached reads it then makes), ready (either)."""
    for h in hists:
        for gate in gates:
            n = dict.fromkeys(("att", "used", "reads", "hit", "ready", "q"), 0)
            t0 = time.time()
            for f in (0, 1):
                st = adapt(runs(f, other=True)[:h])
                for r in runs(f):
                    replay(st, r, gate, n)
            print(json.dumps({"hist": h, "gate": gate, "sessions": len(runs()), "attached": round(n["hit"] / max(n["reads"], 1), 4),
                              "precision": round(n["used"] / max(n["att"], 1), 4), "ready": round(n["ready"] / max(n["reads"], 1), 4),
                              "ms_per_prediction": round((time.time() - t0) * 1000 / max(n["q"], 1), 1)}), flush=True)


def replay(st, r, gate, n):
    acts = [(c, a, c + json.dumps(a, sort_keys=True)) for c, a in r["actions"] if c in TOOLS]
    prior, seen, shown, cached = [], set(), set(), set()
    for j, (c, a, k) in enumerate(acts):
        if c in READ and j:
            n["reads"] += 1
            n["hit"] += k in shown
            n["ready"] += k in shown or k in cached
        seen.add(k)
        if c not in READ:
            continue
        prior.append(rec(c, a))
        ctx = list(prior)
        for _ in range(3):
            got = 0
            n["q"] += 1
            for pn, pa, p in predict(st, r["instruction"], ctx, seen):
                pk = pn + json.dumps(pa, sort_keys=True)
                if p < gate:
                    cached.add(pk)
                    continue
                shown.add(pk)
                n["att"] += 1
                n["used"] += any(pk == m for _, _, m in acts[j + 1:])
                ctx.append(rec(pn, pa))
                got += 1
            if not got:
                break


def serve(h, f):
    st, port = adapt(runs(f, other=True)[:h]), 8765 + f

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prior = [{"tool": tid(n), "args": a, "result": r if isinstance(r, str) else json.dumps(r)} for n, a, r in q["prior"]]
            out = predict(st, q["query"], prior, set(q["seen"]))
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"fold {f}: post-trained on {h} sessions of the other fold, serving on :{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("offline", "serve"))
    ap.add_argument("--hist", default="0,10,50")
    ap.add_argument("--gates", default="0,0.3,0.5,0.7")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    setup()
    if a.mode == "offline":
        offline([int(h) for h in a.hist.split(",")], [float(g) for g in a.gates.split(",")])
    else:
        serve(int(a.hist), a.fold)
