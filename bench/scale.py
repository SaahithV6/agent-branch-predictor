"""How the learned predictor holds up as the tool catalog grows.

Usage: python3 -I bench/scale.py <tau-bench> <domain> <K,K,...> [out.json]

Adds K decoy read tools to the catalog: each copies a real tool's argument names and value shapes
under a near-identical name (get_order_details -> get_order_details_v7), so every value that fits
the real tool also fits its decoys. Agents never call a decoy, so any decoy guess is a false fire.
This is the hard case for a big catalog: many tools that look alike, most of them rarely used.
Decoys are added to the catalog for training and test, like new tools registered on a server.

Reports, per K (3-fold by customer, gates picked out of fold): precision and coverage of the strict
gate, decoy guesses per chat, and the predictor's own time per step (candidates + scoring).
Negatives are subsampled harder as K grows (NEG scaled by real/total tools, weighted back) so
training rows stay roughly constant.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import traces as TR  # noqa: E402

K = 0
_init = G.Featurizer.__init__


def init(self, traces, generic=False):
    _init(self, traces, generic)
    real = sorted(self.args)
    for i in range(K):
        r = real[i % len(real)]
        d = f"{r}_v{i}"
        self.args[d] = self.args[r]
        for a in self.args[r]:
            if f"{r}|{a}" in self.shapes:
                self.shapes[f"{d}|{a}"] = self.shapes[f"{r}|{a}"]
    self.tools = sorted(self.args)
    self.real = set(real)


G.Featurizer.__init__ = init


def fold_job(job):
    global K
    K, train, test, base_neg, n_real = job
    G.NEG = base_neg * min(1.0, n_real / (n_real + K))
    t0 = time.time()
    fz, m, th = G.train_gated(train)
    train_s = time.time() - t0
    trs, ms = [], []
    for t in test:
        a = time.perf_counter()
        trs.append(G.trace(fz, m, t))
        ms.append((time.perf_counter() - a) * 1000 / max(len(trs[-1]["at_call"]), 1))
    decoy = sum(1 for tr in trs for k, ss in tr["best"].items() if k.split("{")[0] not in fz.real
                and max(s for s, _ in ss) >= G.Predictor.gate(th["inject"], k.split("{")[0]))
    return {"pairs": [(tr, th["inject"]) for tr in trs], "decoy": decoy, "chats": len(trs), "ms": ms, "train_s": train_s,
            "pooled": th["inject"].get("*"), "gated_tools": len([x for x in th["inject"] if x != "*"])}


if __name__ == "__main__":
    dom, ks = sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    data = TR.load(dom)
    from concurrent.futures import ProcessPoolExecutor
    report = []
    for k in ks:
        jobs = [(k, tr, te, G.NEG, len(TR.READ[dom])) for tr, te in G.folds(data, 3)]
        with ProcessPoolExecutor(3) as ex:
            res = list(ex.map(fold_job, jobs))
        s = G.score([p for r in res for p in r["pairs"]])
        ms = [x for r in res for x in r["ms"]]
        row = {"domain": dom, "decoys": k, "tools": len(TR.READ[dom]) + k, "precision": round(s["precision"], 3), "coverage": round(s["coverage"], 3),
               "calls_saved": round(s["calls_saved"], 3), "decoy_fires_per_chat": round(sum(r["decoy"] for r in res) / sum(r["chats"] for r in res), 3),
               "predict_ms_median": round(float(np.median(ms)), 2), "predict_ms_p95": round(float(np.percentile(ms, 95)), 2),
               "train_s": round(sum(r["train_s"] for r in res) / len(res), 1), "pooled_gate": [r["pooled"] for r in res],
               "gated_tools": [r["gated_tools"] for r in res]}
        report.append(row)
        print(json.dumps(row), flush=True)
        if out_path:
            json.dump(report, open(out_path, "w"), indent=1)
