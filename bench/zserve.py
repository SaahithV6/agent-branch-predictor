"""Serve zero-shot next-call predictions over the whole catalog: what to prefetch, or nothing.

    p = Predictor()                     # ~40 s: indexes the 1M-tool catalog
    p.predict("Find songs by Adele", prior=[{"tool": id, "args": {...}, "result": "..."}], session=[ids])
      -> {"p": gate score, "fire": bool, "calls": [(tool id, args), ...]}

With session (the tool ids the agent was given), candidates are those tools only; without it, the
whole catalog. Models and gate thresholds come from `zpred.py run` (bench/results/zpred).
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zpred as Z  # noqa: E402


class Predictor:
    def __init__(self, target=None, out=Z.OUT):
        Z.CAT = Z.Catalog(emb=False)
        Z.SLOT = lgb.Booster(model_file=str(out / "slots.txt"))
        Z.INC = lgb.Booster(model_file=str(out / "include.txt"))
        self.m = {}
        for setting in ("session", "catalog"):
            cfg = Z.json.load(open(out / f"gate_{setting}.json"))
            # the precision target the gate was certified for on dev; test precision runs lower (see results)
            # 1.01 = dev could not certify that target: the gate never fires
            th = cfg["thresholds"][str(target or {"session": 0.7, "catalog": 0.5}[setting])]
            self.m[setting] = (lgb.Booster(model_file=str(out / f"ranker_{setting}.txt")),
                               lgb.Booster(model_file=str(out / f"gate_{setting}.txt")), th, cfg)
        Z.SPEC = self.m["catalog"][3]["spec"]

    def predict(self, query, prior=(), session=None):
        t0 = time.perf_counter()
        setting = "session" if session else "catalog"
        ranker, gate, th, cfg = self.m[setting]
        Z.TEMP = cfg.get("temp", 1.0)
        ex ={"query": query, "prior": list(prior), "session": list(session or ())}
        docs, bm, emb = Z.candidates(Z.CAT, ex, cfg["k"], bool(session))
        if not len(docs):
            return {"p": 0.0, "fire": False, "calls": [], "ms": 0.0}
        X = Z.featurize(Z.CAT, ex, docs, bm, emb, bool(session))
        g, picks = Z.speculate(ranker, X, docs, ex)
        p = float(gate.predict(np.array([g], np.float32), num_threads=1)[0])
        return {"p": p, "fire": p >= th, "calls": [(t["id"], args) for _, t, args in picks],
                "ms": (time.perf_counter() - t0) * 1000}


if __name__ == "__main__":  # self-check on held-out test calls: fired predictions should mostly pay off
    P = Predictor()
    xs = Z.load("test", 400, sources=("toolbench",))
    fired = paid = 0
    for x in xs:
        r = P.predict(x["query"], x["prior"])
        if r["fire"]:
            fired += 1
            paid += any(t == tid and Z.same_args(a, b) for t, a in r["calls"] for tid, b in x["future"])
    print(f"catalog: fired on {fired}/{len(xs)}, paid off {paid}/{fired}, "
          f"p50 {np.median([P.predict(x['query'], x['prior'])['ms'] for x in xs[:100]]):.1f} ms")
    assert fired and paid / fired > 0.4
