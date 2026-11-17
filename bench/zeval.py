"""Score the committed zpred models on the held-out ToolBench test set, without retraining.

For a change that only touches prediction (e.g. argument values), this re-measures run_bm25_spec3_k50.json's test
numbers on exactly the models the tau-bench runs used.

    python3 -I bench/zeval.py [--workers 12]     -> bench/results/zpred/eval_toolbench.json
"""
import argparse
import json
import os
import sys

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zpred as Z  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()

xs = Z.load("test", sources=("toolbench",))
Z.CAT = Z.Catalog(emb=False)
Z.SLOT = lgb.Booster(model_file=str(Z.OUT / "slots.txt"))
Z.INC = lgb.Booster(model_file=str(Z.OUT / "include.txt"))
Z.SPEC, Z.WORKERS = 3, a.workers
out = {}
for setting in ("session", "catalog"):
    cfg = json.load(open(Z.OUT / f"gate_{setting}.json"))
    Z.TEMP = cfg["temp"]
    res = Z.build(xs, cfg["k"], setting == "session", a.workers)
    t = Z.top1(lgb.Booster(model_file=str(Z.OUT / f"ranker_{setting}.txt")), res, xs)
    gate = lgb.Booster(model_file=str(Z.OUT / f"gate_{setting}.txt"))
    out[setting] = Z.report(f"{setting} test_toolbench", t, gate, {float(k): v for k, v in cfg["thresholds"].items()})
json.dump(out, open(Z.OUT / "eval_toolbench.json", "w"), indent=1)
