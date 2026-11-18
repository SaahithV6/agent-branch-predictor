"""Next-call prediction over the 1M-tool catalog on real multi-turn agent transcripts, before and after post-training.

The held-out test conversations of `bench/calls.py taumt` (tau2-bench airline/telecom with 4 agent models, tau-bench
airline, APIGen-MT airline; split by customer). The agent's own tool list is never given: candidates come from the
whole catalog (plus, once post-trained, the tools the user's history used). Each domain is one "user": its
conversations are split into two folds, and a fold is predicted after post-training (zadapt.post_train) on H
conversations of the other fold, so no conversation is predicted by a model that saw it.

A prediction is the 3 most likely calls. tool: one of them is a tool the agent calls later in the conversation;
call: one of them is exactly a later call (tool and arguments), the zpred metric.

    python3 -I bench/zmt.py [--hist 0,5,25]     -> bench/results/zpred/mt_catalog.json
"""
import argparse
import collections
import json
import os
import sys
import zlib

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zadapt as A  # noqa: E402
import zpred as Z  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hist", default="0,5,25")
a = ap.parse_args()

convs = collections.defaultdict(list)
for x in Z.load("test", sources=("taumt",)):
    convs[x["conv"]].append(x)
by = collections.defaultdict(lambda: ([], []))  # domain -> (fold 0, fold 1) conversations
for c in sorted(convs):
    steps = sorted(convs[c], key=lambda x: len(x["prior"]))
    by[steps[0]["tool"].split("/")[1]][zlib.crc32(str(c).encode()) % 2].append(steps)

Z.CAT = Z.Catalog(emb=False)
Z.SLOT = lgb.Booster(model_file=str(Z.OUT / "slots.txt"))
Z.INC = lgb.Booster(model_file=str(Z.OUT / "include.txt"))
A.RANKERS[False] = lgb.Booster(model_file=str(Z.OUT / "ranker_catalog.txt"))
A.SESSION = False
Z.TEMP = json.load(open(Z.OUT / "gate_catalog.json"))["temp"]
Z.SPEC = 3

out = {}
for h in map(int, a.hist.split(",")):
    n = collections.Counter()
    for dom, folds in sorted(by.items()):
        for f in (0, 1):
            st = A.post_train(folds[1 - f][:h])
            for steps in folds[f]:
                for x in steps:
                    picks = (A.predict(x, st) or [])[:Z.SPEC]
                    later = {t for t, _ in x["future"]}
                    b = Z.bucket(x)
                    for k in ("all", dom, b):
                        n[k, "steps"] += 1
                        n[k, "tool"] += any(t in later for _, t, _ in picks)
                        n[k, "call"] += any(t == ft and Z.same_args(ar, fa) for _, t, ar in picks for ft, fa in x["future"])
    out[h] = {k: {"steps": n[k, "steps"], "tool": round(n[k, "tool"] / n[k, "steps"], 4), "call": round(n[k, "call"] / n[k, "steps"], 4)}
              for k in ("all", *sorted(by), "first", "same", "new") if n[k, "steps"]}
    print(f"history {h}:", json.dumps(out[h]), flush=True)
json.dump(out, open(Z.OUT / "mt_catalog.json", "w"), indent=1)
