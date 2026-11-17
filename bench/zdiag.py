"""Where zpred loses calls once the tool is right: per-argument breakdown with the committed models.

    python3 -I bench/zdiag.py [--setting session] [--n 4000]
"""
import argparse
import collections
import json
import os
import sys

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zpred  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--setting", default="session")
ap.add_argument("--n", type=int, default=4000)
a = ap.parse_args()

xs = zpred.load("test", a.n, sources=("toolbench",))
keep = {t for x in xs for t in (*x["session"], x["tool"], *(f for f, _ in x["future"]))}
zpred.CAT = zpred.Catalog(emb=False)
zpred.SLOT = lgb.Booster(model_file=str(zpred.OUT / "slots.txt"))
zpred.SPEC = 3
zpred.INC = lgb.Booster(model_file=str(zpred.OUT / "include.txt"))
zpred.TEMP = json.load(open(zpred.OUT / f"gate_{a.setting}.json")).get("temp", 1.0)
SHOW = collections.defaultdict(list)
ranker = lgb.Booster(model_file=str(zpred.OUT / f"ranker_{a.setting}.txt"))
res = zpred.build(xs, 50, a.setting == "session", 8)

C = collections.Counter()
for r, ex in zip(res, xs):
    if r is None:
        continue
    X, y, docs, hit, ms = r
    _, picks = zpred.speculate(ranker, X, docs, ex)
    vals = zpred.context_values(ex)
    # judge against the very next call: the cleanest target for the argument breakdown
    for i, t, args in picks:
        if t["id"] != ex["tool"]:
            continue
        C["tool_right"] += 1
        tgt = {k: str(v).strip().lower() for k, v in ex["args"].items()}
        got = {k: str(v).strip().lower() for k, v in args.items()}
        if tgt == got:
            C["exact"] += 1
            break
        allv = {str(v).strip().lower() for v, *_ in zpred.with_examples(t, vals)}
        req = {x["name"] for x in t["args"] if x.get("required")}
        for k, v in tgt.items():
            if k not in got:
                C["miss_" + ("req" if k in req else "opt") + ("_avail" if v in allv else "_absent")] += 1
            elif got[k] != v:
                if k in req and v in allv and len(SHOW["wrong"]) < 12:
                    SHOW["wrong"].append((t["id"], k, got[k][:40], v[:40], ex["query"][:160]))
                C["wrong_" + ("req" if k in req else "opt") + ("_avail" if v in allv else "_absent")] += 1
        for k in got:
            if k not in tgt:
                if len(SHOW["extra"]) < 6:
                    SHOW["extra"].append((t["id"], k, got[k][:40], ex["query"][:120]))
                C["extra_" + ("req" if k in req else "opt")] += 1
        C["calls_wrong"] += 1
        # is it fixable: every target value is somewhere in context?
        C["wrong_all_values_avail"] += all(v in allv for v in tgt.values())
        break
print(json.dumps(dict(sorted(C.items())), indent=1))
for k, v in SHOW.items():
    print("==", k)
    for r in v:
        print(" ", r)
