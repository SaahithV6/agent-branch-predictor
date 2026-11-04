"""Nested cross-validation on live agent traces.

Usage: python3 -I bench/tau_live_nested.py <tau-bench> results/live_train.jsonl results/live_test.jsonl [out.json]

Why: tau-bench's retail train tasks all quote the order number, so an agent trained only on them
never sees the test-style flow (find user -> user details -> every order). Here the test-task
users are split into 5 outer folds. For each fold the predictor trains on all train traces plus
the other 4 folds, picks its config by inner 5-fold CV on that training set (most calls saved with
CV precision >= 80%, the same rule as tau_live_eval.py), and is scored on the held-out fold, whose
users it never saw. The per-fold choice and rules are saved so the A/B run can reuse them.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import tau_replay as T  # noqa: E402
from tau_live_eval import load, safe, MIN_PRECISION  # noqa: E402

train_path, test_path = sys.argv[2], sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
GRID = {f"text={x} theta={th}": dict(text=x, levels=(1, 2) if x else (0,), mention=x, theta=th, z=0.0, stump=0)
        for x in (False, True) for th in (0.3, 0.5, 0.7)}

train, test = safe(load(train_path)), safe(load(test_path))
users = sorted({t["user"] for t in test})
fold = {u: i % 5 for i, u in enumerate(users)}
tot = {m: [0, 0, 0, 0, 0] for m in ("strict", "text")}
report = {"folds": []}
for f in range(5):
    held = [t for t in test if fold[t["user"]] == f]
    pool = train + [t for t in test if fold[t["user"]] != f]
    inner = {n: T.cv(pool, "retail", c) for n, c in GRID.items()}
    entry = {"fold": f, "tasks": len(held), "inner": inner}
    for mode in ("strict", "text"):
        ok = [n for n in inner if GRID[n]["text"] == (mode == "text") and inner[n]["precision"] >= MIN_PRECISION]
        if not ok:
            entry[mode] = None
            print(f"fold {f} {mode}: no config reaches {MIN_PRECISION:.0%}", flush=True)
            continue
        name = max(ok, key=lambda n: inner[n]["calls_saved"])
        keep = T.fit(pool, "retail", GRID[name])
        r = [0, 0, 0, 0, 0]
        for t in held:
            r = [a + b for a, b in zip(r, T.replay(keep, t, "retail", GRID[name]))]
        tot[mode] = [a + b for a, b in zip(tot[mode], r)]
        entry[mode] = {"config": name, "test": T.summarize(r)}
        json.dump({"cfg": GRID[name], "keep": {repr(k): list(v) for k, v in keep.items()}},
                  open(os.path.join(os.path.dirname(train_path), f"predictor_{mode}_fold{f}.json"), "w"))
        s = entry[mode]["test"]
        print(f"fold {f} {mode:6} {name:22} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%}", flush=True)
    report["folds"].append(entry)
report["fold_of_user"] = fold
for mode in ("strict", "text"):
    s = T.summarize(tot[mode])
    report[mode] = s
    print(f"ALL  {mode:6} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%} predicted={s['predicted']} reads={s['reads']} calls={s['calls']}", flush=True)
if out_path:
    json.dump(report, open(out_path, "w"), indent=1)
