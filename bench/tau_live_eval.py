"""Score the predictor on real agent traces from bench/tau_live.py.

Usage: python3 -I bench/tau_live_eval.py <tau-bench> results/live_train.jsonl results/live_test.jsonl [out.json]

Protocol (fixed before any live results were seen): every config in GRID is scored by 5-fold CV
(split by user) on the training-task traces only. The config with the most calls saved among
those with CV precision >= 80% is picked, retrained on all training traces, and scored once on
the test-task traces. The test traces never influence the choice.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import tau_replay as T  # noqa: E402

train_path, test_path = sys.argv[2], sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
MIN_PRECISION = 0.80


def load(path):
    tasks = []
    for line in open(path):
        r = json.loads(line)
        if r.get("error") or not r["actions"]:
            continue
        tasks.append({"user": r["user"], "instruction": r["instruction"], "lower": r["instruction"].lower(),
                      "actions": [(n, a) for n, a in r["actions"] if n != "calculate"]})
    return tasks


def safe(tasks):
    """Drop calls whose arguments the model got wrong (the tool would have errored), so replay can
    recompute every read result."""
    out = []
    for t in tasks:
        acts = []
        for n, a in t["actions"]:
            if n in T.READS["retail"]:
                try:
                    if T.READS["retail"][n](**a) is None:
                        continue
                except TypeError:
                    continue
            acts.append((n, a))
        out.append({**t, "actions": acts})
    return out


GRID = {}
for text in (False, True):
    for theta in (0.3, 0.5, 0.7):
        for z in (0.0, 1.0):
            for stump in ((0, 3) if text else (0,)):
                GRID[f"text={text} theta={theta} z={z} stump={stump}"] = dict(text=text, levels=(1, 2) if text else (0,), mention=text, theta=theta, z=z, stump=stump)

if __name__ == "__main__":
    train, test = safe(load(train_path)), safe(load(test_path))
    print(f"train traces={len(train)} test traces={len(test)}", flush=True)
    rows = []
    for name, cfg in GRID.items():
        s = T.cv(train, "retail", cfg)
        rows.append((name, cfg, s))
        print(f"cv  {name:40} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%}", flush=True)
    report = {"grid": [{"config": n, "train_cv": s} for n, _, s in rows]}
    for mode in (False, True):
        ok = [r for r in rows if r[1]["text"] == mode and r[2]["precision"] >= MIN_PRECISION]
        if not ok:
            print(f"no {'text' if mode else 'strict'} config reaches {MIN_PRECISION:.0%} CV precision")
            continue
        name, cfg, s = max(ok, key=lambda r: r[2]["calls_saved"])
        final = T.summarize(T.evaluate(train, test, "retail", cfg))
        report["text" if mode else "strict"] = {"config": name, "train_cv": s, "test": final}
        keep = T.fit(train, "retail", cfg)
        json.dump({"cfg": cfg, "keep": {repr(k): list(v) for k, v in keep.items()}},
                  open(os.path.join(os.path.dirname(train_path), f"predictor_{'text' if mode else 'strict'}.json"), "w"))
        print(f"TEST {name:40} precision={final['precision']:.0%} coverage={final['coverage']:.0%} calls_saved={final['calls_saved']:.0%} predicted={final['predicted']} reads={final['reads']} calls={final['calls']}", flush=True)
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1)
