"""Compare gating rules on dumped held-out traces (bench/dump_traces.py), no retraining.

Usage: python3 -I bench/gate_eval.py <tau-bench> <traces.pkl> [...]

Gates are picked on each fold's inner out-of-fold traces and scored on its test customers, as
gpred.train_gated does; only the rule for picking them changes.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import tau_replay as T  # noqa: E402

BASE = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.01)]  # gpred.pick's grid
FINE = BASE + [0.995, 0.998, 0.999, 0.9995, 0.9999]


def pick(trs, tools, xs, target=None):
    """gpred.pick with the threshold grid as a parameter."""
    target = G.TARGET if target is None else target
    by = {}
    for tr in trs:
        for k, ss in tr["best"].items():
            by.setdefault(k.split("{")[0], []).append((max(s for s, _ in ss), k in tr["actual"]))

    def first_ok(preds):
        if not preds:
            return None
        s, h = np.array(preds, dtype=np.float64).T
        for x in xs:
            on = s >= x
            n = int(on.sum())
            if n >= 20 and T.wilson(int(h[on].sum()), n, 1.645) >= target:
                return x
    th = {}
    for tool in tools:
        x = first_ok(by.get(tool, []))
        if x is not None:
            th[tool] = x
    x = first_ok([p for t, ps in by.items() if t not in th for p in ps])
    if x is not None:
        th["*"] = x
    return th


def run(folds, xs, target=None):
    pairs, gates = [], []
    for _, oof, test in folds:
        tools = sorted({k.split("{")[0] for t in oof for k in t["best"]})
        th = pick(oof, tools, xs, target)
        gates.append(th)
        pairs += [(t, th) for t in test]
    s = G.score(pairs)
    return {k: round(s[k], 4) if isinstance(s[k], float) else s[k] for k in ("precision", "coverage", "calls_saved", "wasted_per_chat")}, gates


if __name__ == "__main__":
    for p in sys.argv[2:]:
        folds = pickle.load(open(p, "rb"))
        for name, xs in (("base", BASE), ("fine", FINE)):
            s, gates = run(folds, xs)
            print(os.path.basename(p), name, json.dumps(s), json.dumps(gates[0]), flush=True)
