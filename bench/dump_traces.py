"""Held-out traces for gate experiments: train on 2 customer folds, trace the 3rd, for every fold.

Usage: [CONFIG='{"context": true}'] python3 -I bench/dump_traces.py <tau-bench> <domain> <out.pkl>
Saves [(fold, inner out-of-fold traces, test traces)], so gating rules can be compared without retraining.
"""

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import traces as TR  # noqa: E402

G.CONFIG.update(json.loads(os.environ.get("CONFIG", "{}")))  # top level: pool workers re-import this module


def job(args):
    i, tr, te = args
    oof = []
    for a, b in G.folds(tr, 3):
        fz, m = G.fit(a)
        oof += [G.trace(fz, m, t) for t in b]
    fz, m = G.fit(tr)
    return i, oof, [G.trace(fz, m, t) for t in te]


if __name__ == "__main__":
    dom, out = sys.argv[2], sys.argv[3]
    jobs = [(i, tr, te) for i, (tr, te) in enumerate(G.folds(TR.load(dom), 3))]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "3"))) as ex:
        pickle.dump(list(ex.map(job, jobs)), open(out, "wb"))
