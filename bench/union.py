"""One predictor for several tool catalogs at once vs one predictor per catalog.

Usage: python3 -I bench/union.py <tau-bench> [out.json]

Real tools only: retail, airline and telecom read tools in one catalog (18 tools; retail and airline
both have get_user_details, with different results, as happens when servers are merged). Both arms
use the same 3 customer folds; each domain is scored on its own held-out customers. Gates are picked
out of fold on training data (per tool, plus the pooled gate for rare tools).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import traces as TR  # noqa: E402

DOMS = ["retail", "airline", "telecom"]


def job(args):
    train, test = args
    fz, m, th = G.train_gated(train)
    return [(t["domain"], G.trace(fz, m, t)) for t in test], th["inject"], len(fz.tools)


if __name__ == "__main__":
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    data = {d: TR.load(d) for d in DOMS}
    allt = [t for d in DOMS for t in data[d]]
    sep = {d: G.folds(data[d], 3) for d in DOMS}
    union = G.folds(allt, 3)
    jobs = [sep[d][i] for d in DOMS for i in range(3)] + union
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        res = list(ex.map(job, jobs))
    report = []
    for arm, rs in (("separate", res[:9]), ("union", res[9:])):
        for d in DOMS:
            pairs = [(tr, gates) for traced, gates, _ in rs for dom, tr in traced if dom == d]
            s = G.score(pairs)
            row = {"arm": arm, "domain": d, "tools_in_catalog": rs[0][2] if arm == "union" else len(TR.READ[d]),
                   **{k: s[k] for k in ("precision", "coverage", "calls_saved", "predicted", "reads", "wasted_per_chat", "lead_s_median")}}
            report.append(row)
            print(json.dumps(row), flush=True)
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1)
