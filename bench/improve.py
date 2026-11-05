"""Self-improvement loop over the predictor's structure, driven by its worst cases.

Usage: python3 -I bench/improve.py <tau-bench> [out.json]

Each round:
  1. Score the current structure on validation folds (3-fold by customer) in every domain.
  2. Collect its worst cases: confident wrong guesses (false positives) and lookups it never covered
     (false negatives, split into "no candidate generated" and "scored too low"), grouped by generic
     pattern (tool, where the value came from), never by task or customer.
  3. Turn the dominant patterns into an ordered queue of structural changes (feature groups, model
     capacity), try each, and keep the best one only if it raises the mean score across domains and
     no domain drops by more than 1 point.
Repeat until no change helps. A test split (20% of customers per domain) is never seen by the loop;
the final structure is scored on it exactly once, next to the starting structure.

Score per domain = coverage of lookups attached at the strict gate + half the coverage ready at the
loose gate, with both gates picked on the same out-of-fold predictions for every variant.
"""

import json
import os
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ROUNDS", "150")
os.environ.setdefault("NEG", "0.1")
import gpred as G  # noqa: E402
import traces as TR  # noqa: E402

DOMAINS = ["retail", "airline", "telecom"]
out_path = sys.argv[2] if len(sys.argv) > 2 else None


def split(trs):
    """Test = customers whose id hashes to the last 20%; the loop never sees them."""
    test = lambda t: zlib.crc32(t["group"].encode()) % 5 == 0  # noqa: E731
    return [t for t in trs if not test(t)], [t for t in trs if test(t)]


def oof(job):
    """Out-of-fold traces for one domain under one structure."""
    cfg, trs = job
    G.CONFIG.update(cfg)
    out = []
    for tr, te in G.folds(trs, 3):
        fz, m = G.fit(tr)
        out += [G.trace(fz, m, t) for t in te]
    return out, fz.tools


def run_cfg(job):
    cfg, train, test = job
    G.CONFIG.update(cfg)  # workers start fresh (forkserver), so the structure travels with the job
    return G.run(train, test)


def evaluate(cfg, data):
    with ProcessPoolExecutor(len(DOMAINS)) as ex:
        res = dict(zip(DOMAINS, ex.map(oof, [(cfg, data[d]) for d in DOMAINS])))
    report = {}
    for d, (trs, tools) in res.items():
        g = {"inject": G.pick(trs, tools, G.TARGET), "cache": G.pick(trs, tools, G.CACHE_TARGET)}
        i, c = G.score([(t, g["inject"]) for t in trs]), G.score([(t, g["cache"]) for t in trs])
        report[d] = {"score": round(100 * (i["coverage"] + 0.5 * c["coverage"]), 2), "attached": round(i["coverage"], 3),
                     "attached_precision": round(i["precision"], 3), "ready": round(c["coverage"], 3), "ready_precision": round(c["precision"], 3),
                     "worst": worst(trs, g)}
    report["mean"] = round(sum(report[d]["score"] for d in DOMAINS) / len(DOMAINS), 2)
    return report


def worst(trs, g):
    """Generic failure patterns: confident wrong guesses and uncovered lookups."""
    fp, fn = Counter(), Counter()
    for tr in trs:
        for k, ss in tr["best"].items():
            tool = k.split("{")[0]
            if max(s for s, _ in ss) >= g["inject"].get(tool, 2) and k not in tr["actual"]:
                fp[tool] += 1
        for tool, ss, _ in tr["at_call"]:
            if not ss:
                fn[(tool, "no candidate")] += 1
            elif max(s for s, _ in ss) < g["cache"].get(tool, 2):
                fn[(tool, "scored too low")] += 1
    return {"false_positives": dict(fp.most_common(5)), "missed": {f"{t}: {why}": n for (t, why), n in fn.most_common(6)}}


def queue(cfg, report):
    """Structural changes to try, ordered by what the worst cases point at."""
    fp = sum(sum(report[d]["worst"]["false_positives"].values()) for d in DOMAINS)
    low = sum(n for d in DOMAINS for k, n in report[d]["worst"]["missed"].items() if "too low" in k)
    nocand = sum(n for d in DOMAINS for k, n in report[d]["worst"]["missed"].items() if "no candidate" in k)
    q = []
    if low >= fp and not cfg["siblings"]:  # unsure "which one of several": tell the model what the agent already opened
        q.append({"siblings": True})
    if not cfg["ambiguity"]:
        q.append({"ambiguity": True})
    if fp > low and not cfg["siblings"]:
        q.append({"siblings": True})
    if cfg["words"]:  # vocabulary features can memorize scenarios; check they earn their place
        q.append({"words": False})
    if cfg["cats"]:
        q.append({"cats": False})
    q += [{"leaves": x} for x in (31, 127) if x != cfg["leaves"]]
    print(f"  worst cases: confident wrong={fp} scored-too-low={low} no-candidate={nocand} -> try {q}", flush=True)
    return q


if __name__ == "__main__":
    full = {d: split(TR.load(d)) for d in DOMAINS}
    dev = {d: full[d][0] for d in DOMAINS}
    cfg = dict(G.CONFIG)
    start_cfg = dict(cfg)
    log = []
    cur = evaluate(cfg, dev)
    print(f"round 0 {cfg} mean={cur['mean']} " + " ".join(f"{d}={cur[d]['score']}" for d in DOMAINS), flush=True)
    log.append({"round": 0, "config": dict(cfg), "report": cur})
    for rnd in range(1, 6):
        best = None
        for change in queue(cfg, cur):
            trial = {**cfg, **change}
            r = evaluate(trial, dev)
            ok = r["mean"] >= cur["mean"] + 0.5 and all(r[d]["score"] >= cur[d]["score"] - 1 for d in DOMAINS)
            print(f"  try {change}: mean={r['mean']} " + " ".join(f"{d}={r[d]['score']}" for d in DOMAINS) + (" ACCEPTABLE" if ok else ""), flush=True)
            log.append({"round": rnd, "try": change, "report": r, "acceptable": ok})
            if ok and (best is None or r["mean"] > best[1]["mean"]):
                best = (trial, r)
        if not best:
            print(f"round {rnd}: no structural change helps; stopping", flush=True)
            break
        cfg, cur = best
        print(f"round {rnd} KEEP {cfg} mean={cur['mean']}", flush=True)
        log.append({"round": rnd, "kept": dict(cfg), "report": cur})
    # one look at the untouched test customers: starting structure vs final structure
    final = {}
    for name, c in (("start", start_cfg), ("final", cfg)):
        with ProcessPoolExecutor(len(DOMAINS)) as ex:
            res = dict(zip(DOMAINS, ex.map(run_cfg, [(c, full[d][0], full[d][1]) for d in DOMAINS])))
        final[name] = {}
        for d, (g, t) in res.items():
            i, r = G.score([(x, g["inject"]) for x in t]), G.score([(x, g["cache"]) for x in t])
            final[name][d] = {"attached": round(i["coverage"], 3), "precision": round(i["precision"], 3),
                              "ready": round(r["coverage"], 3), "ready_precision": round(r["precision"], 3), "wasted_per_chat": r["wasted_per_chat"]}
        print(f"TEST {name} {c}: " + " | ".join(f"{d} attached {final[name][d]['attached']:.0%} @ {final[name][d]['precision']:.0%}, ready {final[name][d]['ready']:.0%}" for d in DOMAINS), flush=True)
    if out_path:
        json.dump({"log": log, "final_config": cfg, "test": final}, open(out_path, "w"), indent=1, default=lambda o: o.item())
