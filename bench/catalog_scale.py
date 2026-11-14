"""How retail prediction holds up when the tool catalog grows to a million real tools.

Usage: python3 -I bench/catalog_scale.py <tau-bench> <tools.jsonl> <out.json> [N,N,...]

The retail tools agents actually call are embedded among N tools from public catalogs (bench/catalog.py:
OpenAPI directories, cloud SDK models, RapidAPI). Agents never call a catalog tool, so every catalog
guess that clears a gate is a false fire. The model is trained once on retail customers (fold 0 of 3,
inner out-of-fold traces for the gates, as gpred.train_gated); for each N the catalog's shortlist
(bench/toolindex.py) is attached, gates are re-picked on the out-of-fold traces, and the held-out
customers are scored. One more arm trains with the full catalog attached (its tools as negatives).

Reports per N: precision / coverage on retail, catalog false fires per chat, catalog slots scored per
prediction, ms per prediction, index build time and memory.
"""

import json
import os
import random
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import toolindex as X  # noqa: E402
import traces as TR  # noqa: E402


def attach(fzs, ix):
    for fz in fzs:
        fz.catalog = ix


def traced(fz, m, chats):
    t0, out, slots, steps = time.perf_counter(), [], 0, 0
    for t in chats:
        out.append(G.trace(fz, m, t))
        steps += max(len(out[-1]["at_call"]), 1)
    ms = (time.perf_counter() - t0) * 1000 / steps
    return out, ms


def evaluate(models, test, ix):
    """models: (inner [(fz, m, chats)], (fz, m)); gates re-picked with the catalog attached."""
    inner, (fz, m) = models
    attach([f for f, _, _ in inner] + [fz], ix)
    oof = [tr for f, mm, chats in inner for tr in traced(f, mm, chats)[0]]
    th = G.pick(oof, fz.tools)
    trs, ms = traced(fz, m, test)
    s = G.score([(t, th) for t in trs])
    real = set(fz.args)
    fires = sum(1 for t in trs for k, ss in t["best"].items() if k.split("{")[0] not in real
                and max(x for x, _ in ss) >= G.Predictor.gate(th, k.split("{")[0]))
    return {"precision": round(s["precision"], 4), "coverage": round(s["coverage"], 4), "calls_saved": round(s["calls_saved"], 4),
            "catalog_fires_per_chat": round(fires / len(trs), 3), "ms_per_prediction": round(ms, 2),
            "pooled_gate": th.get("*"), "gated_tools": sorted(t for t in th if t != "*")}


def train(pool, ix=None):
    inner = []
    for tr, te in G.folds(pool, 3):
        fz = G.Featurizer(tr)
        fz.catalog = ix
        X_, y, w = G.dataset(fz, tr)
        m = G.lgb.train({**G.PARAMS, "num_leaves": fz.config["leaves"]},
                        G.lgb.Dataset(X_, y, weight=w, categorical_feature=G.CAT + [i for i, n in enumerate(fz.names()) if n == G.CONTEXT_CAT]), G.ROUNDS)
        inner.append((fz, m, te))
    fz = G.Featurizer(pool)
    fz.catalog = ix
    X_, y, w = G.dataset(fz, pool)
    m = G.lgb.train({**G.PARAMS, "num_leaves": fz.config["leaves"]},
                    G.lgb.Dataset(X_, y, weight=w, categorical_feature=G.CAT + [i for i, n in enumerate(fz.names()) if n == G.CONTEXT_CAT]), G.ROUNDS)
    return inner, (fz, m)


if __name__ == "__main__":
    tau, cat_path, out_path = sys.argv[1:4]
    sizes = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1000,10000,100000,1000000000").split(",")]
    data = TR.load("retail")
    pool, test = G.folds(data, 3)[0]
    tools = [json.loads(line) for line in open(cat_path)]
    rng = random.Random(0)
    tools.sort(key=lambda t: (t.get("tenant", 0) > 0, rng.random()))  # nested samples of one fixed order: real tools first, tenant copies last
    report = []
    t0 = time.time()
    models = train(pool)
    print(f"trained without catalog in {time.time() - t0:.0f}s", flush=True)
    for n in sizes:
        n = min(n, len(tools))
        t0 = time.time()
        ix = X.ToolIndex(tools[:n]) if n else None
        build = time.time() - t0
        row = {"catalog_tools": n, "tenant_copies": sum(1 for t in tools[:n] if t.get("tenant")), "indexed_read_tools": len(ix.args) if ix else 0, "index_build_s": round(build, 1),
               **evaluate(models, test, ix), "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
        report.append(row)
        print(json.dumps(row), flush=True)
        json.dump(report, open(out_path, "w"), indent=1)
        if n == len(tools):
            break
    t0 = time.time()
    full = X.ToolIndex(tools)
    trained = train(pool, full)
    row = {"catalog_tools": len(tools), "trained_with_catalog": True, "train_s": round(time.time() - t0),
           **evaluate(trained, test, full), "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
    report.append(row)
    print(json.dumps(row), flush=True)
    json.dump(report, open(out_path, "w"), indent=1)
