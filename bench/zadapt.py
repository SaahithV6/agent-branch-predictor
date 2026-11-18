"""Per-user adaptation of the zero-shot predictor from the user's own past sessions: counts, no retraining.

ToolBench has no user ids, so a "user" is every held-out conversation whose main API is the same server: the
traffic one team's agent sends to one service. History = that user's first H conversations; evaluation = their
conversations from MIN_HIST on, the same set for every H, so H = 0 is the zero-shot model on the same steps.

  tools      log P(tool | previous tool) from the history is added to the ranker's log P(tool) (weight alpha,
             picked on dev users), and the user's tools join the candidates: a 1M catalog no longer has to find them
  arguments  a (tool, argument)'s most frequent past value joins the context as a documented example, and
             P(passed) is smoothed toward the user's own rate

    python3 -I bench/zadapt.py [--settings session,catalog] [--hist 0,5,10,20]
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import re
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zpred as Z  # noqa: E402

MIN_HIST = 20  # conversations kept back as possible history; users need MIN_HIST + 10 to be evaluated
BASE_EXAMPLES = Z.TOOL_EXAMPLES


def users(split):
    xs = Z.load("test" if split == "test" else "dev", sources=("toolbench",))
    conv = collections.defaultdict(list)
    for x in xs:
        conv[x["conv"]].append(x)
    by = collections.defaultdict(list)
    for c in sorted(conv):  # conversation id order stands in for time
        main = collections.Counter(x["tool"].split("/")[1] for x in conv[c]).most_common(1)[0][0]
        by[main].append(sorted(conv[c], key=lambda x: len(x["prior"])))
    return {u: cs for u, cs in by.items() if len(cs) >= MIN_HIST + 10}


def stats(hist):
    trans, cnt = collections.defaultdict(collections.Counter), collections.Counter()
    vals, inc = collections.defaultdict(collections.Counter), collections.defaultdict(lambda: [0, 0])
    for steps in hist:
        prev = "<s>"
        for x in steps:
            trans[prev][x["tool"]] += 1
            cnt[x["tool"]] += 1
            prev = x["tool"]
            t = Z.CAT.tools[Z.CAT.pos[x["tool"]]]
            for a in t["args"]:
                inc[(x["tool"], a["name"])][1] += 1
                if a["name"] in x["args"]:
                    inc[(x["tool"], a["name"])][0] += 1
                    v = x["args"][a["name"]]
                    if not isinstance(v, (dict, list)):
                        vals[(x["tool"], a["name"])][str(v)] += 1
    ex = collections.defaultdict(dict)
    for (tool, arg), c in vals.items():
        ex[tool][arg] = c.most_common(1)[0][0]
    return trans, cnt, ex, inc


RANKERS, ALPHA, SESSION = {}, 1.0, True


def features(ex, st):
    """Candidates (the user's own tools always among them) and their ranker features."""
    cnt = st[1]
    docs, _, _ = Z.candidates(Z.CAT, ex, 50, SESSION)
    if not SESSION and cnt:  # the user's own tools are always candidates
        docs = np.unique(np.concatenate([docs, [Z.CAT.pos[t] for t in cnt]])).astype(np.int64)
    if not len(docs):
        return None
    q = Z.words(ex["query"])
    if ex["prior"]:
        last = ex["prior"][-1]
        q += Z.words(" ".join(re.split(r"[/_]", last["tool"]))) + Z.words(last["result"][:400])
    bm = Z.CAT.bm25(q, docs)
    return docs, Z.featurize(Z.CAT, ex, docs, bm, None, SESSION)


def predict(ex, st):
    trans, cnt, _, inc = st
    r = features(ex, st)
    if r is None:
        return None
    docs, X = r
    s = RANKERS[SESSION].predict(X, num_threads=1) / Z.TEMP
    lp = s - np.logaddexp.reduce(s)
    if cnt:
        prev = ex["prior"][-1]["tool"] if ex["prior"] else "<s>"
        n, tr = sum(cnt.values()), trans.get(prev, {})
        ids = [Z.CAT.tools[d]["id"] for d in docs]
        uni = np.array([(cnt.get(t, 0) + 0.01) / (n + 0.01 * len(docs)) for t in ids])
        pu = (np.array([tr.get(t, 0) for t in ids]) + uni) / (sum(tr.values()) + 1)
        lp = lp + ALPHA * np.log(pu)
        lp -= np.logaddexp.reduce(lp)
    values = Z.context_values(ex)
    pool = []
    for i in np.argsort(-lp)[:Z.TK]:
        t = Z.CAT.tools[docs[i]]
        opts = []
        for name, p, vs in Z.arg_options(t, values)[0]:
            k, m = inc.get((t["id"], name), (0, 0))
            opts.append((name, (k + 2 * p) / (m + 2), vs))
        for args, la in Z.configs(opts, Z.SPEC):
            pool.append((lp[i] + la, t["id"], args))
    pool.sort(key=lambda c: -c[0])
    return pool  # [(log P(call), tool id, args)], most likely first


ROUNDS = 50  # boosting rounds added on top of each base model by post_train
BASE = {}


def post_train(convs):
    """Post-train on a user's past sessions (each a list of steps: query, prior calls with results, tool, args,
    future). Counts first: transitions, value examples, include rates. Then ROUNDS boosting rounds on the base ranker,
    slot and include models with the same steps (CPU, seconds). Always starts again from the base models."""
    st = stats(convs)
    Z.TOOL_EXAMPLES = {**BASE_EXAMPLES, **{t: {**BASE_EXAMPLES.get(t, {}), **e} for t, e in st[2].items()}}
    BASE.setdefault("slot", Z.SLOT), BASE.setdefault("inc", Z.INC), BASE.setdefault("rank", RANKERS[SESSION])
    Z.SLOT, Z.INC, RANKERS[SESSION] = BASE["slot"], BASE["inc"], BASE["rank"]
    exs = [x for c in convs for x in c]
    if not exs:
        return st
    t0 = time.time()
    P = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1, "num_threads": 8,
         "deterministic": True, "force_row_wise": True}  # same history, same post-trained models
    Xi, yi = [], []
    for ex in exs:  # include rows score values with the base slot model, as zpred's do with an out-of-fold one
        t = Z.CAT.tools[Z.CAT.pos[ex["tool"]]]
        Xi += Z.arg_options(t, Z.context_values(ex), inc=False)[1]
        yi += [a["name"] in ex["args"] for a in t["args"]]
    X, y = Z.slot_data(Z.CAT, exs)
    Z.SLOT = lgb.train(P, lgb.Dataset(X, y, feature_name=Z.SLOT_FEATS), ROUNDS, init_model=BASE["slot"])
    Z.INC = lgb.train(P, lgb.Dataset(np.array(Xi, np.float32), np.array(yi, np.int8), feature_name=Z.INC_FEATS),
                      ROUNDS, init_model=BASE["inc"])
    Xr, yr, g = [], [], []
    for ex in exs:
        r = features(ex, st)
        if r is None:
            continue
        docs, F = r
        later = {Z.CAT.pos[t] for t, _ in ex["future"]}
        Xr.append(F)
        yr.append([2 if d == Z.CAT.pos[ex["tool"]] else 1 if d in later else 0 for d in docs])
        g.append(len(docs))
    RANKERS[SESSION] = lgb.train({**P, "objective": "lambdarank", "metric": "ndcg"},
                                 lgb.Dataset(np.concatenate(Xr), np.concatenate(yr), group=g, feature_name=Z.FEATS),
                                 ROUNDS, init_model=BASE["rank"])
    print(f"post-trained on {len(convs)} sessions ({len(exs)} calls) in {time.time() - t0:.1f}s", flush=True)
    return st


def job(j):
    cs, h = j
    st = stats(cs[:h])
    Z.TOOL_EXAMPLES = {**BASE_EXAMPLES, **{t: {**BASE_EXAMPLES.get(t, {}), **e} for t, e in st[2].items()}}
    out = []
    for steps in cs[MIN_HIST:]:
        for x in steps:
            t0 = time.perf_counter()
            pool = predict(x, st)
            if pool:
                picks = pool[:Z.SPEC]
                right = any(tid == ft and Z.same_args(a, fa) for _, tid, a in picks for ft, fa in x["future"])
                out.append((float(np.logaddexp.reduce([c[0] for c in picks])), right, (time.perf_counter() - t0) * 1000, "first" if not x["prior"] else "later"))
    Z.TOOL_EXAMPLES = BASE_EXAMPLES
    return out


def evaluate(us, h, workers):
    with mp.get_context("fork").Pool(workers) as pool:
        return [r for rs in pool.map(job, [(cs, h) for cs in us.values()], chunksize=1) for r in rs]


def summary(rs, th=None):
    conf = np.array([r[0] for r in rs])
    right = np.array([r[1] for r in rs])
    out = {"steps": len(rs), "call_top3": round(float(right.mean()), 4),
           "first": round(float(np.mean([r[1] for r in rs if r[3] == "first"])), 4),
           "later": round(float(np.mean([r[1] for r in rs if r[3] == "later"])), 4),
           "ms_p50": round(float(np.median([r[2] for r in rs])), 1)}
    if th is None:  # thresholds on this (dev) set: the lowest confidence whose prefix keeps each precision target
        o = np.argsort(-conf)
        cum = np.cumsum(right[o]) / np.arange(1, len(o) + 1)
        th = {}
        for tg in (0.7, 0.8, 0.9):
            ok = np.nonzero(cum >= tg)[0]
            ok = ok[ok >= 50]
            th[tg] = float(conf[o[ok[-1]]]) if len(ok) else 1e9
    for tg, t in th.items():
        f = conf >= t
        out[f"@{tg}"] = {"coverage": round(float(f.mean()), 4), "precision": round(float(right[f].mean()) if f.any() else 0, 4),
                         "served": round(float((f & right).mean()), 4)}
    return out, th


def main():
    global ALPHA, SESSION
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="session,catalog")
    ap.add_argument("--hist", default="0,5,10,20")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    dev, test = users("dev"), users("test")
    keep = {t for us in (dev, test) for cs in us.values() for steps in cs for x in steps
            for t in (*x["session"], x["tool"])}
    Z.CAT = Z.Catalog(emb=False, keep=keep)
    Z.SLOT = lgb.Booster(model_file=str(Z.OUT / "slots.txt"))
    Z.INC = lgb.Booster(model_file=str(Z.OUT / "include.txt"))
    Z.SPEC = 3
    n = lambda us: sum(len(steps) for cs in us.values() for steps in cs[MIN_HIST:])  # noqa: E731
    print(f"users: dev {len(dev)} ({n(dev)} eval steps), test {len(test)} ({n(test)} eval steps)", flush=True)
    res = {"min_hist": MIN_HIST}
    for setting in a.settings.split(","):
        SESSION = setting == "session"
        RANKERS[SESSION] = lgb.Booster(model_file=str(Z.OUT / f"ranker_{setting}.txt"))
        Z.TEMP = json.load(open(Z.OUT / f"gate_{setting}.json"))["temp"]
        res[setting] = {}
        for h in map(int, a.hist.split(",")):
            best = None
            for al in (map(float, a.alphas.split(",")) if h else [1.0]):
                ALPHA = al
                d, th = summary(evaluate(dev, h, a.workers))
                if best is None or d["call_top3"] > best[1]["call_top3"]:
                    best = (al, d, th)
            ALPHA, d, th = best
            t, _ = summary(evaluate(test, h, a.workers), th)
            res[setting][h] = {"alpha": ALPHA, "dev": d, "test": t}
            print(setting, f"history {h}", json.dumps(res[setting][h]["test"]), f"alpha {ALPHA}", flush=True)
    json.dump(res, open(Z.OUT / "adapt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
