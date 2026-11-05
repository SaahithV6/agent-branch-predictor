"""Learned branch predictor: a LightGBM classifier over argument candidates.

Usage: python3 -I bench/lpred.py <tau-bench> results/live_train.jsonl results/live_test.jsonl [out.json]

Replaces the rule table in tau_replay.py. After every read result, each read tool's argument is
paired with every value in the results so far (and, in text mode, every id-like token in the
customer's request) whose shape matches a value that argument took in training. Each candidate
call is described by generic features (does the field name match the argument name, how recent
the result is, how many siblings the value has, how many places the same value appears, call
history, request words) plus a hashed "signature" (tool, argument, source tool, field path), which
is what the rule table was. Label: the agent makes that call later in the task. A confidence gate
picks the lowest threshold whose Wilson lower bound on precision clears TARGET, using out-of-fold
replays, so precision is set by the data rather than tuned by hand.

Protocol matches tau_live_nested.py: test users in the same 5 outer folds; each fold trains on all
train traces plus the other 4 folds, picks its threshold by inner 5-fold CV, and is replayed on
users it never saw. Per-fold models are saved for the live A/B (tau_mcp.py loads them).
"""

import inspect
import json
import os
import re
import sys
import zlib
from collections import Counter

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tau_replay as T  # noqa: E402

DOM = "retail"
TOOLS = sorted(T.READS[DOM])
ARGS = {n: list(inspect.signature(f).parameters) for n, f in T.READS[DOM].items()}
TARGET = float(os.environ.get("TARGET", "0.95"))
NWORDS = 64
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, verbose=-1, num_threads=4, seed=0)
ROUNDS = 300
TOKEN = re.compile(r"#?[A-Za-z0-9][\w@.+-]{3,}")


def toks(s):
    return {w.rstrip("s") for w in re.split(r"[^a-z]+", s.lower()) if len(w) > 1 and w not in ("id", "k")}


def h(*parts, n=4093):
    return zlib.crc32("|".join(map(str, parts)).encode()) % n


class Featurizer:
    """Holds what is learned from the training set besides the trees: argument value shapes and
    the request vocabulary."""

    def __init__(self, tasks, text):
        self.text = text
        self.shapes = {}
        words = Counter()
        for t in tasks:
            words.update(set(re.findall(r"[a-z]{4,}", t["lower"])))
            for n, a in t["actions"]:
                if n in T.READS[DOM] and len(ARGS[n]) == 1:
                    self.shapes.setdefault(n, Counter())[T.shape(str(a[ARGS[n][0]]), 1)] += 1
        self.vocab = [w for w, _ in words.most_common(NWORDS)] if text else []

    def state(self, task):
        return {"words": [w in task["lower"] for w in self.vocab],
                "text": [m.group().rstrip(".,") for m in TOKEN.finditer(task["instruction"])] if self.text else []}

    def candidates(self, task, st, results, called):
        """Yield (call_key, feature row)."""
        leaves = [(j, pat, v, lab) for j, (_, res) in enumerate(results) for pat, v, lab in T.walk(res)]
        where = Counter(v for _, _, v, _ in leaves)
        sib = Counter((j, pat) for j, pat, _, _ in leaves)
        hist = [sum(n == x for n, _ in results) for x in TOOLS]
        base = [len(results)] + hist + st["words"]
        src = [(j, results[j][0], pat, v, lab) for j, pat, v, lab in leaves]
        src += [(-1, "TEXT", ("text",), v, ()) for v in st["text"]]
        for tool in TOOLS:
            args = ARGS[tool]
            if not args:
                k = T.call_key(tool, {})
                if k not in called:
                    yield k, [TOOLS.index(tool), -1, h(tool), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + base
                continue
            if len(args) > 1 or tool not in self.shapes:
                continue
            arg, shp = args[0], self.shapes[tool]
            atoks, total = toks(arg), sum(shp.values())
            for j, stool, pat, v, lab in src:
                freq = shp.get(T.shape(v, 1), 0)
                if not freq:
                    continue
                k = T.call_key(tool, {arg: v})
                if k in called:
                    continue
                last = next((p for p in reversed(pat) if p not in ("*", "{k}")), "")
                ptoks = toks(" ".join(p for p in pat if p not in ("*", "{k}")))
                yield k, [TOOLS.index(tool), TOOLS.index(stool) if stool in TOOLS else -1, h(tool, arg, stool, pat),
                          int(last == arg), len(atoks & ptoks) / max(len(atoks), 1), int(bool(pat) and pat[-1] == "{k}"), len(pat),
                          sib[(j, pat)] if j >= 0 else 0, len(results) - 1 - j if j >= 0 else -1, where[v],
                          len(v), int(T.mentioned(task["lower"], lab)), int(v in task["instruction"]), freq / total] + base


CAT = [0, 1, 2]


def steps(task):
    """Yield (results so far, keys already called, future read keys) before each read call and once at the end."""
    results, called = [], set()
    reads = [(n, a) for n, a in task["actions"] if n in T.READS[DOM]]
    for i, (n, a) in enumerate(reads):
        yield results, called, {T.call_key(x, y) for x, y in reads[i:]}, T.call_key(n, a)
        called = called | {T.call_key(n, a)}
        results = results + [(n, T.READS[DOM][n](**a))]
    yield results, called, set(), None


def rows(fz, tasks):
    X, y = [], []
    for t in tasks:
        st = fz.state(t)
        for results, called, future, _ in steps(t):
            if not results:
                continue  # the live server predicts only after a read result
            for k, f in fz.candidates(t, st, results, called):
                X.append(f)
                y.append(int(k in future))
    return np.array(X, dtype=float), np.array(y)


def fit(tasks, text):
    fz = Featurizer(tasks, text)
    X, y = rows(fz, tasks)
    return fz, lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=CAT), ROUNDS)


def trace(fz, model, t):
    """Per task: best score each predicted key ever got, and for each read call the best score it
    had before the agent made it. Any threshold can then be scored without re-running the model."""
    st, best, at_call, actual = fz.state(t), {}, [], set()
    for results, called, _, nxt in steps(t):
        if results:
            cands = list(fz.candidates(t, st, results, called))
            if cands:
                for (k, _), s in zip(cands, model.predict(np.array([f for _, f in cands], dtype=float))):
                    best[k] = max(best.get(k, 0.0), s)
        if nxt:
            at_call.append((tool_of(nxt), best.get(nxt, 0.0)))
            actual.add(nxt)
    return {"best": best, "at_call": at_call, "actual": actual, "calls": len(t["actions"])}


def tool_of(k):
    return k[:k.index("{")]


def score(traces, th):
    """th: tool -> threshold; tools without one never fire."""
    on = lambda tool, s: s >= th.get(tool, 2)  # noqa: E731
    pred = sum(on(tool_of(k), s) for tr in traces for k, s in tr["best"].items())
    hit = sum(on(tool_of(k), s) and k in tr["actual"] for tr in traces for k, s in tr["best"].items())
    cov = sum(on(tool, s) for tr in traces for tool, s in tr["at_call"])
    reads, calls = sum(len(tr["at_call"]) for tr in traces), sum(tr["calls"] for tr in traces)
    return {"precision": hit / max(pred, 1), "coverage": cov / max(reads, 1), "calls_saved": cov / max(calls, 1),
            "predicted": pred, "hit": hit, "covered": cov, "reads": reads, "calls": calls}


def wilson(hit, n, z=1.645):
    return T.wilson(hit, n, z)


def pick(traces):
    """Per tool, the lowest threshold whose one-sided 95% Wilson bound on precision clears TARGET
    (one gate per argument slot: a noisy slot can't drag down a reliable one). Tools that never
    clear it, or fire fewer than 20 times out of fold, are switched off."""
    th = {}
    for tool in TOOLS:
        for x in np.arange(0.05, 1.0, 0.01):
            s = score(traces, {tool: x})
            if s["predicted"] >= 20 and wilson(s["hit"], s["predicted"]) >= TARGET:
                th[tool] = round(float(x), 2)
                break
    return th, score(traces, th)


def folds(tasks, k=5):
    users = sorted({t["user"] for t in tasks})
    f = {u: i % k for i, u in enumerate(users)}
    return [([t for t in tasks if f[t["user"]] != i], [t for t in tasks if f[t["user"]] == i]) for i in range(k)]


def choose(pool, text):
    oof = []
    for tr, te in folds(pool):
        fz, m = fit(tr, text)
        oof += [trace(fz, m, t) for t in te]
    return pick(oof)


def save(fz, model, th, path):
    model.save_model(path + ".txt")
    json.dump({"threshold": th, "text": fz.text, "shapes": {k: dict(v) for k, v in fz.shapes.items()}, "vocab": fz.vocab},
              open(path + ".json", "w"))


class Predictor:
    """Live use (tau_mcp.py): predict(task, results, called) -> call keys to prefetch."""

    def __init__(self, meta, model_file):
        self.fz = Featurizer([], meta["text"])
        self.fz.shapes = {k: Counter(v) for k, v in meta["shapes"].items()}
        self.fz.vocab = meta["vocab"]
        self.th = meta["threshold"]
        self.model = lgb.Booster(model_file=model_file)
        self.st = None

    def predict(self, task, results, called, scores=False):
        if self.st is None:
            self.st = self.fz.state(task)
        cands = list(self.fz.candidates(task, self.st, results, called))
        if not cands:
            return []
        s = self.model.predict(np.array([f for _, f in cands], dtype=float))
        out = [(k, float(p)) for (k, _), p in sorted(zip(cands, s), key=lambda x: -x[1]) if p >= self.th.get(tool_of(k), 2)]
        return out if scores else [k for k, _ in out]


if __name__ == "__main__":
    from tau_live_eval import load, safe
    train, test = safe(load(sys.argv[2])), safe(load(sys.argv[3]))
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    fold_of = json.load(open(os.path.join(os.path.dirname(sys.argv[3]), "live_nested.json")))["fold_of_user"]
    here = os.path.dirname(sys.argv[3])
    report = {"target": TARGET, "folds": []}
    tag = os.environ.get("TAG", "")
    for mode, text in [m for m in (("strict", False), ("text", True)) if m[0] in os.environ.get("MODES", "strict,text")]:
        for f in range(5):
            held = [t for t in test if fold_of[t["user"]] == f]
            pool = train + [t for t in test if fold_of[t["user"]] != f]
            th, inner = choose(pool, text)
            fz, m = fit(pool, text)
            tr = [trace(fz, m, t) for t in held]
            s = score(tr, th)
            save(fz, m, th, os.path.join(here, f"lgbm{tag}_{mode}_fold{f}"))
            report["folds"].append({"mode": mode, "fold": f, "threshold": th, "inner": inner, "held": s})
            print(f"fold {f} {mode:6} th={th} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%}", flush=True)
        tot = [r["held"] for r in report["folds"] if r["mode"] == mode]
        agg = {k: sum(x[k] for x in tot) for k in ("predicted", "hit", "covered", "reads", "calls")}
        all_ = {"precision": agg["hit"] / max(agg["predicted"], 1), "coverage": agg["covered"] / agg["reads"], "calls_saved": agg["covered"] / agg["calls"], **agg}
        report[mode] = all_
        print(f"ALL  {mode:6} precision={all_['precision']:.0%} coverage={all_['coverage']:.0%} calls_saved={all_['calls_saved']:.0%} predicted={agg['predicted']} reads={agg['reads']} calls={agg['calls']}", flush=True)
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1, default=lambda o: o.item())
