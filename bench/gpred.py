"""Domain-agnostic learned branch predictor, trained and scored on real chat transcripts.

Usage: python3 -I bench/gpred.py <tau-bench> <experiment> [out.json]
  within        per domain, 5-fold by customer, all agent models pooled
  cross_model   per domain, train on the other agent models, test on the held-out one (customers split too)
  cross_domain  generic features only; train on two domains, test on the third

Generalizes lpred.py (retail, one argument per tool) to any tool set: tools and argument names are
learned from the traces, and the model scores argument slots. Each row is (tool, argument, value,
where the value came from); a single-argument call fires per value, a multi-argument call (e.g.
search_direct_flight(origin, destination, date)) fires with the best value per slot and the weakest
slot's score. Values come from earlier tool results and from what the customer has said so far,
never from later messages. A per-tool Wilson gate (TARGET) is picked out of fold on training data.
With timestamps (tau2), each covered call also gets its lead time: how many seconds before the
agent asked for it the prediction could have started.
"""

import json
import os
import re
import statistics as st
import sys
import zlib
from collections import Counter

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tau_replay as T  # noqa: E402  (walk/shape/mentioned/wilson; reads sys.argv[1] = tau-bench root)
import traces as TR  # noqa: E402

TARGET = float(os.environ.get("TARGET", "0.85"))
NWORDS = 64
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, verbose=-1, num_threads=4, seed=0)
ROUNDS = int(os.environ.get("ROUNDS", "300"))
NEG = float(os.environ.get("NEG", "0.2"))
CACHE_TARGET = float(os.environ.get("CACHE_TARGET", "0.5"))
# Structural switches searched by improve.py; defaults reproduce the shipped model.
CONFIG = {"words": True, "cats": True, "siblings": False, "ambiguity": False, "leaves": 63}
OPTIONAL = {"siblings": ["listed beside it, already opened", "share of listed records already opened"],
            "ambiguity": ["candidates for this argument", "source places offering candidates"]}
TOKEN = re.compile(r"[#\w][\w@.+:/-]{2,}")
CAT = [0, 1, 2]
GENERIC = ["field name = argument name", "field/argument name overlap", "field/tool name overlap", "source/tool name overlap",
           "value is a record key", "field depth", "values listed beside it", "results since source", "places value appears",
           "value length", "customer mentioned the record", "customer said the value", "value shape fits argument",
           "value taken from customer's words", "value is numeric", "customer named it (alias table)", "date in the customer's words"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MON = "(" + "|".join(MONTHS) + r")[a-z]*\.?"


def is_code(x):
    """An identifier rather than a name: upper-case code (SFO) or anything with digits."""
    return bool(re.fullmatch(r"[A-Z0-9#_-]{2,12}", x) or re.search(r"\d", x))


def dates(low, year):
    """Dates the customer wrote in words or m/d, as ISO strings in the year the training data uses."""
    if not year:
        return []
    found = {(MONTHS.index(m[1][:3]) + 1, int(m[2])) for m in re.finditer(r"\b" + MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b", low)}
    found |= {(MONTHS.index(m[2][:3]) + 1, int(m[1])) for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON, low)}
    found |= {(int(m[1]), int(m[2])) for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", low)}
    return [f"{year}-{mo:02d}-{d:02d}" for mo, d in found if 1 <= mo <= 12 and 1 <= d <= 31]


def key(name, args):
    return name + json.dumps(args, sort_keys=True)


def toks(s):
    return {w.rstrip("s") for w in re.split(r"[^a-z]+", s.lower()) if len(w) > 1 and w not in ("id", "k", "get", "detail")}


def h(*parts, n=4093):
    return zlib.crc32("|".join(map(str, parts)).encode()) % n


def scalar_args(args):
    return isinstance(args, dict) and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in args.values())


class Featurizer:
    def __init__(self, traces, generic=False):
        self.generic = generic
        sigs, self.shapes, words = {}, {}, Counter()
        for t in traces:
            for s in t["steps"]:
                if s["read"] and s["ok"] and scalar_args(s["args"]):
                    sigs.setdefault(s["name"], Counter())[tuple(sorted(s["args"]))] += 1
                    for a, v in s["args"].items():
                        self.shapes.setdefault(f"{s['name']}|{a}", Counter())[T.shape(str(v), 1)] += 1
            words.update(set(re.findall(r"[a-z]{4,}", t["steps"][-1]["text"].lower())))
        self.args = {n: c.most_common(1)[0][0] for n, c in sigs.items()}
        self.tools = sorted(self.args)
        self.vocab = [] if generic or not CONFIG["words"] else [w for w, _ in words.most_common(NWORDS)]
        self.learn_lexicon(traces)

    def learn_lexicon(self, traces):
        """Aliases for values that are never said verbatim (a customer says "Seattle", the tool takes
        "SEA"; "t-shirt" -> its product id), read from table-shaped tool results seen in training:
        flat dicts of 5+ string entries pairing a code/id with a name ({code: city}, {name: id}).
        Also the year that ISO dates in arguments use."""
        self.lexicon, years = {}, Counter()
        for t in traces:
            for s in t["steps"]:
                if not (s["read"] and s["ok"]):
                    continue
                for v in (s["args"] or {}).values() if scalar_args(s["args"]) else []:
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                        years[str(v)[:4]] += 1
                r = s["result"]
                if isinstance(r, dict) and len(r) >= 5 and all(isinstance(x, str) for x in r.values()):
                    for k, v in r.items():
                        code, name = (k, v) if is_code(k) else (v, k)
                        if is_code(code) and not is_code(name) and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,}", name):
                            self.lexicon[name.lower()] = code
        self.year = years.most_common(1)[0][0] if years else None

    def names(self):
        """Feature names, in row order, for explanations."""
        return (["tool", "source tool", "source field"] + GENERIC + ["results so far", "calls to this tool", "number of arguments"]
                + ([] if self.generic else [f"calls to {x}" for x in self.tools] + [f'customer said "{w}"' for w in self.vocab])
                + [n for k, ns in OPTIONAL.items() if CONFIG[k] for n in ns])

    def meta(self):
        return {"generic": self.generic, "args": self.args, "shapes": {k: dict(v) for k, v in self.shapes.items()}, "vocab": self.vocab,
                "lexicon": self.lexicon, "year": self.year}

    @classmethod
    def load(cls, meta):
        fz = cls([], meta["generic"])
        fz.args = {k: tuple(v) for k, v in meta["args"].items()}
        fz.tools, fz.vocab = sorted(fz.args), meta["vocab"]
        fz.shapes = {k: Counter(v) for k, v in meta["shapes"].items()}
        fz.lexicon, fz.year = meta.get("lexicon", {}), meta.get("year")
        return fz

    def rows(self, text, results, called):
        """[(tool, arg, value, features)] for every argument slot of every known read tool."""
        low = text.lower()
        leaves = [(j, name, pat, v, lab) for j, (name, res) in enumerate(results) for pat, v, lab in T.walk(res)]
        where = Counter(x[3] for x in leaves)
        sib = Counter((x[0], x[2]) for x in leaves)
        src = [x + (0, "") for x in leaves] + [(-1, "TEXT", ("text",), m.group().rstrip(".,"), (), 0, "") for m in TOKEN.finditer(text)]
        src += [(-3, "TEXT", ("date",), d, (), 0, "") for d in dates(low, self.year)]
        src += [(-2, "TEXT", ("alias",), code, (), 1.0, name) for name, code in self.lexicon.items()
                if re.search(r"\b" + re.escape(name) + r"\b", low)]
        n, out = len(results), []
        hist = Counter(name for name, _ in results)
        spec = [] if self.generic else [hist[x] for x in self.tools] + [w in low for w in self.vocab]
        for tool in self.tools:
            args, ttoks = self.args[tool], toks(tool)
            base = [n, hist[tool], len(args)] + spec
            if not args:
                if key(tool, {}) not in called:
                    out.append((tool, "", "", [0 if self.generic else h(tool), 0, 0] + [0] * 17 + base
                                + [0] * sum(len(ns) for k, ns in OPTIONAL.items() if CONFIG[k]), "conversation stage"))
                continue
            if CONFIG["siblings"] or CONFIG["ambiguity"]:
                opened, slot_n, slot_src = Counter(), Counter(), Counter()
                for a in args:
                    shp = self.shapes.get(f"{tool}|{a}", Counter())
                    ok_src = [(j, pat, v) for j, _, pat, v, *_ in src if shp.get(T.shape(v, 1))]
                    slot_n[(tool, a)] = len({v for *_, v in ok_src})
                    slot_src[(tool, a)] = len({(j, pat) for j, pat, _ in ok_src})
                    for j, pat, v in ok_src:
                        if j >= 0 and key(tool, {a: v}) in called:
                            opened[(tool, a, j, pat)] += 1
            for a in args:
                shp, atoks = self.shapes.get(f"{tool}|{a}", Counter()), toks(a)
                total = sum(shp.values()) or 1
                for j, stool, pat, v, lab, conf, phrase in src:
                    freq = shp.get(T.shape(v, 1), 0)
                    if not freq or (len(args) == 1 and key(tool, {a: v}) in called):
                        continue
                    steps = [p for p in pat if p not in ("*", "{k}")]
                    ptoks = toks(" ".join(steps))
                    cats = [0, 0, 0] if self.generic or not CONFIG["cats"] else [h(tool), h(stool), h(tool, a, stool, pat)]
                    opt = []
                    if CONFIG["siblings"]:  # the agent opened k of the n records listed together with this one
                        done_ = opened.get((tool, a, j, pat), 0) if j >= 0 else 0
                        opt += [done_, done_ / max(sib[(j, pat)], 1) if j >= 0 else 0]
                    if CONFIG["ambiguity"]:
                        opt += [slot_n[(tool, a)], slot_src[(tool, a)]]
                    out.append((tool, a, v, cats + [
                        int(bool(steps) and steps[-1] == a), len(atoks & ptoks) / max(len(atoks), 1), len(ttoks & ptoks) / max(len(ttoks), 1),
                        len(ttoks & toks(stool)) / max(len(ttoks), 1), int(bool(pat) and pat[-1] == "{k}"), len(pat),
                        sib[(j, pat)] if j >= 0 else 0, n - 1 - j if j >= 0 else -1, where[v], len(v),
                        int(T.mentioned(low, lab)), int(v.lower() in low), freq / total, int(j < 0), int(v.replace(".", "").isdigit()),
                        conf, int(j == -3)] + base + opt,
                        f'the customer saying "{phrase}"' if j == -2 else "a date the customer gave" if j == -3 else
                        "the customer's words" if j < 0 else f"{stool} → {'.'.join(p for p in pat if p != '*') or 'result'}"))
        return out


def calls(fz, rows, scores, called):
    """Combine slot scores into call keys -> score."""
    best, out = {}, {}
    for (tool, a, v, *_), s in zip(rows, scores):
        if len(fz.args[tool]) <= 1:
            k = key(tool, {a: v} if a else {})
            if k not in called:
                out[k] = max(out.get(k, 0.0), s)
        elif s > best.get((tool, a), ("", -1))[1]:
            best[(tool, a)] = (v, s)
    for tool in fz.tools:
        args = fz.args[tool]
        if len(args) > 1 and all((tool, a) in best for a in args):
            k = key(tool, {a: best[(tool, a)][0] for a in args})
            if k not in called:
                out[k] = min(best[(tool, a)][1] for a in args)
    return out


def steps(t):
    """Before each successful read call, including the first: (what the customer had said by then,
    results so far, keys called, future read keys, this call's key, when the prediction could start,
    call time). The predictor runs on every tool result and every customer message, so it can start
    at whichever of the two came last."""
    reads = [s for s in t["steps"] if s["read"] and s["ok"] and scalar_args(s["args"])]
    for i, s in enumerate(reads):
        avail = [x for x in (reads[i - 1]["t_result"] if i else None, s.get("t_text")) if x is not None]
        yield (s["text"], [(r["name"], r["result"]) for r in reads[:i]], {key(r["name"], r["args"]) for r in reads[:i]},
               {key(r["name"], r["args"]) for r in reads[i:]}, key(s["name"], s["args"]), max(avail) if avail else None, s["t_call"])


def slot_hit(tool, a, v, future):
    return any(k.startswith(tool + "{") and (not a or str(json.loads(k[len(tool):]).get(a)) == v) for k in future)


def dataset(fz, traces):
    """Rows for every slot candidate. Negatives (~98%) are subsampled at NEG and weighted 1/NEG,
    which keeps scores calibrated while cutting memory and training time."""
    X, y, rng = [], [], np.random.default_rng(0)
    for t in traces:
        for text, results, called, future, *_ in steps(t):
            for tool, a, v, f, _ in fz.rows(text, results, called):
                pos = slot_hit(tool, a, v, future)
                if pos or rng.random() < NEG:
                    X.append(f)
                    y.append(int(pos))
    y = np.array(y)
    return np.array(X, dtype=np.float32), y, np.where(y == 1, 1.0, 1 / NEG)


def fit(traces, generic=False):
    fz = Featurizer(traces, generic)
    X, y, w = dataset(fz, traces)
    return fz, lgb.train({**PARAMS, "num_leaves": CONFIG["leaves"]}, lgb.Dataset(X, y, weight=w, categorical_feature=CAT), ROUNDS)


def trace(fz, model, t):
    """best: key -> [(score, time available)] over steps; at_call: (tool, [(score, available)], call time) per read."""
    best, at_call = {}, []
    reads = [s for s in t["steps"] if s["read"] and s["ok"] and scalar_args(s["args"])]
    actual = {key(s["name"], s["args"]) for s in reads}
    for text, results, called, _, nxt, avail, t_call in steps(t):
        rows = fz.rows(text, results, called)
        if rows:
            for k, s in calls(fz, rows, model.predict(np.array([r[3] for r in rows], dtype=np.float32)), called).items():
                best.setdefault(k, []).append((s, avail))
        at_call.append((nxt[:nxt.index("{")], list(best.get(nxt, [])), t_call))
    return {"best": best, "at_call": at_call, "actual": actual, "calls": t["ncalls"]}


def score(pairs):
    """pairs: [(trace, gates)]; gates: tool -> threshold, "*" for any tool (cross-domain); missing tools never fire."""
    pred = hit = cov = reads = calls_ = 0
    lead = []
    for tr, th in pairs:
        gate = lambda k: th.get(k.split("{")[0], th.get("*", 2))  # noqa: E731
        for k, ss in tr["best"].items():
            if max(s for s, _ in ss) >= gate(k):
                pred += 1
                hit += k in tr["actual"]
        for tool, ss, t_call in tr["at_call"]:
            first = next((a for s, a in ss if s >= gate(tool)), "none")
            if first != "none":
                cov += 1
                if first is not None and t_call is not None:
                    lead.append(t_call - first)
        reads += len(tr["at_call"])
        calls_ += tr["calls"]
    return {"precision": hit / max(pred, 1), "coverage": cov / max(reads, 1), "calls_saved": cov / max(calls_, 1),
            "predicted": pred, "hit": hit, "covered": cov, "reads": reads, "calls": calls_,
            "wasted_per_chat": round((pred - hit) / max(len(pairs), 1), 2),
            "lead_s_median": round(st.median(lead), 2) if lead else None, "lead_s_mean": round(st.mean(lead), 2) if lead else None}


def pick(trs, tools, target=None):
    """Per-tool Wilson gate; tools=None picks one threshold for every tool (needed for unseen tools)."""
    target = TARGET if target is None else target
    th = {}
    for tool in tools or ["*"]:
        for x in np.arange(0.05, 1.0, 0.01):
            s = score([(tr, {tool: x}) for tr in trs])
            if s["predicted"] >= 20 and T.wilson(s["hit"], s["predicted"], 1.645) >= target:
                th[tool] = round(float(x), 2)
                break
    return th


def folds(trs, k):
    groups = sorted({t["group"] for t in trs})
    f = {g: i % k for i, g in enumerate(groups)}
    return [([t for t in trs if f[t["group"]] != i], [t for t in trs if f[t["group"]] == i]) for i in range(k)]


def train_gated(pool, generic=False, inner=3):
    oof = []
    for tr, te in folds(pool, inner):
        fz, m = fit(tr, generic)
        oof += [trace(fz, m, t) for t in te]
    fz, m = fit(pool, generic)
    tools = None if generic else fz.tools
    return fz, m, {"inject": pick(oof, tools), "cache": pick(oof, tools, CACHE_TARGET)}


def run(train, test, generic=False):
    """Gates {"inject": strict, "cache": loose} and the held-out traces."""
    fz, m, th = train_gated(train, generic)
    return th, [trace(fz, m, t) for t in test]


def explain(fz, row, contrib, score):
    """Plain-language reasons for one slot candidate: where the value came from and the features that
    pushed the score up most (LightGBM SHAP contributions)."""
    tool, a, v, f, src = row
    names = fz.names()
    names[1:3] = [f"value came from {src.split(' → ')[0]}", f"value came from {src}"]
    top = sorted(((c, n) for c, n in zip(contrib[:-1], names) if c > 0 and n != "tool"), reverse=True)[:3]
    return {"slot": f"{tool}.{a}" if a else tool, "value": v, "score": round(float(score), 3), "from": src,
            "reasons": [n for _, n in top]}


class Predictor:
    """Live use: predict(text, results, called) -> [{"key", "score", "inject", "cache", "why"}].
    Two gates: inject (strict) results go into the agent's context; cache (loose) results are only
    fetched in the background, so an unsure guess costs one wasted read, never a wrong context."""

    def __init__(self, path):
        meta = json.load(open(path + ".json"))
        self.fz = Featurizer.load(meta["featurizer"])
        self.inject, self.cache = meta["inject"], meta["cache"]
        self.model = lgb.Booster(model_file=path + ".txt")

    def predict(self, text, results, called, explain=True):
        """explain=False returns at once with "why" empty; call explain_last() afterwards (off the
        critical path: SHAP costs ~6 ms per explained guess, scoring all candidates ~10 ms)."""
        rows = self.fz.rows(text, results, called)
        if not rows:
            return []
        X = np.array([r[3] for r in rows], dtype=np.float32)
        sc = self.model.predict(X, num_threads=1)  # tiny batch: threads cost more than they save
        fired = []
        for k, s in calls(self.fz, rows, sc, called).items():
            tool = k.split("{")[0]
            if s < self.cache.get(tool, 2) and s < self.inject.get(tool, 2):
                continue
            args = json.loads(k[len(tool):])
            idx = [max((i for i, r in enumerate(rows) if r[0] == tool and r[1] == a and r[2] == str(v)), key=lambda i: sc[i])
                   for a, v in args.items()] or [next(i for i, r in enumerate(rows) if r[0] == tool)]
            fired.append((k, s, tool, idx))
        out = sorted(({"key": k, "score": round(float(s), 3), "inject": s >= self.inject.get(tool, 2), "cache": s >= self.cache.get(tool, 2),
                       "why": [], "_idx": idx} for k, s, tool, idx in fired), key=lambda x: -x["score"])
        self._last = (rows, X, sc, out)
        if explain:
            self.explain_last()
        return out

    def explain_last(self):
        """Fill "why" for the last predict() call. SHAP only for the rows behind guesses that fired."""
        rows, X, sc, out = self._last
        need = sorted({i for g in out for i in g["_idx"]})
        contrib = dict(zip(need, self.model.predict(X[need], pred_contrib=True, num_threads=1))) if need else {}
        for g in out:
            g["why"] = [explain(self.fz, rows[i], contrib[i], sc[i]) for i in g.pop("_idx")]
        return out


def export(here, domain="retail", inject=0.85, cache=0.5):
    """Per-customer-fold live models: fold f never saw the customers in fold f."""
    data = TR.load(domain)
    groups = sorted({t["group"] for t in data})
    fold = {g: i % 5 for i, g in enumerate(groups)}

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(5) as ex:
        for f, i, c in ex.map(one_fold, [(data, fold, f, here, domain, inject, cache) for f in range(5)]):
            print(f"fold {f} inject={i} cache={c}", flush=True)
    json.dump({"fold_of_group": fold}, open(os.path.join(here, f"gpred_{domain}_folds.json"), "w"))


def one_fold(job):
    data, fold, f, here, domain, inject, cache = job
    pool = [t for t in data if fold[t["group"]] != f]
    oof = []
    for tr, te in folds(pool, 3):
        fz, m = fit(tr)
        oof += [trace(fz, m, t) for t in te]
    fz, m = fit(pool)
    path = os.path.join(here, f"gpred_{domain}_fold{f}")
    m.save_model(path + ".txt")
    g = {"featurizer": fz.meta(), "inject": pick(oof, fz.tools, inject), "cache": pick(oof, fz.tools, cache)}  # noqa: E501
    json.dump(g, open(path + ".json", "w"))
    return f, g["inject"], g["cache"]


def parallel(jobs):
    """Run (train, test, generic) jobs in worker processes, one per outer fold."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(len(jobs), 5)) as ex:
        return list(ex.map(run_job, jobs))


def run_job(job):
    return run(*job)


def show(name, s, th=None):
    print(f"{name:42} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%} "
          f"predicted={s['predicted']} reads={s['reads']} lead_median={s['lead_s_median']}s" + (f" gates={th}" if th else ""), flush=True)


if __name__ == "__main__":
    exp = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    doms = os.environ.get("DOMAINS", "retail,airline,telecom").split(",")
    data = {d: TR.load(d) for d in doms}
    report = []
    def both(pairs):
        """Strict gate: lookups attached to the agent's context (calls saved). Either gate: lookups
        ready before the agent asks (strict ones attached, loose ones waiting in the cache)."""
        return {"inject": score([(x, g["inject"]) for x, g in pairs]), "ready": score([(x, g["cache"]) for x, g in pairs])}

    def show2(name, r):
        i, c = r["inject"], r["ready"]
        print(f"{name:42} attached: precision={i['precision']:.0%} coverage={i['coverage']:.0%} calls_saved={i['calls_saved']:.0%} | "
              f"ready ahead: coverage={c['coverage']:.0%} precision={c['precision']:.0%} wasted/chat={c['wasted_per_chat']} lead={c['lead_s_median']}s", flush=True)

    if exp == "within":
        for d in doms:
            pairs, fs = [], folds(data[d], 5)
            for (tr, te), (g, t) in zip(fs, parallel([(tr, te, False) for tr, te in fs])):
                pairs += [(x, g, src["model"]) for x, src in zip(t, te)]
            r = both([(x, g) for x, g, _ in pairs])
            show2(f"{d} within (all models)", r)
            per = {}
            for mname in sorted({p[2] for p in pairs}):
                per[mname] = both([(x, g) for x, g, mm in pairs if mm == mname])
                show2(f"  {mname}", per[mname])
            report.append({"domain": d, "exp": exp, **r, "per_model": per, "gates": pairs[0][1]})
    elif exp == "cross_model":
        for d in doms:
            for held in [m for m in sorted({t["model"] for t in data[d]}) if not os.environ.get("HELD") or m in os.environ["HELD"].split(",")]:
                jobs = [([x for x in tr if x["model"] != held], [x for x in te if x["model"] == held], False) for tr, te in folds(data[d], 5)]
                r = both([(x, g) for g, t in parallel(jobs) for x in t])
                show2(f"{d} train others -> {held}", r)
                report.append({"domain": d, "exp": exp, "held_model": held, **r})
    elif exp == "cross_domain":
        jobs = [([x for o in doms if o != d for x in data[o]], data[d], True) for d in doms]
        for d, (g, t) in zip(doms, parallel(jobs)):
            r = both([(x, g) for x in t])
            show2(f"{'+'.join(o for o in doms if o != d)} -> {d} (unseen tools)", r)
            report.append({"domain": d, "exp": exp, **r, "gates": g})
    elif exp == "export":
        export(os.path.dirname(os.path.abspath(out_path or "bench/results/x")))
    if out_path and exp != "export":
        json.dump(report, open(out_path, "w"), indent=1, default=lambda o: o.item())
