"""Offline replay of the branch predictor on tau-bench ground-truth trajectories.

Usage: python3 -I bench/tau_replay.py <tau-bench checkout> [out.json]

tau-bench (sierra-research/tau-bench) gives each test task the exact tool calls a correct
agent makes. We replay those calls against the benchmark's own JSON data, learn argument-
source rules ("get_order_details.order_id comes from get_user_details -> orders[*]"), and
score predictions on tasks the rules never saw:

  precision   = predicted read calls the agent then made / all predicted read calls
  coverage    = read calls already predicted when the agent made them / all read calls
  calls_saved = covered read calls / all tool calls (calls the model would not have to issue)

Protocol: users are split by crc32 into a tuning half and a holdout half. Every design choice
was made with 5-fold cross-validation inside the tuning half (bench/tau_diag.py); the holdout
half is scored once, with rules learned on the tuning half only.

Modes. "strict" predicts from earlier tool results only. "text" may also read the customer's
words, approximated by the task instruction (the real agent sees them piecemeal through the
conversation, so text-mode numbers are optimistic). Writes are never predicted.

Task files are parsed with ast (never imported); tool results are recomputed from the data.
v1 (single-shape text matching, no mention filter, 5-fold CV over all users) is
bench/tau_replay_v1.py.
"""

import ast
import itertools
import json
import os
import re
import sys
import zlib
from collections import defaultdict

root = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None


def load_tasks(path):
    tasks = []
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Task":
            kw = {k.arg: k.value for k in node.keywords}
            actions = []
            for a in kw["actions"].elts:
                akw = {k.arg: ast.literal_eval(k.value) for k in a.keywords}
                actions.append((akw["name"], akw.get("kwargs", {})))
            text = ast.literal_eval(kw["instruction"])
            tasks.append({"user": ast.literal_eval(kw["user_id"]), "instruction": text, "lower": text.lower(), "actions": actions})
    return tasks


def load(dom, *names):
    return {n: json.load(open(os.path.join(root, "tau_bench/envs", dom, "data", f"{n}.json"))) for n in names}


R = load("retail", "users", "orders", "products")
A = load("airline", "users", "reservations", "flights")


def find_email(email):
    return next((u for u, p in R["users"].items() if p["email"].lower() == email.lower()), None)


def find_name_zip(first_name, last_name, zip):
    return next((u for u, p in R["users"].items() if p["name"]["first_name"].lower() == first_name.lower() and p["name"]["last_name"].lower() == last_name.lower() and p["address"]["zip"] == zip), None)


def direct(origin, destination, date):
    return [{**{k: v for k, v in f.items() if k != "dates"}, **f["dates"][date]} for f in A["flights"].values()
            if f["origin"] == origin and f["destination"] == destination and date in f["dates"] and f["dates"][date]["status"] == "available"]


READS = {
    "retail": {
        "find_user_id_by_email": lambda email: find_email(email),
        "find_user_id_by_name_zip": find_name_zip,
        "get_user_details": lambda user_id: R["users"].get(user_id),
        "get_order_details": lambda order_id: R["orders"].get(order_id),
        "get_product_details": lambda product_id: R["products"].get(product_id),
        "list_all_product_types": lambda: {p["name"]: p["product_id"] for p in R["products"].values()},
    },
    "airline": {
        "get_user_details": lambda user_id: A["users"].get(user_id),
        "get_reservation_details": lambda reservation_id: A["reservations"].get(reservation_id),
        "search_direct_flight": direct,
    },
}


def idlike(k):
    return len(k) >= 6 and any(c.isdigit() for c in k)


def walk(obj, path=()):
    """Yield (pattern, value, labels) for every scalar leaf and id-like dict key. List indices and
    id keys become wildcards; labels are the human-readable strings in the same record (an order
    item's "name"), used to check whether the customer mentioned that entry."""
    if isinstance(obj, dict):
        labels = tuple(v.lower() for k, v in obj.items() if isinstance(v, str) and not idlike(v) and len(v) >= 3 and not v.isdigit())
        for k, v in obj.items():
            step = "{k}" if idlike(k) else k
            if idlike(k):
                yield path + ("{k}",), k, labels_of(v)
            if isinstance(v, (dict, list)):
                yield from walk(v, path + (step,))
            elif v is not None:
                yield path + (step,), str(v), labels
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, path + ("*",))
    elif obj is not None:
        yield path, str(obj), ()


def labels_of(v):
    return tuple(x.lower() for x in v.values() if isinstance(x, str) and not idlike(x) and len(x) >= 3 and not x.isdigit()) if isinstance(v, dict) else ()


def mentioned(text, labels):
    # Full label or its head noun ("Mechanical Keyboard" -> "keyboard").
    return any(l in text or (len(l.split()[-1]) >= 4 and l.split()[-1] in text) for l in labels)


CLS = [(str.isdigit, r"\d"), (str.isupper, "[A-Z]"), (str.islower, "[a-z]")]


def shape(v, level):
    """level 0: exact run lengths; 1: runs become +; 2: alphanumerics collapse to one class."""
    out, prev, n = [], None, 0
    for c in v + "\0":
        cls = next((r for f, r in CLS if f(c)), None) or re.escape(c)
        if level == 2 and cls in (r"\d", "[A-Z]", "[a-z]"):
            cls = "[A-Za-z0-9]"
        if cls == prev:
            n += 1
            continue
        if prev and prev != re.escape("\0"):
            out.append(prev + ((f"{{{n}}}" if n > 1 else "") if level == 0 else "+" if n > 1 else ""))
        prev, n = cls, 1
    return "".join(out)


def call_key(name, args):
    return name + json.dumps(args, sort_keys=True)


def sources(results):
    """index[("R", tool, *pattern)] -> {value: labels}."""
    idx = defaultdict(dict)
    for tool, res in results:
        for pat, v, labels in walk(res):
            idx[("R", tool) + pat].setdefault(v, labels)
    return idx


def text_values(task, shp):
    return set(re.findall(r"(?<![\w#@.])" + shp + r"(?![\w@])", task["instruction"]))


def arg_sources(task, idx, args, cfg):
    opts = {}
    for a, v in args.items():
        if not isinstance(v, str):
            return None
        srcs = [s for s, vals in idx.items() if v in vals]
        if cfg["text"] and v in task["instruction"]:
            srcs += [("I", shape(v, lvl)) for lvl in cfg["levels"]]
        if not srcs:
            return None
        opts[a] = srcs
    return opts


def learn(tasks, dom, cfg):
    """Rules: (tool, ((arg, source), ...)) observed in training trajectories, with support counts."""
    rules = defaultdict(int)
    for t in tasks:
        results = []
        for name, args in t["actions"]:
            if name in READS[dom]:
                opts = arg_sources(t, sources(results), args, cfg)
                if opts is not None:
                    for combo in itertools.product(*[[(a, s) for s in ss] for a, ss in sorted(opts.items())]):
                        rules[(name, combo)] += 1
                results.append((name, READS[dom][name](**args)))
    return rules


def candidates(rule, t, idx, cfg):
    """Yield (call_key, bucket). bucket says whether the customer mentioned the entry the call is
    about (None when the rule has no labelled list source or text mode is off)."""
    name, combo = rule
    vals, bucket_arg = [], None
    for a, s in combo:
        if s[0] == "I":
            vs = {v: () for v in text_values(t, s[1])}
        else:
            vs = idx.get(s, {})
            if bucket_arg is None and any(vs.values()):
                bucket_arg = a
        if not vs:
            return
        vals.append([(a, v, labels) for v, labels in sorted(vs.items())])
    for c in itertools.islice(itertools.product(*vals), 50):
        bucket = None
        if cfg["mention"] and bucket_arg is not None:
            bucket = mentioned(t["lower"], next(l for a, v, l in c if a == bucket_arg))
        yield call_key(name, {a: v for a, v, _ in c}), bucket


def words(t):
    return set(re.findall(r"[a-z]{4,}", t["lower"]))


def score_rules(rules, tasks, dom, cfg):
    """(rule, bucket) -> list of per-task (fired, hit, words) on training tasks."""
    stats = defaultdict(list)
    for t in tasks:
        actual = {call_key(n, a) for n, a in t["actions"] if n in READS[dom]}
        results, seen, per = [], set(), defaultdict(lambda: [0, 0])
        for name, args in t["actions"] + [(None, None)]:
            idx = sources(results)
            for r in rules:
                for k, b in candidates(r, t, idx, cfg):
                    if (r, k) not in seen:
                        seen.add((r, k))
                        per[(r, b)][0] += 1
                        per[(r, b)][1] += k in actual
            if name in READS[dom]:
                results.append((name, READS[dom][name](**args)))
        w = words(t)
        for rb, (fired, hit) in per.items():
            stats[rb].append((fired, hit, w))
    return stats


def wilson(hit, n, z):
    """Lower confidence bound of a hit rate: rules seen on few tasks must clear a higher bar."""
    if not n:
        return 0.0
    p = hit / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)


def fit(train, dom, cfg):
    """keep: (rule, bucket) -> (precision, required word or None). With cfg["stump"] (text mode), a
    rule below theta is still kept if one word in the request, seen in >= stump training tasks,
    lifts it to theta (e.g. product lookups only when the customer says "exchange")."""
    rules = learn(train, dom, cfg)
    keep = {}
    for (r, b), per in score_rules(rules, train, dom, cfg).items():
        fired, hit = sum(p[0] for p in per), sum(p[1] for p in per)
        if rules[r] < 2 or not fired:
            continue
        z = cfg.get("z", 0)
        if wilson(hit, fired, z) >= cfg["theta"]:
            keep[(r, b)] = (hit / fired, None)
        elif cfg.get("stump") and cfg["text"]:
            by_word = defaultdict(lambda: [0, 0, 0])
            for f, h, w in per:
                for x in w:
                    by_word[x][0] += f
                    by_word[x][1] += h
                    by_word[x][2] += 1
            ok = [(h / f, x) for x, (f, h, n) in by_word.items() if n >= cfg["stump"] and wilson(h, f, z) >= cfg["theta"]]
            if ok:
                keep[(r, b)] = max(ok)
    return keep


def predict(keep, t, results, cfg):
    """keep: (rule, bucket) -> (training precision, required word or None)."""
    idx, w, out = sources(results), None, set()
    for r in {r for r, _ in keep}:
        for k, b in candidates(r, t, idx, cfg):
            if (r, b) in keep:
                need = keep[(r, b)][1]
                if need is not None:
                    w = w if w is not None else words(t)
                    if need not in w:
                        continue
                out.add(k)
    return out


def replay(keep, t, dom, cfg):
    results, predicted, covered = [], set(), 0
    actual = {call_key(n, a) for n, a in t["actions"] if n in READS[dom]}
    reads = 0
    for name, args in t["actions"]:
        if name in READS[dom]:
            pending = predict(keep, t, results, cfg)
            predicted |= pending
            covered += call_key(name, args) in pending
            reads += 1
            results.append((name, READS[dom][name](**args)))
    predicted |= predict(keep, t, results, cfg)
    return len(predicted), len(predicted & actual), covered, reads, len(t["actions"])


def evaluate(train, test, dom, cfg):
    keep = fit(train, dom, cfg)
    tot = [0, 0, 0, 0, 0]
    for t in test:
        tot = [x + y for x, y in zip(tot, replay(keep, t, dom, cfg))]
    return tot


def summarize(tot):
    pred, hit, cov, reads, calls = tot
    return {"precision": hit / max(pred, 1), "coverage": cov / max(reads, 1), "calls_saved": cov / max(calls, 1), "predicted": pred, "reads": reads, "calls": calls}


def halves(tasks):
    tune = [t for t in tasks if zlib.crc32(t["user"].encode()) % 2 == 0]
    return tune, [t for t in tasks if zlib.crc32(t["user"].encode()) % 2 == 1]


def cv(tasks, dom, cfg, folds=5):
    users = sorted({t["user"] for t in tasks})
    fold = {u: i % folds for i, u in enumerate(users)}
    tot = [0, 0, 0, 0, 0]
    for f in range(folds):
        r = evaluate([t for t in tasks if fold[t["user"]] != f], [t for t in tasks if fold[t["user"]] == f], dom, cfg)
        tot = [x + y for x, y in zip(tot, r)]
    return summarize(tot)


CONFIGS = {  # v2 settings were chosen by 5-fold CV on the tuning half (bench/sweep.py) before the holdout was scored
    "v1 strict": dict(text=False, levels=(0,), mention=False, theta=0.3),
    "v1 text": dict(text=True, levels=(0,), mention=False, theta=0.3),
    "v2 strict": dict(text=False, levels=(1, 2), mention=False, theta=0.3, z=0.5),
    "v2 text": dict(text=True, levels=(1, 2), mention=True, theta=0.3, stump=3, z=1.0),
}


def suites():
    return {d: load_tasks(os.path.join(root, f"tau_bench/envs/{d}/tasks_test.py")) for d in ("retail", "airline")}


if __name__ == "__main__":
    report = []
    for dom, tasks in suites().items():
        tune, hold = halves(tasks)
        for name, cfg in CONFIGS.items():
            r = {"domain": dom, "config": name, "tune_cv": cv(tune, dom, cfg)}
            if os.environ.get("HOLDOUT") == "1":  # score the locked half only once tuning is done
                r["holdout"] = summarize(evaluate(tune, hold, dom, cfg))
            report.append(r)
            for split in [k for k in ("tune_cv", "holdout") if k in r]:
                s = r[split]
                print(f"{dom:8} {name:10} {split:8} precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%} predicted={s['predicted']} reads={s['reads']} calls={s['calls']}", flush=True)
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1)
