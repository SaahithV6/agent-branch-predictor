"""Offline replay of the branch predictor on tau-bench ground-truth trajectories.

Usage: python3 -I bench/tau_replay.py <tau-bench checkout> [out.json]

tau-bench (sierra-research/tau-bench) gives each test task the exact tool calls a correct
agent makes. We replay those calls against the benchmark's own JSON data, learn argument-
source rules on training folds ("get_order_details.order_id comes from
get_user_details -> orders[*]"), and score predictions on held-out tasks:

  precision  = predicted read calls the agent then made / all predicted read calls
  coverage   = read calls that were already predicted when the agent made them / all read calls
  calls_saved = covered read calls / all tool calls (calls the model would not have to issue)

Task files are parsed with ast (never imported); tool results are recomputed here from the
data files. Writes are never predicted. "results" mode only uses earlier tool results;
"+instruction" also pulls values from the user's instruction text, which the real agent only
sees piecemeal through the conversation, so treat it as an upper bound.
"""

import ast
import itertools
import json
import os
import re
import sys
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
            tasks.append({"user": ast.literal_eval(kw["user_id"]), "instruction": ast.literal_eval(kw["instruction"]), "actions": actions})
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
    """Yield (pattern, value) for every scalar leaf and id-like dict key; list indices and id keys become wildcards."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            step = "{k}" if idlike(k) else k
            if idlike(k):
                yield path + ("{k}",), k
            yield from walk(v, path + (step,))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, path + ("*",))
    elif obj is not None:
        yield path, str(obj)


def shape(v):
    out, prev, n = [], None, 0
    for c in v + "\0":
        cls = r"\d" if c.isdigit() else "[A-Z]" if c.isupper() else "[a-z]" if c.islower() else re.escape(c)
        if cls == prev:
            n += 1
            continue
        if prev and prev != re.escape("\0"):
            out.append(prev + (f"{{{n}}}" if n > 1 else ""))
        prev, n = cls, 1
    return "".join(out)


def call_key(name, args):
    return name + json.dumps(args, sort_keys=True)


def sources(task, results):
    """index[source] -> set of values, where source is ("R", tool, pattern) or ("I", shape)."""
    idx = defaultdict(set)
    for tool, res in results:
        for pat, v in walk(res):
            idx[("R", tool) + pat].add(v)
    return idx


def instr_values(text, shp):
    return set(re.findall(r"(?<![\w#])" + shp + r"(?!\w)", text))


def arg_sources(task, results, args, use_instr):
    idx = sources(task, results)
    opts = {}
    for a, v in args.items():
        if not isinstance(v, str):
            return None
        srcs = [s for s, vals in idx.items() if v in vals]
        if use_instr and v in task["instruction"]:
            srcs.append(("I", shape(v)))
        if not srcs:
            return None
        opts[a] = srcs
    return opts


def learn(tasks, dom, use_instr):
    """Rules: (tool, ((arg, source), ...)) observed in training trajectories."""
    rules = defaultdict(int)
    for t in tasks:
        results = []
        for name, args in t["actions"]:
            if name in READS[dom]:
                opts = arg_sources(t, results, args, use_instr)
                if opts is not None:
                    for combo in itertools.product(*[[(a, s) for s in ss] for a, ss in sorted(opts.items())]):
                        rules[(name, combo)] += 1
                results.append((name, READS[dom][name](**args)))
    return rules


def predict(rules, t, results, dom):
    idx = sources(t, results)
    out = set()
    for (name, combo), _ in rules.items():
        vals = []
        for a, s in combo:
            vs = instr_values(t["instruction"], s[1]) if s[0] == "I" else idx.get(s, set())
            if not vs:
                break
            vals.append([(a, v) for v in sorted(vs)])
        else:
            for c in itertools.islice(itertools.product(*vals), 50):
                out.add(call_key(name, dict(c)))
    return out


def replay(rules, t, dom):
    """Per-task: predicted set, covered reads, total reads, total calls."""
    results, predicted, covered = [], set(), 0
    actual = [call_key(n, a) for n, a in t["actions"] if n in READS[dom]]
    for name, args in t["actions"]:
        if name in READS[dom]:
            pending = predict(rules, t, results, dom)
            predicted |= pending
            covered += call_key(name, args) in pending
            results.append((name, READS[dom][name](**args)))
    predicted |= predict(rules, t, results, dom)
    return predicted, set(actual), covered, len(actual), len(t["actions"])


def score_rules(rules, tasks, dom):
    """Per-rule (fired, hit) on training tasks, so low-confidence rules can be dropped."""
    stats = defaultdict(lambda: [0, 0])
    for t in tasks:
        actual = {call_key(n, a) for n, a in t["actions"] if n in READS[dom]}
        results = []
        for name, args in t["actions"] + [(None, None)]:
            for r in rules:
                for k in predict({r: 1}, t, results, dom):
                    stats[r][0] += 1
                    stats[r][1] += k in actual
            if name in READS[dom]:
                results.append((name, READS[dom][name](**args)))
    return stats


def run(dom, tasks, use_instr, theta, folds=5):
    users = sorted({t["user"] for t in tasks})
    fold_of = {u: i % folds for i, u in enumerate(users)}
    tot = dict(pred=0, hit=0, cov=0, reads=0, calls=0)
    for f in range(folds):
        train = [t for t in tasks if fold_of[t["user"]] != f]
        test = [t for t in tasks if fold_of[t["user"]] == f]
        rules = learn(train, dom, use_instr)
        stats = score_rules(rules, train, dom)
        keep = {r: n for r, n in rules.items() if n >= 2 and stats[r][1] / max(stats[r][0], 1) >= theta}
        for t in test:
            predicted, actual, cov, reads, calls = replay(keep, t, dom)
            tot["pred"] += len(predicted)
            tot["hit"] += len(predicted & actual)
            tot["cov"] += cov
            tot["reads"] += reads
            tot["calls"] += calls
    return {"precision": tot["hit"] / max(tot["pred"], 1), "coverage": tot["cov"] / max(tot["reads"], 1),
            "calls_saved": tot["cov"] / max(tot["calls"], 1), "predicted": tot["pred"], **{k: tot[k] for k in ("reads", "calls")}}


if __name__ == "__main__":
    suites = {"retail": load_tasks(os.path.join(root, "tau_bench/envs/retail/tasks_test.py")),
              "airline": load_tasks(os.path.join(root, "tau_bench/envs/airline/tasks_test.py"))}
    report = []
    for dom, tasks in suites.items():
        for use_instr in (False, True):
            for theta in (0.0, 0.3, 0.5, 0.7):
                r = run(dom, tasks, use_instr, theta)
                r.update(domain=dom, tasks=len(tasks), mode="+instruction" if use_instr else "results", theta=theta)
                report.append(r)
                print(f"{dom:8} {r['mode']:13} theta={theta:.1f}  precision={r['precision']:.0%}  coverage={r['coverage']:.0%}  calls_saved={r['calls_saved']:.0%}  predicted={r['predicted']}  reads={r['reads']}  calls={r['calls']}", flush=True)
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1)
