import json, re, sqlite3, sys
from collections import Counter

sys.path.insert(0, ".")
STOP = set("the and for with that this from have what your about into when were will would there their them they then than which while right now check tell give".split())
ENUMS = {"auth", "billing", "search", "checkout", "notify", "ingest", "gateway", "reports", "acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka"}


def words(q):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9._-]{2,}", q.lower()) if w not in STOP}


def load():
    db = sqlite3.connect("file:../bench.db?mode=ro", uri=True)
    out = []
    for (body,) in db.execute("select body from records where kind='tool-trace' order by rowid"):
        d = json.loads(body)
        if d.get("mode") != "shadow":
            continue
        calls = [(c["name"], json.dumps({k: v for k, v in c["args"].items() if k not in ("limit",) and not k.startswith("include_")}, sort_keys=True)) for c in d["calls"]]
        calls = [c for c in calls if c[0].startswith("sim_") and c[0] not in ("sim_page_oncall", "sim_create_ticket", "sim_post_status_update")]
        out.append({"query": d["query"], "calls": set(calls)})
    return out


def entity(q, known):
    hits = [w for w in re.findall(r"[a-z0-9_-]+", q.lower()) if w in known]
    return hits[0] if hits else None


def slot(call, e):
    name, args = call
    return (name, args.replace(f'"{e}"', '"$E"')) if e else call


def fill(tmpl, e):
    name, args = tmpl
    return (name, args.replace('"$E"', f'"{e}"')) if e else tmpl


def predictor(variant, theta, tau):
    hist = []

    def predict(q):
        known = ENUMS if variant != "exact" else set()
        e = entity(q, known)
        w = words(q) - ({e} if e else set())
        if variant == "exact":
            nbrs = [h for h in hist if len(words(q) & words(h["query"])) >= max(2, len(words(q)) / 2)]
            keyed = [h["calls"] for h in nbrs]
        else:
            levels = []
            same_e = [h for h in hist if h["e"] == e]
            for pool in ([h for h in same_e if jacc(w, h["w"]) >= tau], [h for h in hist if jacc(w, h["w"]) >= tau]):
                if len(pool) >= 2:
                    levels.append(pool)
            nbrs = levels[0] if (variant == "tage" and levels) else (levels[-1] if levels else [])
            keyed = [h["tmpl"] for h in nbrs]
        if len(keyed) < 2:
            return set()
        cnt = Counter(c for s in keyed for c in s)
        chosen = {c for c, n in cnt.items() if n >= 2 and n / len(keyed) >= theta}
        return chosen if variant == "exact" else {fill(c, e) for c in chosen}

    def learn(q, calls):
        e = entity(q, ENUMS)
        hist.append({"query": q, "calls": calls, "e": e, "w": words(q) - ({e} if e else set()), "tmpl": {slot(c, e) for c in calls}})

    return predict, learn


def jacc(a, b):
    return len(a & b) / len(a | b) if a | b else 0


def run(traces, variant, theta, tau, eval_from):
    predict, learn = predictor(variant, theta, tau)
    P = H = A = turns_pred = 0
    for i, t in enumerate(traces):
        p = predict(t["query"])
        if i >= eval_from:
            P += len(p); H += len(p & t["calls"]); A += len(t["calls"]); turns_pred += bool(p)
        learn(t["query"], t["calls"])
    return P, H, A, turns_pred


if __name__ == "__main__":
    traces = load()
    n = len(traces)
    print(f"{n} shadow traces; scoring turns 10..{n - 1} (online), and the final 24 (held-out eval mix)")
    rows = []
    for variant in ["exact", "slots", "tage"]:
        for theta in [0.5, 0.7, 0.9]:
            for tau in ([0] if variant == "exact" else [0.3, 0.5]):
                for label, start in [("all", 10), ("eval24", n - 24)]:
                    P, H, A, tp = run(traces, variant, theta, tau, start)
                    rows.append((variant, theta, tau, label, P, H, A, tp))
                    print(f"{variant:6s} theta={theta} tau={tau} {label:6s} precision={H / max(P, 1):5.0%} coverage={H / max(A, 1):5.0%} predicted={P:3d} turns_with_pred={tp}")
    json.dump(rows, open("replay-results.json", "w"))
