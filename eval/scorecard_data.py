import glob, json, math, statistics as st

ok_status = lambda r: r.get("status") in ("ok", "RUN_FINISHED")


def load(p):
    return [json.loads(l) for l in open(p)]


def paired(a_rows, b_rows, key="seconds", by_text=True):
    d = []
    for a, b in zip(a_rows, b_rows):
        if by_text:
            assert a.get("text") == b.get("text"), (a.get("text"), b.get("text"))
        if ok_status(a) and ok_status(b) and a.get(key) is not None and b.get(key) is not None:
            d.append(b[key] - a[key])
    return d


def interleaved(path, a, b, key):
    by = {}
    for r in load(path):
        by.setdefault(r["i"], {})[r["arm"]] = r
    return [v[b][key] - v[a][key] for v in by.values() if a in v and b in v and ok_status(v[a]) and ok_status(v[b]) and v[a].get(key) is not None and v[b].get(key) is not None]


def summ(d):
    se = st.stdev(d) / math.sqrt(len(d))
    m = st.mean(d)
    return {"n": len(d), "mean": round(m, 2), "lo": round(m - 1.96 * se, 2), "hi": round(m + 1.96 * se, 2), "median": round(st.median(d), 2), "faster": sum(x < 0 for x in d)}


def medians(rows, key="seconds"):
    return round(st.median(r[key] for r in rows if ok_status(r) and r.get(key) is not None), 1)


exp = []


def add(name, workload, model, design, base, test, d, base_rows=None, test_rows=None, key="seconds", note=""):
    row = {"name": name, "workload": workload, "model": model, "design": design, **summ(d), "note": note}
    if base_rows is not None:
        row["base_median"] = medians(base_rows, key)
        row["test_median"] = medians(test_rows, key)
        row["base_calls"] = round(st.mean(r.get("calls", 0) for r in base_rows), 2)
        row["test_calls"] = round(st.mean(r.get("calls", 0) for r in test_rows), 2)
    exp.append(row)


B, S = load("run1/bench-baseline.jsonl"), load("run1/bench-prefetch.jsonl")
add("Prefetch into prompt (exact-call predictor)", "Simulated skills", "Sonnet", "separate runs", "baseline", "prefetch", paired(B, S), B, S)
B, S = load("bench-baseline.jsonl"), load("bench-speculate.jsonl")
add("Slot predictor + speculative cache", "Simulated skills", "Sonnet", "separate runs", "baseline", "speculate", paired(B, S), B, S)
B, S = load("real1/real-baseline.jsonl"), load("real1/real-speculate.jsonl")
add("Speculate (5-12 ms tools)", "Job log + repos (real)", "Sonnet", "separate runs", "baseline", "speculate", paired(B, S), B, S)
B, S = load("real-baseline.jsonl"), load("real-inject.jsonl")
add("Inject live results", "Job log + repos (real)", "Sonnet", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("real-baseline-haiku.jsonl"), load("real-inject-haiku.jsonl")
add("Inject live results", "Job log + repos (real)", "Haiku", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("real-baseline-cold.jsonl"), load("real-inject-cold.jsonl")
add("Inject, cold start (classifier only)", "Job log + repos (real)", "Haiku", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("biz2/biz-baseline-haiku.jsonl"), load("biz2/biz-inject-haiku.jsonl")
add("Inject, predicted calls", "Business desk (SQL + Wikipedia)", "Haiku", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("biz3/biz-baseline-haiku.jsonl"), load("biz3/biz-inject-haiku.jsonl")
add("Inject, full set per account", "Business desk (SQL + Wikipedia)", "Haiku", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("biz-baseline-haiku.jsonl"), load("biz-inject-haiku.jsonl")
add("Inject, full set labeled background", "Business desk (SQL + Wikipedia)", "Haiku", "separate runs", "baseline", "inject", paired(B, S), B, S)
B, S = load("research-baseline-haiku.jsonl"), load("research-speculate-haiku.jsonl")
add("Chained speculation", "Research desk (live HN + fetch)", "Haiku", "separate runs", "baseline", "speculate", paired(B, S), B, S)
add("Chained speculation", "Research desk (live HN + fetch)", "Haiku", "interleaved", "off", "speculate", interleaved("ab-research.jsonl", "off", "speculate", "seconds"))
add("Augment: attach follow-up pages", "Research desk (live HN + fetch)", "Haiku", "interleaved", "off", "augment", interleaved("ab-augment.jsonl", "off", "augment", "seconds"))
d = interleaved("ttft-biz-off-inject.jsonl", "off", "inject", "total") + interleaved("ttft-combo.jsonl", "off", "inject", "total") + interleaved("ttft-fastpath.jsonl", "off", "inject", "total")
add("Inject live results (streamed timing)", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "off", "inject", d)
add("Inject, full set (streamed timing)", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "off", "inject+entity", interleaved("ttft-combo.jsonl", "off", "inject+entity", "total"))
add("Fast path: lite turn, no tools, thinking off", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "off", "inject+calls+fast", interleaved("ttft-fastpath.jsonl", "off", "inject+calls+fast", "total"))
add("Low effort on inject turns", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "inject", "inject:low", interleaved("ttft-inject-effort.jsonl", "inject", "inject:low", "total"))


def pool_diff(folder, key):
    def load_pool(p):
        return [r for f in sorted(glob.glob(f"{folder}profile-pool{p}-*.json")) for r in json.load(open(f))]
    pairs = {}
    for r in load_pool(0):
        pairs.setdefault(r["text"], [[], []])[0].append(r[key])
    for r in load_pool(1):
        pairs.setdefault(r["text"], [[], []])[1].append(r[key])
    return [st.mean(v[1]) - st.mean(v[0]) for v in pairs.values() if v[0] and v[1]]


add("Warm process pool, single slot", "Business + research", "Haiku", "alternating blocks", "pool off", "pool on", pool_diff("pool1/", "total"))
add("Warm process pool, keyed by launch shape", "Business + research", "Haiku", "alternating blocks", "pool off", "pool on", pool_diff("", "total"))

ttft = []
d = interleaved("ttft-biz-off-inject.jsonl", "off", "inject", "ttft") + interleaved("ttft-combo.jsonl", "off", "inject", "ttft") + interleaved("ttft-fastpath.jsonl", "off", "inject", "ttft")
ttft.append({"name": "Inject live results", **summ(d)})
ttft.append({"name": "Fast path: lite turn, thinking off", **summ(interleaved("ttft-fastpath.jsonl", "off", "inject+calls+fast", "ttft"))})
ttft.append({"name": "Fast path, full account (loose)", **summ(interleaved("ttft-loose.jsonl", "off", "inject+entity+fast+loose", "ttft"))})
ttft.append({"name": "Inject, full set", **summ(interleaved("ttft-combo.jsonl", "off", "inject+entity", "ttft"))})
ttft.append({"name": "Low effort on inject turns", **summ(interleaved("ttft-inject-effort.jsonl", "inject", "inject:low", "ttft"))})
ttft.append({"name": "Warm pool, single slot (first model step)", **summ(pool_diff("pool1/", "first_step"))})
ttft.append({"name": "Warm pool, keyed (first model step)", **summ(pool_diff("", "first_step"))})

accuracy = [
    {"name": "History only, exact calls", "data": "Your Claude Code transcripts", "precision": 2.0, "coverage": 2.0},
    {"name": "Recency argument ranker", "data": "Your Claude Code transcripts", "precision": 2.7, "coverage": 2.7},
    {"name": "Exact-call history", "data": "Simulated skills (replay)", "precision": 21, "coverage": 7},
    {"name": "History slots", "data": "Simulated skills (replay)", "precision": 90, "coverage": 57},
    {"name": "History slots (live)", "data": "Simulated skills", "precision": 94, "coverage": 68},
    {"name": "History slots (live)", "data": "Job log + repos (real)", "precision": 95, "coverage": 83},
    {"name": "Haiku zero-shot", "data": "Job log + repos (real)", "precision": 92, "coverage": 67},
    {"name": "Distilled classifier, no history", "data": "Job log + repos (real)", "precision": 82, "coverage": 93},
    {"name": "Chained URL speculation", "data": "Research desk (live HN)", "precision": 32, "coverage": 86},
]

profile = json.load(open("profile-pool0-block1.json")) + json.load(open("profile-pool0-block3.json"))
anatomy = []
for group, sel in (("Business desk", lambda r: not r["text"].startswith("What's")), ("Research desk", lambda r: r["text"].startswith("What's"))):
    rows = [r for r in profile if sel(r)]
    anatomy.append({"name": group, **{k: round(st.median(r[k] for r in rows), 2) for k in ("first_step", "tool", "between_steps", "final_answer", "total")}})

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def rescored(r):
    if "expected" not in r or "reply" not in r:
        return r.get("correct", False)
    low = r["reply"].lower()
    def hit(x):
        alts = x.split("|")
        if x in MONTHS:
            alts.append(f"-{MONTHS.index(x) + 1:02d}-")
        return any(a.lower() in low for a in alts)
    return all(hit(x) for x in r["expected"])


def arm_rows(path, labels):
    rows = [{**r, "correct": rescored(r)} for r in load(path)]
    out = []
    for arm, label in labels:
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "RUN_FINISHED"]
        out.append({"arm": label, "total": round(st.median(r["total"] for r in rs), 1), "ttft": round(st.median(r["ttft"] for r in rs if r["ttft"]), 1),
                    "correct": sum(r.get("correct", False) for r in rows if r["arm"] == arm), "n": sum(r["arm"] == arm for r in rows)})
    return out


arms = {"business": arm_rows("ttft-fastpath.jsonl", [("off", "No prediction"), ("inject", "Inject"), ("inject+calls+fast", "Inject + fast path")]),
        "loose": arm_rows("ttft-loose.jsonl", [("off", "No prediction"), ("inject+calls+fast", "Fast path, predicted calls"), ("inject+entity+fast+loose", "Fast path, full account")])}
add("Fast path, full account (loose)", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "off", "inject+entity+fast+loose", interleaved("ttft-loose.jsonl", "off", "inject+entity+fast+loose", "total"))
try:
    arms["long"] = arm_rows("ttft-long.jsonl", [("off", "No prediction"), ("inject+calls+deep", "Deep prefetch"), ("inject+calls+deep+fast", "Deep prefetch + fast path")])
    add("Deep prefetch (list, then every account)", "Portfolio reviews (multi-step)", "Haiku", "interleaved", "off", "inject+calls+deep", interleaved("ttft-long.jsonl", "off", "inject+calls+deep", "total"))
    add("Deep prefetch + fast path", "Portfolio reviews (multi-step)", "Haiku", "interleaved", "off", "inject+calls+deep+fast", interleaved("ttft-long.jsonl", "off", "inject+calls+deep+fast", "total"))
except (FileNotFoundError, st.StatisticsError, ZeroDivisionError):
    pass

tokens = []
for path, group, labels in (("ttft-biz-final.jsonl", "Single-account questions", [("off", "No prediction"), ("inject+entity+fast+loose", "Fast path, full account")]),
                            ("ttft-long-final.jsonl", "Portfolio reviews", [("off", "No prediction"), ("inject+entity+deep+fast+loose", "Deep prefetch + fast path")])):
    try:
        rows = load(path)
    except FileNotFoundError:
        continue
    for arm, label in labels:
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "RUN_FINISHED" and r.get("tokens_in") is not None]
        if rs:
            tokens.append({"group": group, "arm": label, "tokens_in": round(st.median(r["tokens_in"] for r in rs)), "tokens_out": round(st.median(r["tokens_out"] for r in rs)),
                           "cost": round(st.mean(r["cost_usd"] for r in rs), 4), "n": len(rs), "fast": sum(r.get("fast_reason") == "fast" for r in rows if r["arm"] == arm)})
    if path == "ttft-long-final.jsonl":
        arms["longfast"] = arm_rows(path, [("off", "No prediction"), ("inject+entity+deep+fast+loose", "Deep prefetch + fast path")])
        add("Deep prefetch + fast path (loose)", "Portfolio reviews (multi-step)", "Haiku", "interleaved", "off", "inject+entity+deep+fast+loose", interleaved(path, "off", "inject+entity+deep+fast+loose", "total"))
    else:
        add("Fast path, full account (loose), with tokens", "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "off", "inject+entity+fast+loose", interleaved(path, "off", "inject+entity+fast+loose", "total"))

FULL = "inject+entity+deep+fast+loose+compact+nothink+prune"
for path, key, group, label in (("ttft-stack-biz.jsonl", "stackbiz", "Single-account questions, full stack", "Business desk (SQL + Wikipedia)"),
                                ("ttft-stack-long.jsonl", "stacklong", "Portfolio reviews, full stack", "Portfolio reviews (multi-step)")):
    try:
        rows = load(path)
    except FileNotFoundError:
        continue
    arms[key] = arm_rows(path, [("off", "No prediction"), (FULL, "Full stack")])
    add("Full stack: deep + fast path + compact + no-think + prune", label, "Haiku", "interleaved", "off", FULL, interleaved(path, "off", FULL, "total"))
    ttft.append({"name": f"Full stack, {label.split(' (')[0].lower()}", **summ(interleaved(path, "off", FULL, "ttft"))})
    for arm, name in (("off", "No prediction"), (FULL, "Full stack")):
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "RUN_FINISHED" and r.get("tokens_in") is not None]
        tokens.append({"group": group, "arm": name, "tokens_in": round(st.median(r["tokens_in"] for r in rs)), "tokens_out": round(st.median(r["tokens_out"] for r in rs)),
                       "cost": round(st.mean(r["cost_usd"] for r in rs), 4), "n": len(rs), "fast": sum(r.get("fast_reason") == "fast" for r in rows if r["arm"] == arm)})

try:
    rows = load("ttft-memo.jsonl")
    memo_arms = []
    for arm, rep, label in ((FULL.replace("prune", "prunefam"), 1, "Full stack"), (FULL.replace("prune", "prunefam") + "+memo", 1, "Full stack + memo")):
        rs = [r for r in rows if r["arm"] == arm and r["rep"] == rep and r["status"] == "RUN_FINISHED"]
        memo_arms.append({"arm": label, "total": round(st.median(r["total"] for r in rs), 2), "ttft": round(st.median(r["ttft"] for r in rs if r["ttft"]), 2), "correct": sum(rescored(r) for r in rs), "n": len(rs)})
        tokens.append({"group": "Repeat requests", "arm": label, "tokens_in": round(st.median(r["tokens_in"] or 0 for r in rs)), "tokens_out": round(st.median(r["tokens_out"] or 0 for r in rs)),
                       "cost": round(st.mean(r["cost_usd"] or 0 for r in rs), 4), "n": len(rs), "fast": len(rs)})
    arms["memo"] = memo_arms
except FileNotFoundError:
    pass
try:
    for arm, label in (("inject+calls+prune", "Tool pruning, per turn"), ("inject+calls+prunefam", "Tool pruning, by family")):
        add(label, "Business desk (SQL + Wikipedia)", "Haiku", "interleaved", "inject+calls", arm, interleaved("ttft-prunefam.jsonl", "inject+calls", arm, "total"))
except FileNotFoundError:
    pass

json.dump({"experiments": exp, "ttft": ttft, "accuracy": accuracy, "anatomy": anatomy, "arms": arms, "tokens": tokens}, open("scorecard.json", "w"), indent=1)
for e in exp:
    print(f"{e['mean']:+6.2f} [{e['lo']:+6.2f},{e['hi']:+6.2f}] n={e['n']:2d} faster={e['faster']:2d} {e['design']:18s} {e['workload'][:22]:22s} {e['model']:6s} {e['name']}")
print("ttft:", [(t['name'], t['mean'], t['lo'], t['hi']) for t in ttft])
print("anatomy:", anatomy)
