import json, os, random, sqlite3, sys, time, urllib.request

BRIDGE, QM = "http://127.0.0.1:8787", "http://127.0.0.1:8080"
MODEL = os.environ.get("BENCH_MODEL")
ARMS = tuple(os.environ.get("ARMS", "off,inject").split(","))
THREAD = os.environ.get("THREAD", "new")
DB = sqlite3.connect(f"file:{os.path.expanduser('~/Documents/qm-native/biz.db')}?mode=ro", uri=True)
ACCTS = [r[0] for r in DB.execute("select name from accounts order by name")]
ASKS = ["Prep me for the QBR with {e}.", "Is {e} at risk of churning before renewal?", "Any overdue or disputed invoices for {e}?",
        "{e} escalated to their exec sponsor. Summarize their open tickets and who I should contact."]
N = int(os.environ.get("N", 30))
LONG = os.environ.get("TASKSET") == "long"
CSMS = ["Priya Shah", "Marcus Lee", "Lena Ortiz", "Tomas Berg", "Aiko Tanaka"]
LONG_ASKS = ["Review {e}'s book of business: check every account they own for open tickets, overdue or disputed invoices, and usage trend, then list the at-risk accounts.",
             "Weekly risk sweep for {e}: which of their accounts have overdue or disputed invoices or urgent tickets?",
             "Renewal readiness rundown for every account {e} owns: health, billing status, and open tickets for each."]


def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(QM + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=60))


def run_turn(text, thread, level=None):
    t0 = time.time()
    body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": thread}, "text": text, **({"model": MODEL} if MODEL else {}), **({"thinkingLevel": level} if level else {})}
    queued = post("/v1/turns?async=1", body)
    run_id = queued.get("runId")
    first = finished = None
    reply = []
    status = "incomplete"
    chars = 0
    with urllib.request.urlopen(f"{QM}/v1/runs/{run_id}/events", timeout=400) as stream:
        for raw in stream:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev.get("name") == "delta" and ev["value"]["delta"].strip():
                chars += len(ev["value"]["delta"])
                reply.append(ev["value"]["delta"])
                first = first or time.time() - t0
            if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
                finished = time.time() - t0
                status = ev["type"]
                break
    usage = {}
    if os.environ.get("RUNLOG"):
        for _ in range(20):
            for line in open(os.environ["RUNLOG"], errors="replace"):
                if line.startswith("[claude] usage ") and f'"runId":"{run_id}"' in line:
                    usage = json.loads(line[len("[claude] usage "):])
            if usage:
                break
            time.sleep(0.25)
    tl = [x for x in json.load(urllib.request.urlopen(BRIDGE + "/timelines")) if x["query"] == text]
    tl = tl[-1] if tl else {}
    calls = tl.get("timeline", [])
    return {"ttft": first, "total": finished, "status": status, "chars": chars, "reply": "".join(reply),
            "tokens_in": (usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0)) if usage else None, "tokens_out": usage.get("output"), "cache_read": usage.get("cacheRead"), "cost_usd": usage.get("costUsd"), "fast_path_turn": usage.get("fastPath"),
            "to_recall": tl["at"] / 1000 - t0 if tl else None, "recall_ms": tl.get("recallDone"), "fast_reason": tl.get("fastReason"),
            "first_tool": calls[0]["start"] / 1000 + tl["at"] / 1000 - t0 if calls and tl else None}


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def expected(ask, e):
    if e in CSMS:
        return sorted({a for (a,) in DB.execute("select i.account from invoices i join accounts a on a.name = i.account where a.csm = ? and i.status != 'paid' and i.month >= '2025-07'", (e,))})
    a = dict(zip(["name", "plan", "arr", "csm", "renewal", "health", "region"], DB.execute("select * from accounts where name=?", (e,)).fetchone()))
    if "QBR" in ask:
        return [a["csm"].split()[0]]
    if "churning" in ask:
        return [f'{MONTHS[int(a["renewal"][5:7]) - 1]}|{a["renewal"][4:8]}']
    if "invoices" in ask:
        bad = [s for (s,) in DB.execute("select status from invoices where account=? order by month desc limit 6", (e,)) if s != "paid"]
        return bad[:1]
    return [DB.execute("select name from contacts where account=? and title='VP Operations'", (e,)).fetchone()[0].split()[0]]


rng = random.Random(int(os.environ.get("SEED", 41)))
tasks = [(rng.choice(LONG_ASKS), rng.choice(CSMS)) for _ in range(N)] if LONG else [(rng.choice(ASKS), rng.choice(ACCTS)) for _ in range(N)]
log = open(f"ttft-{os.environ.get('BENCH_TAG', 'run')}.jsonl", "w")
for i, (ask, acct) in enumerate(tasks):
    for arm, rep_i in [(a, k) for a in (ARMS if i % 2 == 0 else ARMS[::-1]) for k in range(int(os.environ.get("REPEAT", 0)) + 1)]:
        mode, _, level = arm.partition(":")
        urllib.request.urlopen(urllib.request.Request(f"{BRIDGE}/mode/{mode}", data=b"", method="POST"))
        thread = f"ttft-{arm.replace(':', '-')}-{i}-{rep_i}-{int(time.time() * 1000)}" if THREAD == "new" else f"ttft-{arm}-same"
        try:
            r = run_turn(ask.format(e=acct), thread, level or None)
        except Exception as ex:
            r = {"ttft": None, "total": None, "status": f"error: {ex}", "chars": 0}
        exp = expected(ask, acct)
        row = {"i": i, "rep": rep_i, "arm": arm, "text": ask.format(e=acct), **r, "expected": exp, "correct": all(any(alt.lower() in r.get("reply", "").lower() for alt in x.split("|")) for x in exp)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{i:3d} {arm:16s} ttft={r['ttft'] and round(r['ttft'], 1)} total={r['total'] and round(r['total'], 1)} chars={r['chars']} ok={row['correct']} {r['status']}", flush=True)
