import json, os, random, sqlite3, sys, time, urllib.request

BRIDGE = "http://127.0.0.1:8787"
QM = "http://127.0.0.1:8080"
DB = sqlite3.connect(f"file:{os.path.expanduser('~/Documents/qm-native/biz.db')}?mode=ro", uri=True)
CUSTOMERS = [r[0] for r in DB.execute("select name from accounts order by name")]
UNSEEN = set(CUSTOMERS[::4])
TEMPLATES = {
    "qbr": ["Prep me for the QBR with {e}.", "I have a quarterly business review with {e} tomorrow. What do I need to know?", "QBR brief for {e}: usage, tickets, billing, and anything notable about the company."],
    "renewal": ["Is {e} at risk of churning before renewal?", "Renewal risk check on {e}.", "How healthy is the {e} account heading into renewal?"],
    "billing": ["{e} says their invoice is wrong. What's going on with their billing?", "Any overdue or disputed invoices for {e}?", "Billing check on {e}, anything unpaid?"],
    "escalation": ["{e} escalated to their exec sponsor. Summarize their open tickets and who I should contact.", "What's open in support for {e}, and who's the right contact there?"],
    "draft_checkin": ["Draft a check-in email to the VP Operations at {e} about their recent usage. Save it as a draft."],
}


def expected(t, e):
    a = dict(zip(["name", "plan", "arr", "csm", "renewal", "health", "region"], DB.execute("select * from accounts where name=?", (e,)).fetchone()))
    if t == "qbr":
        return [a["csm"].split()[0]]
    if t == "renewal":
        return [["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][int(a["renewal"][5:7]) - 1][:3]]
    if t == "billing":
        bad = [s for (s,) in DB.execute("select status from invoices where account=? order by month desc limit 6", (e,)) if s != "paid"]
        return bad[:1]
    if t == "escalation":
        return [DB.execute("select name from contacts where account=? and title='VP Operations'", (e,)).fetchone()[0].split()[0]]
    return []


def get(path):
    return json.load(urllib.request.urlopen(BRIDGE + path))


def tasks(phase):
    rng = random.Random(21 if phase == "warmup" else 22)
    seen_t = ["qbr", "renewal", "billing", "escalation"]
    seen_c = [c for c in CUSTOMERS if c not in UNSEEN]
    if phase == "warmup":
        return [(t, "seen", rng.choice(seen_c), rng.choice(TEMPLATES[t])) for t in (rng.choice(seen_t) for _ in range(32))]
    out = [(t, "seen", rng.choice(seen_c), rng.choice(TEMPLATES[t])) for t in (rng.choice(seen_t) for _ in range(14))]
    out += [(t, "unseen-entity", rng.choice(sorted(UNSEEN)), rng.choice(TEMPLATES[t])) for t in (rng.choice(seen_t) for _ in range(6))]
    out += [("draft_checkin", "unseen-template", c, TEMPLATES["draft_checkin"][0]) for c in rng.sample(CUSTOMERS, 4)]
    rng.shuffle(out)
    return out


def main():
    phase = sys.argv[1]
    model = os.environ.get("BENCH_MODEL")
    log = open(f"biz-{phase}{os.environ.get('BENCH_TAG', '')}.jsonl", "w")
    for i, (t, split, e, phrasing) in enumerate(tasks(phase)):
        text = phrasing.format(e=e)
        before = get("/predictor")
        t0 = time.time()
        body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"biz-{phase}-{i}-{int(t0)}"}, "text": text, **({"model": model} if model else {})}
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns", data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=300))
        except Exception as ex:
            r = {"status": f"error: {ex}"}
        dt = time.time() - t0
        after = get("/predictor")
        reply = r.get("reply") or ""
        exp = expected(t, e)
        row = {"i": i, "phase": phase, "template": t, "split": split, "entity": e, "text": text, "seconds": round(dt, 2), "status": r.get("status"),
               "calls": after["calls"] - before["calls"], "predicted": after["predicted"] - before["predicted"], "injected": after.get("injected", 0) - before.get("injected", 0),
               "expected": exp, "correct": all(x.lower() in reply.lower() for x in exp), "reply_len": len(reply), "reply": reply[:600]}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{i:3d} {dt:6.1f}s calls={row['calls']} pred={row['predicted']} ok={row['correct']} {split:15s} {text}", flush=True)


if __name__ == "__main__":
    main()
