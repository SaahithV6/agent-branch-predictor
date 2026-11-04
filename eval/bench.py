import json, random, sys, time, urllib.request

BRIDGE = "http://127.0.0.1:8787"
QM = "http://127.0.0.1:8080"

TEMPLATES = {
    "incident": ("service", [
        "Is {e} healthy right now? If not, who's on call and what's the top error?",
        "Quick triage on {e}: health, top error, on-call.",
        "{e} might be down. Check it and tell me who to ping and the error code.",
    ]),
    "release": ("service", [
        "Can we ship {e} today? Check CI and blockers.",
        "Release readiness for {e}: CI, blockers, changelog size.",
        "Is {e} good to deploy?",
    ]),
    "escalation": ("customer", [
        "{e} is escalating. What's their situation (tier, tickets, billing)?",
        "Brief me on {e}: tier, open tickets, anything overdue.",
        "Customer {e} called angry, give me their account picture.",
    ]),
    "usage": ("customer", [
        "How is {e}'s usage trending and are they paying on time?",
        "Usage and billing check for {e}.",
        "Is {e} growing or shrinking, and do they owe us money?",
    ]),
    "status_update": ("service", [
        "Post a status update for {e} summarizing its health and latest deploy version.",
    ]),
}
SEEN = {"service": ["auth", "billing", "search", "checkout", "notify", "ingest"], "customer": ["acme", "globex", "initech", "umbrella", "hooli", "stark"]}
UNSEEN = {"service": ["gateway", "reports"], "customer": ["wayne", "wonka"]}


def get(path):
    return json.load(urllib.request.urlopen(BRIDGE + path))


def expected(world, template, e):
    if template in ("incident",):
        s = world["services"][e]
        return [] if s["healthy"] else [s["error"], s["oncall"]]
    if template == "release":
        s = world["services"][e]
        return [s["ci"]]
    if template == "status_update":
        return [world["services"][e]["version"]]
    c = world["customers"][e]
    facts = [c["tier"]] if template == "escalation" else [c["usageTrend"].split()[0]]
    return facts + ([c["overdue"]] if c["overdue"] != "none" else [])


def tasks(phase):
    rng = random.Random(7 if phase == "warmup" else 11)
    seen_templates = ["incident", "release", "escalation", "usage"]
    out = []
    if phase == "warmup":
        for _ in range(40):
            t = rng.choice(seen_templates)
            kind, phr = TEMPLATES[t]
            out.append((t, "seen", rng.choice(SEEN[kind]), rng.choice(phr)))
        return out
    for _ in range(16):
        t = rng.choice(seen_templates)
        kind, phr = TEMPLATES[t]
        out.append((t, "seen", rng.choice(SEEN[kind]), rng.choice(phr)))
    for _ in range(4):
        t = rng.choice(seen_templates)
        kind, phr = TEMPLATES[t]
        out.append((t, "unseen-entity", rng.choice(UNSEEN[kind]), rng.choice(phr)))
    for e in ["auth", "search", "gateway", "notify"]:
        out.append(("status_update", "unseen-template", e, TEMPLATES["status_update"][1][0]))
    rng.shuffle(out)
    return out


def main():
    phase = sys.argv[1]
    world = get("/sim/world")
    log = open(f"bench-{phase}.jsonl", "w")
    for i, (t, split, e, phrasing) in enumerate(tasks(phase)):
        text = phrasing.format(e=e)
        before = get("/predictor")
        t0 = time.time()
        req = urllib.request.Request(QM + "/v1/turns", data=json.dumps({"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"{phase}-{i}-{int(t0)}"}, "text": text}).encode(), headers={"content-type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as ex:
            r = {"status": f"error: {ex}"}
        dt = time.time() - t0
        after = get("/predictor")
        reply = r.get("reply") or ""
        exp = expected(world, t, e)
        row = {
            "i": i, "phase": phase, "template": t, "split": split, "entity": e, "text": text,
            "seconds": round(dt, 2), "status": r.get("status"),
            "calls": after["calls"] - before["calls"],
            "predicted": after["predicted"] - before["predicted"],
            "speculated": after.get("speculated", 0) - before.get("speculated", 0), "saved_ms": after.get("savedMs", 0) - before.get("savedMs", 0),
            "expected": exp, "correct": all(x.lower() in reply.lower() for x in exp), "reply": reply[:400],
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{i:3d} {dt:6.1f}s calls={row['calls']} pred={row['predicted']} ok={row['correct']} {split:15s} {text}", flush=True)


if __name__ == "__main__":
    main()
