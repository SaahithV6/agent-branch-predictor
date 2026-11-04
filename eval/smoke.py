import json, os, time, urllib.request
QM = "http://127.0.0.1:8080"
LOG = os.path.expanduser("~/Documents/qm-native/run.log")
for text in ["Any overdue or disputed invoices for Netflix?", "Weekly risk sweep for Marcus Lee: which of their accounts have overdue or disputed invoices or urgent tickets?", "Any overdue or disputed invoices for Netflix?"]:
    t0 = time.time()
    q = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns?async=1", data=json.dumps({"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"smoke-{time.time()}"}, "text": text}).encode(), headers={"content-type": "application/json"})))
    first, reply = None, []
    for raw in urllib.request.urlopen(f"{QM}/v1/runs/{q['runId']}/events", timeout=300):
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        if ev.get("name") == "delta":
            first = first or time.time() - t0
            reply.append(ev["value"]["delta"])
        if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
            break
    total = time.time() - t0
    time.sleep(0.5)
    usage = [json.loads(l[len("[claude] usage "):]) for l in open(LOG, errors="replace") if l.startswith("[claude] usage ") and q["runId"] in l]
    u = usage[-1] if usage else {}
    print(f"{total:5.1f}s first={first and round(first, 1)}s fast={u.get('fastPath')} memo={u.get('memo', False)} tokens_in={u.get('input', 0) + u.get('cacheRead', 0) + u.get('cacheWrite', 0)} | {text[:50]}")
    print("    ", "".join(reply)[:200].replace("\n", " "))
