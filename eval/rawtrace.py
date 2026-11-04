import json, time, urllib.request

QM = "http://127.0.0.1:8080"
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8787/mode/inject", data=b"", method="POST"))
for text in ["Any overdue or disputed invoices for Netflix?", "Prep me for the QBR with Boeing."]:
    t0 = time.time()
    body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"raw-{t0}"}, "text": text, "model": "claude-haiku-4-5"}
    q = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns?async=1", data=json.dumps(body).encode(), headers={"content-type": "application/json"})))
    first = None
    for raw in urllib.request.urlopen(f"{QM}/v1/runs/{q['runId']}/events", timeout=300):
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        t = time.time() - t0
        if ev.get("type") in ("TOOL_CALL_START", "TOOL_CALL_RESULT"):
            print(f"  {t:5.1f}s {ev['type']} {ev.get('toolCallName', '')} {json.dumps(ev.get('args', ''))[:100]}")
        elif ev.get("name") == "delta" and first is None:
            first = t
            print(f"  {t:5.1f}s first text")
        elif ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
            print(f"  {t:5.1f}s {ev['type']}")
            break
    print("---", text, flush=True)
