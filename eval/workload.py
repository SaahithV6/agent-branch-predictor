import json, sys, time, urllib.request

BRIDGE = "http://127.0.0.1:8787"
QM = "http://127.0.0.1:8080"


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def mcp(name, args):
    return post(f"{BRIDGE}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})


PROJECTS = {
    "scrubline": "Scrubline status: 6 US launch sites live; next milestone is pad-throughput dashboard; blocker is FAA data lag.",
    "propflow": "Propflow status: RDE SSTO PDR locked at 78 t to LEO at 1,000 t GLOW; next is thermal model fidelity pass.",
    "cortex": "Cortex status: paused; Native backend passes conformance 29/0/2; ranking fix for word order pending.",
}


def seed():
    for name, body in PROJECTS.items():
        mcp("create_record", {"type": "Document", "kind": "note", "name": f"{name} project status", "body": body, "reason": "seed for predictor eval", "sources": [], "format": "json"})


ASKS = [
    "Check Native for the {p} project status and tell me the blocker or next step.",
    "What's the latest on {p}? Look it up in Native.",
    "Pull the {p} project status from Native and summarize it in one line.",
    "Status check: {p}. Use Native.",
]


def turn(i, text):
    t0 = time.time()
    r = post(f"{QM}/v1/turns", {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"wl-{i}-{int(t0)}"}, "text": text})
    return time.time() - t0, r.get("status"), (r.get("reply") or "")[:120]


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed()
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 1
    out = []
    i = 0
    for r in range(rounds):
        for p in PROJECTS:
            text = ASKS[(r + i) % len(ASKS)].format(p=p)
            dt, status, reply = turn(i, text)
            stats = json.load(urllib.request.urlopen(f"{BRIDGE}/predictor"))
            out.append({"i": i, "text": text, "seconds": round(dt, 1), "status": status, "stats": stats})
            print(f"{i:3d} {dt:5.1f}s {status} pred={stats['predicted']} hits={stats['hits']} ro={stats['readOnlyCalls']} | {text}", flush=True)
            i += 1
    print(json.dumps(out[-1]["stats"]))
