import json, os, random, sys, time, urllib.request

BRIDGE = "http://127.0.0.1:8787"
QM = "http://127.0.0.1:8080"
TOPICS = ["sqlite", "webassembly", "risc-v", "postgres", "kubernetes", "rust async", "nuclear power", "starship", "llm agents", "vector databases", "htmx", "zig", "deno",
          "local-first software", "e-bikes", "solid-state batteries", "fusion energy", "open source licensing", "remote work", "static site generators", "tailscale", "nixos",
          "home assistant", "apple vision pro", "chess engines", "quantum computing", "lidar", "gpu prices", "terraform", "bun runtime"]
TEMPLATES = ["What's Hacker News saying about {e}? Read the top two linked articles and summarize the key points.",
             "Find recent HN discussion on {e} and tell me what the two most relevant linked articles argue.",
             "Research {e}: search HN, read a couple of the most relevant articles, and give me a short brief."]


def get(path):
    return json.load(urllib.request.urlopen(BRIDGE + path))


def tasks(phase):
    rng = random.Random(31 if phase == "warmup" else 32)
    pool = TOPICS[:12] if phase == "warmup" else TOPICS[12:]
    return [(t, rng.choice(TEMPLATES)) for t in rng.sample(pool, 10 if phase == "warmup" else 18)]


def main():
    phase = sys.argv[1]
    model = os.environ.get("BENCH_MODEL")
    log = open(f"research-{phase}{os.environ.get('BENCH_TAG', '')}.jsonl", "w")
    for i, (e, phrasing) in enumerate(tasks(phase)):
        text = phrasing.format(e=e)
        before = get("/predictor")
        t0 = time.time()
        body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"res-{phase}-{i}-{int(t0)}"}, "text": text, **({"model": model} if model else {})}
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns", data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=400))
        except Exception as ex:
            r = {"status": f"error: {ex}"}
        dt = time.time() - t0
        after = get("/predictor")
        reply = r.get("reply") or ""
        d = lambda k: after.get(k, 0) - before.get(k, 0)
        row = {"i": i, "phase": phase, "topic": e, "text": text, "seconds": round(dt, 2), "status": r.get("status"), "calls": d("calls"),
               "chain_speculated": d("chainSpeculated"), "chain_hits": d("chainHits"), "chain_saved_ms": d("chainSavedMs"), "saved_ms": d("savedMs"),
               "reply_len": len(reply), "reply": reply[:600]}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{i:3d} {dt:6.1f}s calls={row['calls']} chain spec={row['chain_speculated']} hits={row['chain_hits']} saved={row['chain_saved_ms']}ms | {text[:70]}", flush=True)


if __name__ == "__main__":
    main()
