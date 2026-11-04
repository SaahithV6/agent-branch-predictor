import json, os, sys, time, urllib.request

BRIDGE = "http://127.0.0.1:8787"
QM = "http://127.0.0.1:8080"
TOPICS = ["rust in the linux kernel", "git worktrees", "sqlite wal mode", "cloudflare workers", "openstreetmap", "duckdb", "raspberry pi 5", "framework laptop", "mechanical keyboards",
          "neovim", "emacs", "ebpf", "io_uring", "flatpak", "wayland", "pipewire", "tor browser", "signal protocol", "passkeys", "lets encrypt", "ipv6 adoption", "starlink latency",
          "heat pumps", "geothermal energy", "lab-grown meat", "crispr", "james webb telescope", "voyager probes", "mars helicopter", "airships"]
if os.environ.get("TOPICSET") == "2":
    TOPICS = ["solar sails", "kelp farming", "e-ink displays", "open source firmware", "coreboot", "risc-v laptops", "self-hosting email", "matrix chat", "mastodon", "bluesky",
              "small language models", "speculative decoding", "flash attention", "mixture of experts", "rlhf", "lora fine-tuning", "gguf quantization", "llama.cpp", "ollama", "vllm",
              "postgres extensions", "timescaledb", "clickhouse", "redis license change", "valkey", "opentofu", "nomad", "firecracker", "gvisor", "unikernels"]
ARMS = tuple(os.environ.get("ARMS", "off,speculate").split(","))
ASK = "What's Hacker News saying about {e}? Read the two most relevant linked articles. Answer in exactly 5 bullet points."
MODEL = os.environ.get("BENCH_MODEL")


def get(path, data=None):
    req = urllib.request.Request(BRIDGE + path, data=data, method="POST" if data is not None else "GET")
    return json.load(urllib.request.urlopen(req))


log = open(f"ab-{os.environ.get('BENCH_TAG', 'research')}.jsonl", "w")
for i, topic in enumerate(TOPICS):
    for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
        get(f"/mode/{arm}", b"")
        text = ASK.format(e=topic)
        before = get("/predictor")
        t0 = time.time()
        body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"ab-{arm}-{i}-{int(t0)}"}, "text": text, **({"model": MODEL} if MODEL else {})}
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns", data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=400))
        except Exception as ex:
            r = {"status": f"error: {ex}"}
        dt = time.time() - t0
        after = get("/predictor")
        d = lambda k: after.get(k, 0) - before.get(k, 0)
        row = {"i": i, "arm": arm, "topic": topic, "seconds": round(dt, 2), "status": r.get("status"), "calls": d("calls"), "chain_speculated": d("chainSpeculated"),
               "chain_hits": d("chainHits"), "chain_saved_ms": d("chainSavedMs"), "augmented": d("augmented"), "augment_reused": d("augmentReused"), "reply_len": len(r.get("reply") or "")}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"{i:3d} {arm:9s} {dt:6.1f}s calls={row['calls']} hits={row['chain_hits']} saved={row['chain_saved_ms']}ms len={row['reply_len']} | {topic}", flush=True)
