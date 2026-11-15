"""Embed every catalog tool and every evaluated context with a hosted embedding model (Nebius).

Writes data/calls/emb/tools.npy (catalog order, as zpred.Catalog loads it) and ctx_<split>.npy +
ctx_<split>.json (example key -> row) for the same samples zpred evaluates. Vectors are truncated
to DIM and re-normalized (Qwen3 embeddings are trained for truncation), stored float16.

Resumable: finished batches are kept in emb/parts/ and skipped on a rerun. Spend is capped by
--max-tokens across the whole run; the API's own usage counts are what is tallied.

    NEBIUS_API_KEY in env.local
    python3 -I bench/embed.py tools|contexts [--max-tokens 300000000]
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zpred as Z  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EMB = Z.EMB
PARTS = EMB / "parts"
BASE = os.environ.get("NEBIUS_BASE", "https://api.tokenfactory.nebius.com/v1")
MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
DIM = 256
BATCH = 96
INSTRUCT = "Instruct: Given an AI agent's task and its last tool result, retrieve the API it should call next\nQuery: "


def key():
    for line in open(ROOT / "env.local"):
        if line.startswith("NEBIUS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("NEBIUS_API_KEY missing from env.local")


class Spend:
    def __init__(self, cap):
        self.cap, self.used, self.lock = cap, 0, threading.Lock()

    def add(self, n):
        with self.lock:
            self.used += n
            if self.used > self.cap:
                raise SystemExit(f"token cap reached: {self.used:,} > {self.cap:,}")


def embed(texts, api_key, spend):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    for attempt in range(6):
        req = urllib.request.Request(f"{BASE}/embeddings", body, {"Authorization": f"Bearer {api_key}",
                                                                     "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            spend.add(r.get("usage", {}).get("total_tokens", 0))
            v = np.array([d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])], np.float32)[:, :DIM]
            return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r}")
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError("embedding batch failed 6 times")


def run(name, texts, a):
    if a.local:  # a static (model2vec) embedder on CPU: ~90k texts/s, no API, no spend
        from model2vec import StaticModel
        v = StaticModel.from_pretrained(a.local).encode([readable(t) for t in texts], batch_size=4096)[:, :DIM]
        print(f"{name}: {len(texts):,} texts embedded locally with {a.local}", flush=True)
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float16)
    PARTS.mkdir(parents=True, exist_ok=True)
    api_key, spend = key(), Spend(a.max_tokens)
    n = (len(texts) + BATCH - 1) // BATCH
    todo = [b for b in range(n) if not (PARTS / f"{name}_{b:06d}.npy").exists()]
    print(f"{name}: {len(texts):,} texts, {n:,} batches, {len(todo):,} to do", flush=True)
    t0, done = time.time(), [0]

    def one(b):
        np.save(PARTS / f"{name}_{b:06d}.npy", embed(texts[b * BATCH:(b + 1) * BATCH], api_key, spend))
        done[0] += 1
        if done[0] % 200 == 0:
            el = time.time() - t0
            print(f"  {done[0]:,}/{len(todo):,} batches, {spend.used:,} tokens, {el:.0f}s", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, todo))
    out = np.concatenate([np.load(PARTS / f"{name}_{b:06d}.npy") for b in range(n)])
    print(f"{name}: done, {spend.used:,} tokens this run", flush=True)
    return out


def readable(s):
    # split identifiers into words: a static embedder only knows words ("getUserRepos" -> "get User Repos")
    return re.sub(r"[_/\-.~@]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.replace(INSTRUCT, "")))


def tool_text(t):
    args = ", ".join(x["name"] for x in t.get("args", []))
    return f"{t['server']} / {t['name']}({args}): {t.get('desc', '')[:400]}"


def context_text(ex):
    s = ex["query"]
    if ex["prior"]:
        last = ex["prior"][-1]
        s += f"\nLast call: {last['tool']} -> {last['result'][:400]}"
    return INSTRUCT + s[:2000]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tools", "contexts", "probe"])
    ap.add_argument("--max-tokens", type=int, default=300_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--local", help="embed on CPU with this model2vec model instead, e.g. minishlab/potion-retrieval-32M")
    a = ap.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    if a.what == "probe":  # one tiny call: checks key, model name and the price basis
        v = embed(["weather forecast for a city"], key(), Spend(10_000))
        print("ok", v.shape, MODEL, BASE)
    elif a.what == "tools":
        seen, texts = set(), []
        for p in Z.CATALOGS:  # the same order and de-duplication as zpred.Catalog
            for line in open(p):
                t = json.loads(line)
                if t["id"] not in seen:
                    seen.add(t["id"])
                    texts.append(tool_text(t))
        np.save(EMB / "tools.npy", run("tools", texts, a))
    else:
        # every example of every split: zpred samples train/dev itself (same seed), test is used whole
        for split in ("train", "dev", "test"):
            xs = Z.load(split)
            np.save(EMB / f"ctx_{split}.npy", run(f"ctx_{split}", [context_text(x) for x in xs], a))
            json.dump({f"{x['src']}:{x['conv']}:{len(x['prior'])}": i for i, x in enumerate(xs)}, open(EMB / f"ctx_{split}.json", "w"))
