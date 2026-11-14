"""prefetchd: speculative tool execution as a long-running service.

Usage: python3 -I bench/prefetchd.py <tau-bench> serve [--shards N] [--port P] [--models DIR]

Sessions live in shard processes (shard i listens on P+i; a session id names its shard). Clients
report what the customer says and route every tool call through the service, which executes it
for real (bench/realtools.py: tau2's own tools and databases, one environment per session). After
each event the shard's predictor guesses the next lookups and runs them in the background, so a
later call is served from the cache instead of waiting on the tool.

  POST /session {domain, fold, task}           -> {sid}
  POST /text    {sid, text}                    customer message
  POST /call    {sid, name, args}              -> {result, source: cache|inflight|exec, attached}
  POST /task    {sid, task, reset=true}        next ticket in a long workflow: fresh environment and
                                               (unless reset=false) a clean slate for the predictor
  POST /close   {sid}
  POST /exec    {name, args}                   run a tool for a session placed in another partition (router.py)
  GET  /metrics

Built to run without limits on workflow length or session count:
- Bounded memory per session: the predictor sees the last WINDOW read results and TEXT_CHARS of
  customer text, the view its model was trained on (longctx.py records it next to the models;
  WINDOW/TEXT_CHARS env vars otherwise). The cache keeps CACHE_MAX entries. Idle sessions beyond
  MAX_SESSIONS per shard are evicted, least recent first.
- Prediction is off the request path: one predictor thread per shard, a queue that holds each
  session at most once (a burst of events costs one prediction on the latest state).
- Never a stale read: every write bumps the session's generation and clears its cache; a prefetch
  stamped with an older generation is never served, and one that finishes late is dropped.
  Only tools that are on the domain's read allowlist and that tau2 itself marks as non-mutating
  are ever run ahead of time.
- Upstream quota: with UPSTREAM_RATE set, each upstream server gets a token bucket per shard. The
  agent's own calls always draw from it and are never refused; a prefetch runs only while the bucket
  holds a whole token, so speculation spends only quota the agent left unused.
"""

import collections
import json
import os
import random
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import realtools as RT  # noqa: E402
import traces as TR  # noqa: E402

WINDOW = int(os.environ.get("WINDOW", "24"))
TEXT_CHARS = int(os.environ.get("TEXT_CHARS", "4000"))
CACHE_MAX = int(os.environ.get("CACHE_MAX", "256"))
MAX_PREFETCH = int(os.environ.get("MAX_PREFETCH", "8"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "256"))  # each holds its own tau2 database (25-80 MB); real tools would share one
WAIT = float(os.environ.get("PREFETCH_WAIT", "2.0"))  # longest a call waits on its own prefetch
# calls/s one shard may send each upstream server (its quota / shards); 0 = unlimited
UPSTREAM_RATE = float(os.environ.get("UPSTREAM_RATE", "0"))
UPSTREAM_BURST = float(os.environ.get("UPSTREAM_BURST", "0")) or max(UPSTREAM_RATE, 1.0)
SAMPLES = 20000  # latency reservoir per metric


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.n = collections.Counter()
        self.ms = collections.defaultdict(list)

    def add(self, **kw):
        with self.lock:
            self.n.update(kw)

    def time(self, name, ms):
        with self.lock:
            xs = self.ms[name]
            if len(xs) < SAMPLES:
                xs.append(ms)
            else:  # reservoir sampling: every sample equally likely to be kept
                i = random.randrange(self.n[f"_{name}"] + 1)
                if i < SAMPLES:
                    xs[i] = ms
            self.n[f"_{name}"] += 1

    def snapshot(self):
        with self.lock:
            pct = {k: {q: round(float(np.percentile(v, q)), 3) for q in (50, 90, 99)} for k, v in self.ms.items() if v}
            return {"counts": {k: v for k, v in self.n.items() if not k.startswith("_")}, "ms": pct}


class Bucket:
    """Token bucket for one upstream server. take(force=True) is the agent: it always gets through,
    and may drive the balance negative so prefetches wait until the agent's use is paid back."""

    def __init__(self, rate, burst):
        self.rate, self.burst, self.tokens, self.t = rate, burst, burst, time.monotonic()
        self.lock = threading.Lock()

    def take(self, force=False):
        if not self.rate:
            return True
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if force or self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class Session:
    def __init__(self, sid, domain, task, pred, view):
        self.sid, self.domain, self.pred = sid, domain, pred
        window, self.text_chars = view
        self.env = RT.make_env(domain, task)
        self.lock = threading.Lock()  # guards env and everything below
        self.gen = 0
        self.window = collections.deque(maxlen=window)  # (key, name, parsed result, gen) of reads the agent saw
        self.text = ""
        self.cache = collections.OrderedDict()  # key -> (gen, result, "prefetch" | "memo"); LRU
        self.inflight = {}  # key -> Future
        self.attached = {}  # key -> gen: strict-gate guesses handed to the agent, not yet called
        self.used = set()  # prefetched keys served at least once (for the waste count)
        self.seen = time.time()

    def readable(self, name):
        return name in TR.READ.get(self.domain, (name,)) and not self.env._is_mutating_tool(name)


def values(args):
    """Every scalar argument value, lists included, as strings."""
    out = set()
    for v in args.values():
        for x in v if isinstance(v, list) else [v]:
            if not isinstance(x, (dict, list)):
                out.add(str(x))
    return out


class Shard:
    def __init__(self, idx, models):
        self.idx, self.models = idx, models
        self.sessions = collections.OrderedDict()  # sid -> Session, least recently used first
        self.lock = threading.Lock()
        self.preds = {}
        self.catalog = None  # set by serve_shard before any session opens
        self.io = ThreadPoolExecutor(int(os.environ.get("IO_THREADS", "16")))
        self.queue, self.queued, self.cv = collections.deque(), {}, threading.Condition()
        self.stats = Stats()
        # ponytail: a server is the session's domain (one tau2 backend each); key by tool owner once
        # catalog tools are prefetched. Split per shard: a central limiter if shard load is uneven.
        self.buckets = collections.defaultdict(lambda: Bucket(UPSTREAM_RATE, UPSTREAM_BURST))
        self.count = 0
        threading.Thread(target=self.predict_loop, daemon=True).start()

    # sessions -----------------------------------------------------------------------------------
    def predictor(self, domain, fold):
        """(Predictor, (window, text chars) it was trained to see)."""
        k = (domain, fold)
        if k not in self.preds:
            meta = os.path.join(self.models, f"gpred_{domain}_folds.json")
            m = json.load(open(meta)) if os.path.exists(meta) else {}
            path = os.path.join(self.models, f"gpred_{domain}_fold{fold}")
            # no model for this domain yet: serve its tools unpredicted rather than refuse the session
            pred = G.Predictor(path) if os.path.exists(path + ".txt") else None
            if pred is not None:
                pred.fz.catalog = self.catalog  # this shard's partition of a large tool catalog, if any
            self.preds[k] = (pred, (m.get("window", WINDOW), m.get("text_chars", TEXT_CHARS)))
        return self.preds[k]

    def open(self, domain, fold, task):
        with self.lock:
            self.count += 1
            sid = f"{self.idx}-{self.count}"
            pred, view = self.predictor(domain, fold)
        s = Session(sid, domain, task, pred, view)
        with self.lock:
            self.sessions[sid] = s
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
                self.stats.add(evicted=1)
        self.stats.add(sessions=1)
        return {"sid": sid}

    def get(self, sid):
        with self.lock:
            s = self.sessions[sid]
            self.sessions.move_to_end(sid)
        s.seen = time.time()
        return s

    def close(self, sid):
        with self.lock:
            self.sessions.pop(sid, None)
        return {}

    def task(self, sid, task, reset):
        s = self.get(sid)
        env = RT.make_env(s.domain, task)
        with s.lock:
            s.env, s.gen = env, s.gen + 1
            s.cache.clear()
            s.attached.clear()
            if reset:
                s.window.clear()
                s.text = ""
        return {}

    # events -------------------------------------------------------------------------------------
    def text(self, sid, text):
        s = self.get(sid)
        with s.lock:
            s.text = (s.text + "\n" + text)[-s.text_chars:]
        self.schedule(sid)
        return {}

    def call(self, sid, name, args):
        s = self.get(sid)
        k, t0 = G.key(name, args), time.perf_counter()
        out, src, fut = None, "exec", None
        with s.lock:
            was_attached = s.attached.pop(k, None) == s.gen
            if s.readable(name):
                e = s.cache.get(k)
                if e and e[0] == s.gen:
                    out, src = e[1], "cache" if e[2] == "prefetch" else "memo"
                    s.cache.move_to_end(k)
                else:
                    fut = s.inflight.get(k)
        if fut is not None:  # its prefetch is running: waiting is cheaper than a second call
            try:
                fut.result(timeout=WAIT)
            except Exception:
                pass
            with s.lock:
                e = s.cache.get(k)
                if e and e[0] == s.gen:
                    out, src = e[1], "inflight"
        with s.lock:
            if out is None:
                self.buckets[s.domain].take(force=True)
                out, write = RT.call(s.env, name, args)
                if write:
                    s.gen += 1
                    s.cache.clear()
                    s.attached.clear()
                    s.used.clear()
                    # only reads of what the write touched become rereadable; the rest still count as called
                    touched = values(args)
                    s.window = collections.deque(((k_, n_, r_, g_ if values(json.loads(k_[len(n_):])) & touched else s.gen)
                                                  for k_, n_, r_, g_ in s.window), maxlen=s.window.maxlen)
                    self.stats.add(writes=1)
            else:
                write = False
                if src != "memo" and k not in s.used:
                    s.used.add(k)
                    self.stats.add(prefetch_used=1)
            if not write and s.readable(name) and not out.startswith("Error") and G.scalar_args(args):
                s.window.append((k, name, TR.parse(out), s.gen))
                if src == "exec":  # read-through: the same read again in this generation is free
                    self.store(s, k, out, s.gen, "memo")
        self.stats.add(calls=1, **{f"served_{src}": 1}, attached_hit=int(was_attached))
        self.stats.time("call", (time.perf_counter() - t0) * 1000)
        self.schedule(sid)
        return {"result": out, "source": src, "attached": was_attached}

    # prediction ---------------------------------------------------------------------------------
    def schedule(self, sid):
        if self.get(sid).pred is None:
            return
        with self.cv:
            if sid in self.queued:
                self.stats.add(coalesced=1)
                return
            self.queued[sid] = time.perf_counter()
            self.queue.append(sid)
            self.cv.notify()

    def predict_loop(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                sid = self.queue.popleft()
                t_in = self.queued.pop(sid)
            try:
                self.predict(sid, t_in)
            except KeyError:  # closed or evicted meanwhile
                pass
            except Exception as e:  # a broken prediction must never take the service down
                self.stats.add(predict_errors=1)
                print(f"predict error {sid}: {e!r}", file=sys.stderr, flush=True)

    def predict(self, sid, t_in):
        with self.lock:
            s = self.sessions[sid]
        t0 = time.perf_counter()
        self.stats.time("queue_lag", (t0 - t_in) * 1000)
        with s.lock:
            gen, text = s.gen, s.text
            results = [(n, r) for _, n, r, _ in s.window]
            # "already called" holds only within a generation: after a new ticket, or a write touching
            # the record, the agent rereads what it saw before, and those rereads are predictable
            called = {k for k, _, _, g in s.window if g == gen}
            seen = {k for k, *_ in s.window} - called  # rereads: prefetch them, but the agent already saw them
        guesses = s.pred.predict(text, results, called, explain=False)
        self.stats.time("predict", (time.perf_counter() - t0) * 1000)
        self.stats.add(predictions=1)
        started = 0
        with s.lock:
            if s.gen != gen:  # a write landed while predicting: the next event predicts again
                self.stats.add(predictions_stale=1)
                return
            for g in guesses:
                k = g["key"]
                name = k[:k.index("{")]
                if not s.readable(name):
                    continue
                if g["inject"] and k not in seen:
                    if s.attached.get(k) != gen:
                        self.stats.add(attached=1)
                    s.attached[k] = gen
                if started >= MAX_PREFETCH or k in s.inflight or (k in s.cache and s.cache[k][0] == gen):
                    continue
                if not self.buckets[s.domain].take():
                    self.stats.add(prefetch_over_budget=1)
                    continue
                started += 1
                s.inflight[k] = self.io.submit(self.prefetch, s, k, name, json.loads(k[len(name):]), gen)
        self.stats.add(prefetch_started=started)

    def prefetch(self, s, k, name, args, gen):
        try:
            with s.lock:
                if s.gen != gen:
                    self.stats.add(prefetch_skipped=1)
                    return
                out, write = RT.call(s.env, name, args)
                if write:  # tau2 says it mutates after all: treat it as a write, never cache it
                    s.gen += 1
                    s.cache.clear()
                    self.stats.add(prefetch_wrote=1)
                    return
                self.store(s, k, out, gen, "prefetch")
                self.stats.add(prefetch_done=1)
        except Exception as e:
            self.stats.add(prefetch_errors=1)
            print(f"prefetch error {k}: {e!r}", file=sys.stderr, flush=True)
        finally:
            with s.lock:
                s.inflight.pop(k, None)

    @staticmethod
    def store(s, k, out, gen, origin):
        """Caller holds s.lock."""
        s.cache[k] = (gen, out, origin)
        s.cache.move_to_end(k)
        while len(s.cache) > CACHE_MAX:
            s.cache.popitem(last=False)

    def metrics(self):
        with self.lock:
            n = len(self.sessions)
        with self.cv:
            q = len(self.queue)
        return {"shard": self.idx, "sessions_open": n, "queue": q, "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                **self.stats.snapshot()}


def handler(shard):
    routes = {"/session": lambda b: shard.open(b["domain"], b.get("fold", 0), b.get("task")),
              "/text": lambda b: shard.text(b["sid"], b["text"]),
              "/call": lambda b: shard.call(b["sid"], b["name"], b.get("args") or {}),
              "/task": lambda b: shard.task(b["sid"], b.get("task"), b.get("reset", True)),
              "/close": lambda b: shard.close(b["sid"]),
              # a tool owned by this shard's partition, called from a session placed elsewhere (bench/router.py).
              # No real backend for catalog tools in this harness: a stub upstream answers deterministically.
              "/exec": lambda b: {"result": json.dumps({"tool": b["name"], "args": b.get("args") or {}, "upstream": "stub"}),
                                  "source": "exec", "attached": False}}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, shard.metrics()) if self.path == "/metrics" else self.reply(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            fn = routes.get(self.path)
            if not fn:
                return self.reply(404, {"error": "not found"})
            try:
                self.reply(200, fn(body))
            except KeyError as e:
                self.reply(404, {"error": f"unknown {e}"})
            except Exception as e:
                self.reply(500, {"error": repr(e)})

        def log_message(self, *a):
            pass
    return H


def partition_catalog(idx, catalog):
    """This shard's catalog partition as a shortlist index; None without a catalog. catalog is the directory
    `router.py shard-catalog` wrote, one file per partition (a shard never parses the rest). The catalog is
    stateless, so its C partitions are hosted round-robin on the shard pool: partition idx % C."""
    if not catalog:
        return None
    import toolindex as X
    n = len([f for f in os.listdir(catalog) if f.startswith("p") and f.endswith(".jsonl")])
    return X.ToolIndex(json.loads(line) for line in open(os.path.join(catalog, f"p{idx % n}.jsonl")))


def serve_shard(idx, port, models, catalog=None):
    shard = Shard(idx, models)
    shard.catalog = partition_catalog(idx, catalog)
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler(shard))
    srv.daemon_threads = True
    srv.request_queue_size = 1024
    srv.serve_forever()


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("tau")
    ap.add_argument("cmd", choices=["serve"])
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "live"))
    ap.add_argument("--catalog", help="partitioned tool catalog (bench/router.py shard-catalog dir); shard i indexes partition i %% C")
    a = ap.parse_args()
    if a.shards == 1:
        serve_shard(0, a.port, a.models, a.catalog)
    ps = [mp.Process(target=serve_shard, args=(i, a.port + i, a.models, a.catalog), daemon=True) for i in range(a.shards)]
    for p in ps:
        p.start()
    print(f"prefetchd: {a.shards} shards on 127.0.0.1:{a.port}-{a.port + a.shards - 1}", flush=True)
    try:
        for p in ps:
            p.join()
    except KeyboardInterrupt:
        pass
