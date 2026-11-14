"""Router for prefetchd over a partitioned tool catalog: placement, forwarding, failover.

Usage: python3 -I bench/router.py plan <load.json> <partitions> <shards> <plan.json>
       python3 -I bench/router.py shard-catalog <plan.json> <tools.jsonl> <outdir>
       python3 -I bench/router.py serve <plan.json> [--port 9500] [--routers K]

A session never sees the whole catalog; it is connected to a few servers. Servers that are used
together live in the same partition, and each partition owns a group of prefetchd shards sized to its
load (plan). A new session goes to the partition holding most of its servers, and to a shard in that
group by rendezvous hashing on the session id. The shard is written into the session id the router
hands back, so every later event routes without a lookup table: routers keep no state, and K of them
share one port (SO_REUSEPORT).

  POST /session {servers, ...}    placed, forwarded to the chosen shard; sid -> "<shard>.<sid>"
  POST /text|/call|/task|/close   forwarded to the shard named in the sid
  POST /call {sid, name, server}  a tool owned by another partition is executed there (/exec), never
                                  on the session's shard; the session's own tools stay local
  GET  /metrics                   router counters

A shard that stops answering is marked down for 5 s: new sessions go to the rest of its group, and its
sessions' events get 503 {"reopen": true}, so the client opens a fresh session (placed on a live shard)
and carries on cold. Prefetch state is a cache: losing it costs speed, never correctness.
"""

import collections
import hashlib
import os
import http.client
import json
import socket
import sys
import threading
import time
import math
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# planning -----------------------------------------------------------------------------------------
def owner_key(server, tool, split):
    """A server too big for one partition is split by tool name: "server#chunk"."""
    k = split.get(server)
    return f"{server}#{zlib.crc32(tool.encode()) % k}" if k and tool else server


def plan(load, co_use, partitions, shards, splittable=False):
    """load: {server: weight}; co_use: [[a, b, weight]]; returns {"partition_of": {server: p}, "shards": [[port index]]}.
    Servers joined by heavy co-use edges form one group (union-find); groups are packed largest-first onto the
    least-loaded partition; shards are split across partitions in proportion to load (at least one each).
    splittable: servers are stateless API surfaces, so one larger than half a partition is cut into chunks by tool
    name. Never for servers holding state (a tau domain's database): their tools must stay together."""
    cap = sum(load.values()) / partitions
    can = (lambda s: s in splittable) if isinstance(splittable, (set, list, tuple)) else (lambda s: bool(splittable))  # noqa: E731
    split = {s: math.ceil(w / (cap / 2)) for s, w in load.items() if can(s) and w > cap / 2}  # chunks of at most half a partition
    load = {**{s: w for s, w in load.items() if s not in split},
            **{f"{s}#{i}": load[s] / k for s, k in split.items() for i in range(k)}}
    parent, size = {s: s for s in load}, dict(load)  # size: load of the group rooted at s

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    total = sum(load.values()) or 1
    for a, b, w in sorted(co_use, key=lambda e: -e[2]):
        if a in parent and b in parent and w >= 0.01 * total:  # only co-use worth a placement constraint
            ra, rb = find(a), find(b)
            if ra != rb and size[ra] + size[rb] <= total / partitions:
                parent[ra] = rb
                size[rb] += size[ra]
    groups = collections.defaultdict(list)
    for s in load:
        groups[find(s)].append(s)
    bins = [[0.0, p, []] for p in range(partitions)]
    for g in sorted(groups.values(), key=lambda g: -sum(load[s] for s in g)):
        b = min(bins, key=lambda x: (x[0], x[1]))
        b[0] += sum(load[s] for s in g)
        b[2] += g
    want = [max(1.0, shards * b[0] / total) for b in bins]
    alloc = [int(w) for w in want]
    while sum(alloc) < shards:
        alloc[max(range(partitions), key=lambda i: want[i] - alloc[i])] += 1
    while sum(alloc) > shards:
        alloc[max(range(partitions), key=lambda i: alloc[i] - want[i] if alloc[i] > 1 else -9)] -= 1
    ports, k = [], 0
    for n in alloc:
        ports.append(list(range(k, k + n)))
        k += n
    return {"partition_of": {s: b[1] for b in bins for s in b[2]}, "shards": ports, "split": split,
            "load": [round(b[0] / total, 4) for b in bins]}


def rendezvous(key, choices):
    return max(choices, key=lambda c: hashlib.blake2b(f"{key}|{c}".encode(), digest_size=8).digest())


# serving ------------------------------------------------------------------------------------------
class Router:
    def __init__(self, pl, base_port, cat=None, n_shards=None):
        self.part_of, self.groups, self.base = pl["partition_of"], pl["shards"], base_port
        self.split = pl.get("split", {})
        # a stateless catalog laid over the same shards: its partition c lives on every shard s with s % C == c
        self.cat = cat or {"partition_of": {}, "shards": [], "split": {}}
        self.n_shards = n_shards or sum(len(g) for g in self.groups)
        self.down = {}  # shard -> time it failed; retried after 5 s
        self.local = threading.local()
        self.n = collections.Counter()
        self.lock = threading.Lock()

    def count(self, **kw):
        with self.lock:
            self.n.update(kw)

    def alive(self, group):
        now = time.time()
        return [s for s in group if now - self.down.get(s, 0) > 5] or group

    def home(self, servers):
        votes = collections.Counter(self.part_of.get(owner_key(s, "", {}) if s not in self.split else f"{s}#0", 0) for s in servers or [None])
        return votes.most_common(1)[0][0]

    def post(self, shard, path, body):
        conns = self.local.__dict__.setdefault("conns", {})
        for attempt in range(2):
            c = conns.get(shard) or http.client.HTTPConnection("127.0.0.1", self.base + shard, timeout=60)
            conns[shard] = c
            try:
                c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, json.loads(r.read())
            except (ConnectionError, http.client.HTTPException, TimeoutError, OSError):
                c.close()
                conns.pop(shard, None)
                if attempt:
                    self.down[shard] = time.time()
                    raise

    def open(self, body):
        p = self.home(body.get("servers"))
        for _ in range(len(self.groups[p])):
            shard = rendezvous(json.dumps(body, sort_keys=True) + str(time.perf_counter_ns()), self.alive(self.groups[p]))
            try:
                st, out = self.post(shard, "/session", body)
            except OSError:
                self.count(failover=1)
                continue
            if st == 200:
                out["sid"] = f"{shard}.{out['sid']}"
                self.count(sessions=1, **{f"sessions_p{p}": 1})
            return st, out
        return 503, {"error": "no shard in partition"}

    def forward(self, path, body):
        shard, sid = body["sid"].split(".", 1)
        shard = int(shard)
        if path == "/call" and body.get("server") is not None:
            owner = self.part_of.get(owner_key(body["server"], body.get("name", ""), self.split))
            hosts, stateless = (self.groups[owner], False) if owner is not None else (None, True)
            if hosts is None:  # a catalog tool: the shards hosting its catalog partition
                c = self.cat["partition_of"].get(owner_key(body["server"], body.get("name", ""), self.cat.get("split", {})))
                n = len(self.cat["shards"])
                hosts = [s for s in range(self.n_shards) if c is not None and s % n == c] or None
            if hosts and (stateless or shard not in hosts):  # catalog tools always run stateless; others run where they live
                ex = shard if shard in hosts else rendezvous(body["server"], self.alive(hosts))
                self.count(forwarded=int(ex != shard), exec_local=int(ex == shard))
                return self.post(ex, "/exec", {k: v for k, v in body.items() if k != "sid"})
        self.count(**{path[1:]: 1})
        try:
            return self.post(shard, path, {**body, "sid": sid})
        except OSError:
            self.count(shard_down=1)
            return 503, {"error": f"shard {shard} down", "reopen": True}


def handler(rt):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, dict(rt.n)) if self.path == "/metrics" else self.reply(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                st, out = rt.open(body) if self.path == "/session" else rt.forward(self.path, body)
            except (OSError, KeyError, ValueError) as e:
                st, out = 502, {"error": repr(e)}
            self.reply(st, out)

        def log_message(self, *a):
            pass
    return H


class ReusePortServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


def serve(pl, port, shard_base, cat=None):
    ReusePortServer(("127.0.0.1", port), handler(Router(pl, shard_base, cat))).serve_forever()


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        load = json.load(open(sys.argv[2]))
        sp = load.get("splittable", False)
        pl = plan(load["load"], load.get("co_use", []), int(sys.argv[3]), int(sys.argv[4]), set(sp) if isinstance(sp, list) else sp)
        json.dump(pl, open(sys.argv[5], "w"), indent=1)
        print(json.dumps({"partitions": len(pl["shards"]), "shards": [len(g) for g in pl["shards"]], "load": pl["load"]}))
    elif sys.argv[1] == "shard-catalog":  # plan.json tools.jsonl outdir: one file of read tools per partition
        pl, outdir = json.load(open(sys.argv[2])), sys.argv[4]
        os.makedirs(outdir, exist_ok=True)
        files = [open(os.path.join(outdir, f"p{p}.jsonl"), "w") for p in range(len(pl["shards"]))]
        n = collections.Counter()
        for line in open(sys.argv[3]):
            t = json.loads(line)
            p = pl["partition_of"].get(owner_key(t["server"], t["name"], pl.get("split", {})))
            if t["read"] and p is not None:
                files[p].write(line)
                n[p] += 1
        print(json.dumps({"read_tools_per_partition_min_max": [min(n.values()), max(n.values())], "partitions": len(files)}))
    elif sys.argv[1] == "serve":
        import argparse
        import multiprocessing as mp
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("plan")
        ap.add_argument("--port", type=int, default=9500)
        ap.add_argument("--shard-port", type=int, default=9600)
        ap.add_argument("--routers", type=int, default=2)
        ap.add_argument("--catalog-plan", help="plan of a stateless tool catalog hosted round-robin on the same shards")
        a = ap.parse_args()
        pl = json.load(open(a.plan))
        cat = json.load(open(a.catalog_plan)) if a.catalog_plan else None
        ps = [mp.Process(target=serve, args=(pl, a.port, a.shard_port, cat), daemon=True) for _ in range(a.routers)]
        for p in ps:
            p.start()
        print(f"router: {a.routers} processes on 127.0.0.1:{a.port} -> shards {a.shard_port}+", flush=True)
        for p in ps:
            p.join()
