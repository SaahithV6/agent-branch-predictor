"""Self-check for the router: placement, routing by session id, cross-partition forwarding, failover.
Fake shards (tiny HTTP servers that echo who they are); no models, no tau2.

Usage: python3 -I bench/test_router.py
"""

import http.client
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router as R  # noqa: E402

BASE, RPORT = 9870, 9860


def fake_shard(idx):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = {"shard": idx, "path": self.path, "body": body}
            if self.path == "/session":
                out["sid"] = f"s{idx}-{body.get('n', 0)}"
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", BASE + idx), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post(path, body):
    c = http.client.HTTPConnection("127.0.0.1", RPORT, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, json.loads(r.read())


PLAN = R.plan({"retail": 6, "airline": 3, "github": 1, "jira": 1}, [["github", "jira", 5]], 3, 6)
SHARDS = [fake_shard(i) for i in range(6)]
CAT = R.plan({"stripe": 1, "slack": 1}, [], 2, 2)  # a catalog of 2 partitions over the same 6 shards
srv = R.ReusePortServer(("127.0.0.1", RPORT), R.handler(R.Router(PLAN, BASE, CAT)))
threading.Thread(target=srv.serve_forever, daemon=True).start()


def test_plan_keeps_co_used_servers_together_and_sizes_groups():
    p = PLAN["partition_of"]
    assert p["github"] == p["jira"] and len({p["retail"], p["airline"], p["github"]}) == 3
    assert len(PLAN["shards"][p["retail"]]) > len(PLAN["shards"][p["github"]])
    assert sorted(s for g in PLAN["shards"] for s in g) == list(range(6))


def test_oversized_stateless_server_is_split_and_tools_route_stably():
    pl = R.plan({"azure": 40, "github": 5, "jira": 5}, [], 4, 8, splittable=True)
    assert "azure" in pl["split"] and "azure" not in pl["partition_of"]
    owners = {R.owner_key("azure", f"op{i}", pl["split"]) for i in range(200)}
    assert len(owners) == pl["split"]["azure"] and all(o in pl["partition_of"] for o in owners)
    assert R.owner_key("azure", "op7", pl["split"]) == R.owner_key("azure", "op7", pl["split"])
    assert len({pl["partition_of"][o] for o in owners}) > 1  # the big server is spread over partitions
    assert "retail" in PLAN["partition_of"]  # without splittable (stateful servers) nothing is cut


def test_session_lands_in_its_partition_and_events_follow_the_sid():
    group = PLAN["shards"][PLAN["partition_of"]["airline"]]
    for n in range(20):
        st, out = post("/session", {"servers": ["airline"], "n": n})
        assert st == 200 and out["shard"] in group
        st, ev = post("/text", {"sid": out["sid"], "text": "hi"})
        assert ev["shard"] == out["shard"] and ev["body"]["sid"] == out["sid"].split(".", 1)[1]


def test_foreign_tool_runs_in_its_owner_partition():
    _, s = post("/session", {"servers": ["retail"]})
    st, out = post("/call", {"sid": s["sid"], "name": "list_issues", "server": "jira", "args": {}})
    assert out["path"] == "/exec" and out["shard"] in PLAN["shards"][PLAN["partition_of"]["jira"]]
    assert "sid" not in out["body"]
    _, own = post("/call", {"sid": s["sid"], "name": "get_order_details", "server": "retail", "args": {}})
    assert own["path"] == "/call" and own["shard"] == s["shard"]


def test_catalog_tool_runs_on_a_shard_hosting_its_partition():
    _, s = post("/session", {"servers": ["retail"]})
    c = CAT["partition_of"]["slack"]
    st, out = post("/call", {"sid": s["sid"], "name": "list_channels", "server": "slack", "args": {}})
    assert out["shard"] % 2 == c and out["path"] == "/exec"  # stateless: never the session's tau environment
    if s["shard"] % 2 == c:
        assert out["shard"] == s["shard"]  # hosted locally: no hop


def test_dead_shard_asks_client_to_reopen_and_new_sessions_avoid_it():
    group = PLAN["shards"][PLAN["partition_of"]["retail"]]
    _, s = post("/session", {"servers": ["retail"]})
    SHARDS[s["shard"]].shutdown()
    SHARDS[s["shard"]].server_close()
    st, out = post("/text", {"sid": s["sid"], "text": "x"})
    assert st == 503 and out["reopen"]
    for n in range(10):
        st, again = post("/session", {"servers": ["retail"], "n": n})
        assert st == 200 and again["shard"] in group and again["shard"] != s["shard"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in sorted(tests, key=lambda f: f.__code__.co_firstlineno):  # the failover test kills a shard: run it last
        t()
        print("ok", t.__name__, flush=True)
    print(f"{len(tests)} passed")
