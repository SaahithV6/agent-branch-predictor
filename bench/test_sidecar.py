"""Self-check for the prefetch sidecar's failure handling (tau_chat_mcp.py). No model, no network.

Usage: python3 -I bench/test_sidecar.py <tau-bench>
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import Process

os.environ.setdefault("TAU_ROOT", sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Documents/tau-bench"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tau_chat_mcp as M  # noqa: E402

READ_TOOL, WRITE_TOOL = "get_order_details", "cancel_pending_order"
ARGS = {"order_id": "#W2611340"}
K = M.key(READ_TOOL, ARGS)


def fresh():
    return tempfile.mkdtemp(prefix="sidecar_test_")


def dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_dead_owner_not_waited_on():
    sd = fresh()
    open(M.cpath(sd, K, "pending"), "w").write(f"{dead_pid()} {time.time()}")
    t0 = time.time()
    assert M.from_cache(sd, K, timeout=5) is None
    assert time.time() - t0 < 0.5, "waited on a prefetch whose process is gone"
    assert not os.path.exists(M.cpath(sd, K, "pending"))


def test_write_during_prefetch_never_served():
    sd = fresh()
    real = M.execute
    started = threading.Event()

    def slow(name, args):
        started.set()
        time.sleep(0.3)
        return real(name, args)
    M.execute = slow
    try:
        th = threading.Thread(target=M.prefetch, args=(sd, READ_TOOL, ARGS, {}, "test"))
        th.start()
        started.wait()
        M.execute = real
        M.call(sd, None, WRITE_TOOL, {"order_id": "#W2611340", "reason": "no longer needed"})  # write lands mid-prefetch
        th.join()
    finally:
        M.execute = real
    assert M.from_cache(sd, K, timeout=1) is None, "served a read that raced a write"
    rows = [json.loads(x) for x in open(os.path.join(sd, "trace.jsonl"))]
    assert any(r.get("discarded") for r in rows)


def test_stale_generation_rejected():
    sd = fresh()
    M.prefetch(sd, READ_TOOL, ARGS, {}, "test")
    assert M.from_cache(sd, K, timeout=1) is not None
    open(os.path.join(sd, "gen"), "w").write("7")  # a write happened; the file was not (yet) removed
    assert M.from_cache(sd, K, timeout=1) is None


def test_tool_crash_cleans_up():
    sd = fresh()
    real = M.execute
    M.execute = lambda name, args: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        assert M.prefetch(sd, READ_TOOL, ARGS, {}, "test") is None
    finally:
        M.execute = real
    assert not os.path.exists(M.cpath(sd, K, "pending"))
    t0 = time.time()
    assert M.from_cache(sd, K, timeout=5) is None and time.time() - t0 < 0.5


def _append(sd, tag, n):
    for i in range(n):
        M.update_state(sd, lambda st: st["results"].append([tag, i]))


def test_no_lost_state_updates():
    sd = fresh()
    ps = [Process(target=_append, args=(sd, f"p{j}", 50)) for j in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    assert len(M.load_state(sd)["results"]) == 200, "concurrent state writes lost updates"


class FakePred:
    def __init__(self, n):
        self.n = n

    def predict(self, text, results, called, explain=False):
        return [{"key": M.key(READ_TOOL, {"order_id": f"#W{i:07d}"}), "score": 1 - i / 1000, "inject": False, "cache": True, "why": []}
                for i in range(self.n)]

    def explain_last(self):
        return []


def test_prefetch_cap_and_dedupe():
    sd = fresh()
    started = M.start(sd, FakePred(40), "", "test")
    assert len(started) == M.MAX_PREFETCH
    for *_, t, _, _ in started:
        t.join()
    again = M.start(sd, FakePred(40), "", "test")  # the first MAX_PREFETCH are cached now: not re-run
    keys = {M.key(n, a) for n, a, *_ in again}
    assert not keys & {M.key(n, a) for n, a, *_ in started}
    for *_, t, _, _ in again:
        t.join()


def test_predictor_crash_serves_tools():
    class Broken(FakePred):
        def predict(self, *a, **k):
            raise ValueError("feature count mismatch")
    sd = fresh()
    out = M.call(sd, Broken(1), READ_TOOL, ARGS)
    assert out and "Error" not in out[:5]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
