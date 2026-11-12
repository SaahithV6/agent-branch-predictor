"""Self-check for prefetchd's invariants, on tau2's real retail tools. No model, no HTTP.

Usage: python3 -I bench/test_prefetchd.py <tau-bench>
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefetchd as P  # noqa: E402
import realtools as RT  # noqa: E402

ORDER = "#W5918442"  # pending in tau2's retail database
READ = ("get_order_details", {"order_id": ORDER})
WRITE = ("cancel_pending_order", {"order_id": ORDER, "reason": "no longer needed"})


class Fake:
    """Guesses the given keys every time; optionally blocks or crashes."""

    def __init__(self, keys, gate=None, crash=False):
        self.keys, self.gate, self.crash, self.calls = keys, gate, crash, 0

    def predict(self, text, results, called, explain=False):
        self.calls += 1
        if self.gate:
            self.gate.wait()
        if self.crash:
            raise ValueError("feature count mismatch")
        return [{"key": k, "score": 0.99, "inject": True, "cache": True, "why": []} for k in self.keys if k not in called]


def shard(pred):
    sh = P.Shard(0, "unused")
    sh.preds[("retail", 0)] = (pred, (P.WINDOW, P.TEXT_CHARS))
    return sh


def settle(sh, s, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with sh.cv:
            idle = not sh.queue and not sh.queued
        with s.lock:
            busy = bool(s.inflight)
        if idle and not busy:
            time.sleep(0.05)
            return
        time.sleep(0.01)
    raise AssertionError("service did not settle")


def test_prefetch_served_and_equal_to_direct():
    sh = shard(Fake([P.G.key(*READ)]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.text(sid, f"hi, about order {ORDER}")
    settle(sh, s)
    r = sh.call(sid, *READ)
    assert r["source"] == "cache" and r["attached"], r["source"]
    assert r["result"] == RT.call(RT.make_env("retail"), *READ)[0]


def test_write_invalidates_cache():
    sh = shard(Fake([P.G.key(*READ)]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.text(sid, f"cancel {ORDER}")
    settle(sh, s)
    assert P.G.key(*READ) in s.cache
    sh.call(sid, *WRITE)
    r = sh.call(sid, *READ)
    assert r["source"] == "exec" and json.loads(r["result"])["status"] == "cancelled", r["source"]


def test_write_during_prediction_drops_guesses():
    gate = threading.Event()
    sh = shard(Fake([P.G.key(*READ)], gate=gate))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.text(sid, "hello")  # predictor now blocked holding a pre-write snapshot
    time.sleep(0.1)
    sh.call(sid, *WRITE)
    gate.set()
    settle(sh, s)
    assert sh.stats.n["predictions_stale"] >= 1
    r = sh.call(sid, *READ)
    assert json.loads(r["result"])["status"] == "cancelled"


def test_prefetch_queued_before_write_is_skipped():
    sh = shard(Fake([]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    with s.lock:
        gen = s.gen
    s.lock.acquire()  # hold the session: the prefetch is submitted but cannot start
    f = sh.io.submit(sh.prefetch, s, P.G.key(*READ), READ[0], READ[1], gen)
    s.gen += 1  # a write lands first
    s.lock.release()
    f.result()
    assert P.G.key(*READ) not in s.cache and sh.stats.n["prefetch_skipped"] == 1


def test_writes_never_prefetched():
    sh = shard(Fake([P.G.key(*WRITE)]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.text(sid, "cancel please")
    settle(sh, s)
    r = sh.call(sid, *READ)
    assert json.loads(r["result"])["status"] == "pending" and all(e[2] == "memo" for e in s.cache.values())


def test_memory_bounded_in_long_session():
    sh = shard(Fake([]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    orders = list(s.env.tools.db.orders)[:300]
    for i in range(1500):
        sh.call(sid, "get_order_details", {"order_id": orders[i % len(orders)]})
        if i % 7 == 0:
            sh.text(sid, "x" * 500)
    settle(sh, s, timeout=30)
    assert len(s.window) == P.WINDOW and len(s.text) <= P.TEXT_CHARS and len(s.cache) <= P.CACHE_MAX


def test_sessions_evicted_lru():
    old = P.MAX_SESSIONS
    P.MAX_SESSIONS = 5
    try:
        sh = shard(Fake([]))
        sids = [sh.open("retail", 0, None)["sid"] for _ in range(8)]
        assert len(sh.sessions) == 5 and sids[0] not in sh.sessions and sids[-1] in sh.sessions
    finally:
        P.MAX_SESSIONS = old


def test_predictor_crash_does_not_stop_service():
    bad = Fake([P.G.key(*READ)], crash=True)
    sh = shard(bad)
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.text(sid, "hi")
    settle(sh, s)
    r = sh.call(sid, *READ)
    assert r["source"] == "exec" and r["result"].startswith("{")
    bad.crash = False
    sh.text(sid, "again")
    settle(sh, s)
    assert sh.stats.n["predict_errors"] >= 1 and sh.stats.n["predictions"] >= 1


def test_new_ticket_resets_state():
    sh = shard(Fake([P.G.key(*READ)]))
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.call(sid, *WRITE)
    sh.text(sid, "next")
    settle(sh, s)
    sh.task(sid, None, reset=True)
    assert not s.cache and not s.window and not s.text
    assert json.loads(sh.call(sid, *READ)["result"])["status"] == "pending"  # fresh environment


def test_repeat_read_served_from_memo_until_write():
    sh = shard(Fake([]))
    sid = sh.open("retail", 0, None)["sid"]
    assert sh.call(sid, *READ)["source"] == "exec"
    assert sh.call(sid, *READ)["source"] == "memo"
    sh.call(sid, *WRITE)
    r = sh.call(sid, *READ)
    assert r["source"] == "exec" and json.loads(r["result"])["status"] == "cancelled"


def test_reread_after_write_is_predicted():
    sh = shard(Fake([P.G.key(*READ)]))  # Fake skips keys in "called", like the real predictor
    sid = sh.open("retail", 0, None)["sid"]
    s = sh.get(sid)
    sh.call(sid, *READ)
    sh.call(sid, *WRITE)  # the earlier read is stale now: guessing it again must be allowed
    settle(sh, s)
    r = sh.call(sid, *READ)
    assert r["source"] == "cache" and json.loads(r["result"])["status"] == "cancelled", r["source"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"{len(tests)} passed")
