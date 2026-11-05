"""Stdio MCP server for multi-turn chats (spawned by tau_chat.py once per agent turn).

Usage: python3 -I tau_chat_mcp.py <tau-bench> <state dir> [predictor path]

The Claude CLI starts a fresh server for every agent turn, so everything that must survive a turn
lives in the state dir:
  text.txt      what the customer has said so far (written by tau_chat.py)
  state.json    the read results so far and the call keys already predicted
  cache/        one file per prefetched read: <crc>.json when done, <crc>.pending while running
  trace.jsonl   every call and prediction with timestamps (appended by server and runner)

With a predictor (gpred.Predictor), every successful read triggers a prediction from the results so
far plus the customer's words (tau_chat.py also predicts on every new customer message). Strict-gate
guesses scoring >= ATTACH_MIN run in parallel and, if done within PREFETCH_WAIT, are attached to the
result as compact JSON (every attached token is written to the prompt cache, the expensive part); loose-gate
guesses only go to the cache, so a wrong one costs a wasted read but never puts a wrong record in
front of the model. A call found in the cache returns at once, or waits only for what is running.
"""

import json
import os
import sys
import threading
import time
import zlib

if __name__ == "__main__":
    os.environ.setdefault("TAU_ROOT", sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_tools import READ, TOOLS, WAIT, execute, key  # noqa: E402

KEEP = set(json.load(open(os.environ["PRUNE"]))) if os.environ.get("PRUNE") else None  # learned field list (prune.py)


def trim(name, result):
    """Attached records carry only fields agents use later (prune.py, ~20-40 us); the agent's own calls are never trimmed."""
    if KEEP is None or not isinstance(result, (dict, list)):
        return result
    import prune
    return prune.project(name, result, KEEP) or result

ATTACH_MIN = float(os.environ.get("ATTACH_MIN", "0.9"))  # attach only near-certain guesses; the rest wait in the cache
ATTACH_MAX = int(os.environ.get("ATTACH_MAX", "3"))  # per lookup: a pile of records confuses the agent (pending vs delivered mix-ups)

NOTE = ("\n\n[prefetched lookups: read-only results fetched ahead of time, so you do not need to call these tools again. "
        "Nothing has been changed yet: still carry out every action the request needs with the tools.]\n")


def log(sd, row):
    with open(os.path.join(sd, "trace.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def load_state(sd):
    try:
        return json.load(open(os.path.join(sd, "state.json")))
    except (OSError, ValueError):
        return {"results": [], "seen": []}


def save_state(sd, st):
    tmp = os.path.join(sd, f"state.{os.getpid()}.{threading.get_ident()}.tmp")
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, os.path.join(sd, "state.json"))


def cpath(sd, k, ext):
    os.makedirs(os.path.join(sd, "cache"), exist_ok=True)
    return os.path.join(sd, "cache", f"{zlib.crc32(k.encode()):08x}.{ext}")


def prefetch(sd, name, args, why, trigger, box=None):
    """Run one predicted read and drop it in the cache (thread-safe, process-safe)."""
    k = key(name, args)
    open(cpath(sd, k, "pending"), "w").write(str(time.time()))
    out, t0, ms = execute(name, args)
    tmp = cpath(sd, k, f"{os.getpid()}.{threading.get_ident()}.tmp")
    json.dump({"key": k, "result": out}, open(tmp, "w"))
    os.replace(tmp, cpath(sd, k, "json"))
    log(sd, {"name": name, "args": args, "prefetched": True, "t": round(t0, 3), "ms": ms, "trigger": trigger, **why})
    if box is not None:
        box[k] = out
    return out


def start(sd, pred, text, trigger):
    """Predict from the state + customer text; start every guess. Returns [(name, args, thread, inject)]."""
    st = load_state(sd)
    tp = time.time()
    try:
        guesses = pred.predict(text, [tuple(r) for r in st["results"]], set(st["seen"]), explain=False)
    except Exception as e:  # a predictor bug must never take the tools down: log it and serve without prediction
        log(sd, {"name": "predictor_error", "t": round(tp, 3), "error": repr(e)[:300], "trigger": trigger})
        return []
    pred_ms = round((time.time() - tp) * 1000, 2)
    st["seen"] = sorted(set(st["seen"]) | {g["key"] for g in guesses})
    save_state(sd, st)
    out, box = [], {}
    for g in guesses:
        n = g["key"].split("{")[0]
        a = json.loads(g["key"][len(n):])
        attach = g["inject"] and g["score"] >= ATTACH_MIN
        why = {"score": g["score"], "gate": "inject" if attach else "cache", "pred_ms": pred_ms}
        t = threading.Thread(target=prefetch, args=(sd, n, a, why, trigger, box), daemon=True)
        t.start()
        out.append((n, a, t, attach, box))
    try:  # reasons are computed while the lookups run, so nobody waits on them
        te = time.time()
        for g in pred.explain_last():
            log(sd, {"name": "explanation", "key": g["key"], "t": round(te, 3), "why": g["why"], "explain_ms": round((time.time() - te) * 1000, 2)})
    except Exception as e:
        log(sd, {"name": "predictor_error", "t": round(te, 3), "error": repr(e)[:300], "trigger": trigger})
    return out


def from_cache(sd, k, timeout=10.0):
    t0 = time.time()
    while True:
        if os.path.exists(cpath(sd, k, "json")):
            return json.load(open(cpath(sd, k, "json")))["result"]
        if not os.path.exists(cpath(sd, k, "pending")) or time.time() - t0 > timeout:
            return None
        time.sleep(0.02)


def text_of(sd):
    try:
        return open(os.path.join(sd, "text.txt")).read()
    except OSError:
        return ""


def call(sd, pred, name, args):
    k = key(name, args)
    tw = time.time()
    out = from_cache(sd, k) if name in READ else None
    if out is not None:
        log(sd, {"name": name, "args": args, "t": round(tw, 3), "ms": round((time.time() - tw) * 1000, 2), "cached": True})
    else:
        out, t0, ms = execute(name, args)
        log(sd, {"name": name, "args": args, "t": round(t0, 3), "ms": ms})
    if name not in READ:  # a write may change what reads return
        for f in os.listdir(os.path.join(sd, "cache")) if os.path.isdir(os.path.join(sd, "cache")) else []:
            os.remove(os.path.join(sd, "cache", f))
    text = out if isinstance(out, str) else json.dumps(out)
    if not (pred and name in READ and not text.startswith("Error")):
        return text
    st = load_state(sd)
    st["results"].append([name, out])
    st["seen"] = sorted(set(st["seen"]) | {k})
    save_state(sd, st)
    extra, deadline = [], time.time() + WAIT
    for _ in range(3):  # chain: user -> orders -> products
        strict = [(n, a, t, box) for n, a, t, inj, box in start(sd, pred, text_of(sd), name) if inj][:ATTACH_MAX - len(extra)]
        got = 0
        for n, a, t, box in strict:
            t.join(max(deadline - time.time(), 0))
            r = box.get(key(n, a))
            if r is None or (isinstance(r, str) and r.startswith("Error")):
                continue
            st = load_state(sd)
            st["results"].append([n, r])
            save_state(sd, st)
            extra.append(f"{n}({json.dumps(a, separators=(',', ':'))}) -> {json.dumps(trim(n, r), separators=(',', ':'))}")
            got += 1
        if not got or len(extra) >= ATTACH_MAX:
            break
    return text + (NOTE + "\n".join(extra) if extra else "")


def reply(id_, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    sd = sys.argv[2]
    pred = None
    if len(sys.argv) > 3:
        import gpred
        pred = gpred.Predictor(sys.argv[3])
    for line in sys.stdin:
        msg = json.loads(line)
        method, id_ = msg.get("method"), msg.get("id")
        if method == "initialize":
            reply(id_, {"protocolVersion": msg["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "tau", "version": "1"}})
        elif method == "tools/list":
            reply(id_, {"tools": [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p, "required": list(p)}} for n, (d, p, _) in TOOLS.items()]})
        elif method == "tools/call":
            reply(id_, {"content": [{"type": "text", "text": call(sd, pred, msg["params"]["name"], msg["params"].get("arguments") or {})}]})
        elif id_ is not None:
            reply(id_, {})
