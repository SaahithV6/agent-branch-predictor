"""Multi-turn live run: a Claude Haiku agent talks to a simulated customer on tau-bench retail tasks.

Usage: python3 -I bench/tau_chat.py <tau-bench> <out.jsonl> [limit] [workers]
With PREDICTOR=results/gpred_retail_fold{fold} and FOLDS=results/gpred_retail_folds.json the tools
prefetch predicted reads (A/B arm); each task uses the model trained without its customer.

The customer is tau-bench's user simulator (its own prompt), played by Claude Haiku: it knows the
task instruction but reveals details only as the agent asks for them, and ends with ###STOP###.
The agent keeps one Claude session across turns (--resume) and gets the retail policy as its system
prompt. With a predictor, every new customer message also triggers a prediction before the agent
starts its turn, so lookups the conversation now points to are already running while it thinks.
Everything is timestamped: customer messages, model events, tool calls, prefetches with reasons.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TAU_ROOT", sys.argv[1])
from tau_replay import load_tasks  # noqa: E402  (reads sys.argv[1] = tau-bench root)
import tau_chat_mcp as M  # noqa: E402

root, out_path = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MAX_TURNS = 12
here = os.path.dirname(os.path.abspath(__file__))
policy = open(os.path.join(root, "tau_bench/envs/retail/wiki.md")).read()
token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or open(os.path.expanduser("~/Documents/qm-native/claude.token")).read().strip()
env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}
env.pop("ANTHROPIC_API_KEY", None)
GREETING = "Hi! How can I help you today?"
predictor_path = os.environ.get("PREDICTOR")
fold_of = json.load(open(os.environ["FOLDS"]))["fold_of_group"] if predictor_path else {}
if predictor_path:
    import gpred
tasks = load_tasks(os.path.join(root, "tau_bench/envs/retail/tasks_test.py"))[:limit]
done = {json.loads(l)["i"] for l in open(out_path)} if os.path.exists(out_path) else set()
lock = Lock()


def user_prompt(instruction):  # tau-bench's user simulator, verbatim rules
    return f"""You are a user interacting with an agent.

Instruction: {instruction}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


def claude(args, timeout=300):
    """Run the CLI with stream-json; return (timestamped events, final result event)."""
    events, result = [], {"error": "no result"}
    p = subprocess.Popen(["nice", "-n", "10", "claude", "-p", *args, "--output-format", "stream-json", "--verbose"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tempfile.gettempdir())
    timer = threading.Timer(timeout, p.kill)
    timer.start()
    for line in p.stdout:
        if not line.startswith("{"):
            continue
        ev = json.loads(line)
        if ev.get("type") == "result":
            result = ev
        else:
            m = ev.get("message") or {}
            blocks = m.get("content") if isinstance(m.get("content"), list) else []
            events.append({"t": round(time.time(), 3), "type": ev.get("type"), "usage": m.get("usage"),
                           "blocks": [{k: b.get(k) for k in ("type", "name", "input", "id", "tool_use_id") if b.get(k) is not None} for b in blocks]})
    p.wait()
    timer.cancel()
    if p.returncode and result.get("error") == "no result":
        result = {"error": "timeout" if p.returncode < 0 else p.stderr.read()[-300:]}
    return events, result


def customer(instruction, convo):
    lines = "\n".join(f"{'Agent' if r == 'agent' else 'You'}: {t}" for r, t in convo)
    _, res = claude([f"Conversation so far:\n{lines}\n\nWrite only your next message to the agent.", "--model", "claude-haiku-4-5",
                     "--system-prompt", user_prompt(instruction), "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--max-turns", "1"], 120)
    return (res.get("result") or "").strip() or "###STOP###"


def run(i, t):
    try:
        chat(i, t)
    except Exception:  # a crashed task should not vanish silently in the pool
        import traceback
        traceback.print_exc()


def chat(i, t):
    sd = tempfile.mkdtemp(prefix="tauchat_")
    pred_path = None
    if predictor_path:
        pred_path = predictor_path.format(fold=fold_of.get("retail:" + t["user"], 0))
        pred = gpred.Predictor(pred_path)
    args = ["-I", os.path.join(here, "tau_chat_mcp.py"), root, sd] + ([pred_path] if pred_path else [])
    cfg = os.path.join(sd, "mcp.json")
    json.dump({"mcpServers": {"tau": {"command": sys.executable, "args": args}}}, open(cfg, "w"))
    base = ["--model", "claude-haiku-4-5", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", cfg, "--tools", "",
            "--allowedTools", "mcp__tau", "--append-system-prompt", policy, "--max-turns", "20"]
    t0 = time.time()
    convo, events, usage, agent_s, turns, session, err, said = [("agent", GREETING)], [], {}, 0.0, 0, None, None, ""
    for _ in range(MAX_TURNS):
        msg = customer(t["instruction"], convo)
        events.append({"t": round(time.time(), 3), "type": "customer", "text": msg})
        if "###STOP###" in msg:
            break
        convo.append(("customer", msg))
        said += "\n" + msg
        open(os.path.join(sd, "text.txt"), "w").write(said)
        if pred_path:  # predict on the new customer message before the agent starts its turn
            M.start(sd, pred, said, "customer message")
        ta = time.time()
        ev, res = claude([msg] + base + (["--resume", session] if session else []))
        agent_s += time.time() - ta
        events += ev
        if res.get("error") or res.get("is_error"):
            err = res.get("error") or res.get("subtype")
            break
        session = res.get("session_id") or session
        turns += res.get("num_turns") or 0
        for k, v in (res.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        reply = (res.get("result") or "").strip()
        convo.append(("agent", reply))
        events.append({"t": round(time.time(), 3), "type": "agent_reply", "text": reply[:500]})
        if any(e.get("blocks") and any(b.get("name", "").endswith("transfer_to_human_agents") for b in e["blocks"]) for e in ev):
            break
    time.sleep(0.2)
    rows = [json.loads(l) for l in open(os.path.join(sd, "trace.jsonl"))] if os.path.exists(os.path.join(sd, "trace.jsonl")) else []
    rows.sort(key=lambda r: r.get("t") or 0)
    shutil.rmtree(sd, ignore_errors=True)
    row = {"i": i, "user": t["user"], "instruction": t["instruction"],
           "actions": [[r["name"], r["args"]] for r in rows if not r.get("prefetched") and r["name"] not in ("predictor_error", "explanation")],
           "predictor_errors": sum(r["name"] == "predictor_error" for r in rows),
           "prefetched": [[r["name"], r["args"]] for r in rows if r.get("prefetched")],
           "seconds": round(agent_s, 1), "seconds_total": round(time.time() - t0, 1), "num_turns": turns,
           "customer_turns": sum(r == "customer" for r, _ in convo), "usage": usage, "t0": round(t0, 3),
           "trace": rows, "events": events, "convo": convo, "error": err}
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"{i:4} {len(row['actions']):2} calls {row['customer_turns']:2} customer turns {row['seconds']:6.1f}s agent {err or ''}", flush=True)


with ThreadPoolExecutor(workers) as ex:
    for i, t in enumerate(tasks):
        if i not in done:
            ex.submit(run, i, t)
