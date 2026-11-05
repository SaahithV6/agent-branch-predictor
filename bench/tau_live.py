"""Run a real agent (Claude Haiku via the Claude Code CLI, subscription auth) on tau-bench retail
tasks in solo mode and record every tool call it makes.

Usage: python3 -I bench/tau_live.py <tau-bench> <split: train|test> <out.jsonl> [limit] [workers]
With PREDICTOR=results/predictor_text.json the tool server prefetches predicted reads (A/B arm).
PREDICTOR may contain {fold}; FOLDS=results/live_nested.json then picks each task's fold, so a
task is only ever served rules trained without its customer.

Solo mode: the agent gets the retail policy (wiki.md) plus the customer's whole request up front,
and is told the customer pre-confirmed every action, so no user simulator is needed. Tasks
already in out.jsonl are skipped, so the run can be resumed. Needs ../qm-native/claude.token
(or CLAUDE_CODE_OAUTH_TOKEN).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

sys.path.insert(0, os.path.dirname(__file__))
from tau_replay import load_tasks  # noqa: E402  (tau_replay reads sys.argv[1] = tau-bench root)

root, split, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 3
here = os.path.dirname(os.path.abspath(__file__))
policy = open(os.path.join(root, "tau_bench/envs/retail/wiki.md")).read()
token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or open(os.path.expanduser("~/Documents/qm-native/claude.token")).read().strip()
SYSTEM = policy + "\n\n# Solo mode\nThe customer is not available for follow-up questions. Their full request is in the first message, written to them in second person. They have already authenticated and confirmed every action in advance. Use the tools to complete the request, then reply with a one-line summary."

tasks = load_tasks(os.path.join(root, f"tau_bench/envs/retail/tasks_{split}.py"))[:limit]
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["i"] for l in open(out_path)}
lock = Lock()


def run(i, t):
    trace = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    args = ["-I", os.path.join(here, "tau_mcp.py"), root, trace]
    if os.environ.get("PREDICTOR"):
        path = os.environ["PREDICTOR"]
        if "{fold}" in path:
            path = path.format(fold=json.load(open(os.environ["FOLDS"]))["fold_of_user"][t["user"]])
        p = json.load(open(path))
        json.dump({**p, "instruction": t["instruction"], "model_file": os.path.abspath(path.removesuffix(".json") + ".txt")}, open(trace + ".pred.json", "w"))
        args.append(trace + ".pred.json")
    cfg = {"mcpServers": {"tau": {"command": sys.executable, "args": args}}}
    cfg_path = trace + ".mcp.json"
    json.dump(cfg, open(cfg_path, "w"))
    env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}
    env.pop("ANTHROPIC_API_KEY", None)
    cmd = ["nice", "-n", "10", "claude", "-p", "Customer request:\n" + t["instruction"], "--model", "claude-haiku-4-5",
           "--setting-sources", "", "--strict-mcp-config", "--mcp-config", cfg_path, "--tools", "",
           "--allowedTools", "mcp__tau", "--append-system-prompt", SYSTEM, "--max-turns", "40", "--output-format", "stream-json", "--verbose"]
    t0 = time.time()
    events, meta = [], {"error": "no result"}
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tempfile.gettempdir())
        timer = threading.Timer(300, p.kill)
        timer.start()
        for line in p.stdout:  # timestamp every stream event: model messages, tool uses, tool results
            if not line.startswith("{"):
                continue
            ev = json.loads(line)
            if ev.get("type") == "result":
                meta = ev
            else:
                m = ev.get("message") or {}
                blocks = m.get("content") if isinstance(m.get("content"), list) else []
                events.append({"t": round(time.time(), 3), "type": ev.get("type"), "usage": m.get("usage"),
                               "blocks": [{k: b.get(k) for k in ("type", "name", "input", "id", "tool_use_id") if b.get(k) is not None} for b in blocks]})
        p.wait()
        timer.cancel()
        if p.returncode and meta.get("error") == "no result":
            meta = {"error": "timeout" if p.returncode < 0 else p.stderr.read()[-500:]}
    except OSError as e:
        meta = {"error": str(e)}
    calls = [json.loads(l) for l in open(trace)] if os.path.getsize(trace) else []
    for f in (trace, cfg_path, trace + ".pred.json"):
        if os.path.exists(f):
            os.unlink(f)
    row = {"i": i, "split": split, "user": t["user"], "instruction": t["instruction"],
           "actions": [[c["name"].removeprefix("mcp__tau__"), c["args"]] for c in calls if not c.get("prefetched")],
           "prefetched": [[c["name"], c["args"]] for c in calls if c.get("prefetched")],
           "seconds": round(time.time() - t0, 1), "usage": meta.get("usage"), "cost_usd": meta.get("total_cost_usd"),
           "num_turns": meta.get("num_turns"), "t0": round(t0, 3), "trace": calls, "events": events, "result": (meta.get("result") or "")[:300], "error": meta.get("error") or (meta.get("is_error") and meta.get("subtype"))}
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"{i:4} {len(row['actions']):2} calls {row['seconds']:6.1f}s {row['error'] or ''}", flush=True)


with ThreadPoolExecutor(workers) as ex:
    for i, t in enumerate(tasks):
        if i not in done:
            ex.submit(run, i, t)
