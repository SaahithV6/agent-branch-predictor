"""Real tool backends: tau2-bench's own environments (retail, airline, telecom), their own tool code
and databases, writes included. Each session gets its own environment, initialized to its task's
starting state the way tau2's evaluator does it, so a write really changes what later reads return.

    env = make_env(domain, task)            # task: the tau2 task dict stored with each trajectory
    out, write = call(env, name, args)      # out: the JSON string tau2 hands the agent

Also loads the tau2 trajectories with their task definitions (traces.py drops them), and checks
fidelity: replaying a trajectory's calls in a fresh environment should reproduce what was recorded.

Usage: python3 -I bench/realtools.py [domain ...]     fidelity report
"""

import glob
import json
import os
import sys

from loguru import logger

logger.remove()  # tau2 logs every tool response at DEBUG
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.registry import registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traces as TR  # noqa: E402

DOMAINS = ("retail", "airline", "telecom")


def make_env(domain, task=None):
    env = registry.get_env_constructor(domain)()
    if task:
        st = Task.model_validate(task).initial_state
        if st:
            env.set_state(st.initialization_data, st.initialization_actions, [])
    return env


def call(env, name, args):
    """(result string, is_write). Errors come back as "Error: ..." strings, as tau2 returns them."""
    write = env._is_mutating_tool(name)  # tau2's own flag; unknown tools count as writes
    try:
        out = env.to_json_str(env.make_tool_call(name, requestor="assistant", **args))
        env.sync_tools()
    except Exception as e:
        out = f"Error: {e}"
    return out, write


def load(domain):
    """traces.load_tau2(domain) rows, each with its "task" (initial state and all)."""
    tasks, rows = {}, []
    for f in sorted(glob.glob(os.path.join(TR.DATA, "tau2", f"*_{domain}_*.json"))):
        j = json.load(open(f))
        tasks.update({(f, t["id"]): t for t in j["tasks"]})
    _finish = TR.finish
    order = []  # load_tau2 walks files and simulations in this same order
    for f in sorted(glob.glob(os.path.join(TR.DATA, "tau2", f"*_{domain}_*.json"))):
        order += [(f, s["task_id"]) for s in json.load(open(f))["simulations"] if any_call(s)]
    it = iter(order)
    TR.finish = lambda *a: {**_finish(*a), "task": tasks[next(it)]}
    try:
        rows = TR.load_tau2(domain)
    finally:
        TR.finish = _finish
    return rows


def any_call(sim):
    """load_tau2 keeps a simulation only when an assistant tool call got an answer."""
    ids = {c.get("id") for m in sim["messages"] if m.get("role") == "assistant"
           for c in m.get("tool_calls") or [] if c.get("requestor", "assistant") == "assistant"}
    return any(m.get("role") == "tool" and m.get("id") in ids for m in sim["messages"])


def same(recorded, out):
    return TR.parse(out) == recorded


def fidelity(domain, limit=None):
    """Share of recorded reads a fresh replay reproduces exactly. Telecom's customer also changes
    the device and line state with their own tools between agent calls; those calls are not replayed."""
    rows = load(domain)[:limit]
    ok = n = errs = 0
    for t in rows:
        env = make_env(domain, t["task"])
        for s in t["steps"]:
            out, _ = call(env, s["name"], s["args"])
            if s["read"]:
                n += 1
                ok += same(s["result"], out)
                errs += out.startswith("Error")
    return {"domain": domain, "chats": len(rows), "reads": n, "reproduced": round(ok / max(n, 1), 4), "errors": errs}


if __name__ == "__main__":
    for d in sys.argv[1:] or DOMAINS:
        print(json.dumps(fidelity(d)), flush=True)
