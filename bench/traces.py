"""Load real agent chat transcripts into one normalized trace format.

Sources (downloaded into data/, never committed):
  tau2   data/tau2/*_{retail,airline,telecom}_*.json   tau2-bench leaderboard dumps (Sierra's public S3
         bucket): multi-turn chats between an agent model and a simulated customer, with tool calls,
         JSON results and per-message timestamps.
  tau1   ~/Documents/tau-bench/historical_trajectories/*.json   tau-bench's published gpt-4o and
         sonnet-3.5 transcripts (no timestamps).

Trace: {"domain", "model", "source", "group", "steps": [...], "ncalls"}. Each step is one agent tool
call: {"name", "args", "result", "ok", "read", "text", "t_text", "t_call", "t_result"}, where "text" is
what the customer had said before the call (its last message at "t_text") and "read" marks tools on the domain's read-only allowlist (the
only ones it is safe to run speculatively). "group" is the customer's account id, so
splits never put the same customer in train and test. "agent_said" is everything the agent wrote;
each step's "said_at" marks how much of it came before the call.
"""

import glob
import json
import os
from datetime import datetime

READ = {  # operator allowlist: lookups with no side effects
    "retail": {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details",
               "get_product_details", "get_item_details", "list_all_product_types"},
    "airline": {"get_user_details", "get_reservation_details", "search_direct_flight", "search_onestop_flight",
                "get_flight_status", "list_all_airports"},
    "telecom": {"get_customer_by_phone", "get_customer_by_id", "get_details_by_id", "get_data_usage", "get_bills_for_customer"},
}
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def parse(content):
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except ValueError:
        return content


def ts(s):
    return datetime.fromisoformat(s).timestamp() if s else None


def customer(steps, fallback):
    """The customer's account id, so the same person is grouped across models and sources."""
    for s in steps:
        if not s["ok"]:
            continue
        if s["name"] == "get_user_details":
            return s["args"].get("user_id")
        if s["name"].startswith("find_user_id") and isinstance(s["result"], str):
            return s["result"]
        if s["name"].startswith("get_customer_by") and isinstance(s["result"], dict):
            return s["result"].get("customer_id")
    return fallback


def finish(domain, model, source, steps, task_id):
    # telecom reuses one customer for every task, so its scenarios are split by task instead
    group = f"{domain}:" + (f"task{task_id}" if domain == "telecom" else str(customer(steps, f"{source}:{task_id}")))
    return {"domain": domain, "model": model, "source": source, "group": group, "steps": steps, "ncalls": len(steps)}


def load_tau2(domain):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "tau2", f"*_{domain}_*.json"))):
        model = os.path.basename(f).split(f"_{domain}_")[0]
        for sim in json.load(open(f))["simulations"]:
            text, steps, open_calls, t_text, said = "", [], {}, None, ""
            for m in sim["messages"]:
                role = m.get("role")
                if role == "user" and m.get("content"):
                    text += "\n" + m["content"]
                    t_text = ts(m.get("timestamp"))
                if role == "assistant" and isinstance(m.get("content"), str):
                    said += "\n" + m["content"]
                for c in m.get("tool_calls") or []:
                    if role == "assistant" and c.get("requestor", "assistant") == "assistant":
                        open_calls[c.get("id")] = {"name": c["name"], "args": c.get("arguments") or {}, "text": text, "t_text": t_text, "t_call": ts(m.get("timestamp"))}
                if role == "tool" and m.get("id") in open_calls:
                    s = open_calls.pop(m["id"])
                    r = parse(m.get("content"))
                    ok = not m.get("error") and not (isinstance(r, str) and r.startswith("Error"))
                    steps.append({**s, "result": r, "ok": ok, "read": s["name"] in READ[domain], "t_result": ts(m.get("timestamp")), "said_at": len(said)})
            if steps:
                out.append({**finish(domain, model, "tau2", steps, sim["task_id"]), "agent_said": said})
    return out


def load_tau1(domain):
    out = []
    for f in sorted(glob.glob(os.path.expanduser(f"~/Documents/tau-bench/historical_trajectories/*-{domain}.json"))):
        model = os.path.basename(f).removesuffix(f"-{domain}.json")
        for r in json.load(open(f)):
            text, steps, open_calls, said = "", [], {}, ""
            for m in r["traj"]:
                role = m.get("role")
                if role == "user" and m.get("content"):
                    text += "\n" + m["content"]
                if role == "assistant" and isinstance(m.get("content"), str):
                    said += "\n" + m["content"]
                for c in m.get("tool_calls") or []:
                    fn = c["function"]
                    open_calls[c["id"]] = {"name": fn["name"], "args": parse(fn.get("arguments")) or {}, "text": text, "t_text": None, "t_call": None}
                if role == "tool" and m.get("tool_call_id") in open_calls:
                    s = open_calls.pop(m["tool_call_id"])
                    res = parse(m.get("content"))
                    ok = not (isinstance(res, str) and res.startswith("Error"))
                    steps.append({**s, "result": res, "ok": ok, "read": s["name"] in READ[domain], "t_result": None, "said_at": len(said)})
            if steps:
                out.append({**finish(domain, model, "tau1", steps, r["task_id"]), "agent_said": said})
    return out


def load(domain, sources=("tau2", "tau1")):
    return (load_tau2(domain) if "tau2" in sources else []) + (load_tau1(domain) if "tau1" in sources and domain != "telecom" else [])


if __name__ == "__main__":
    for d in READ:
        tr = load(d)
        reads = sum(s["read"] and s["ok"] for t in tr for s in t["steps"])
        print(d, len(tr), "transcripts", len({t["group"] for t in tr}), "customers", sum(t["ncalls"] for t in tr), "calls", reads, "ok reads",
              sorted({t["model"] for t in tr}))
