"""Stdio MCP server exposing tau-bench's retail tools over the benchmark's own data.

Usage (spawned by bench/tau_live.py): python3 -I tau_mcp.py <tau-bench checkout> <trace.jsonl> [predictor.json]

Reads return what tau-bench's tools return; writes are checked loosely and return "ok" without
changing the data (each task gets a fresh process, so nothing leaks between tasks). Every call is
appended to the trace file as {"name", "args"}.

With predictor.json (rules + config + the task's request text, written by tau_live.py), each read
result also carries the predicted follow-up reads, already executed and marked "prefetched", so
the agent can skip those calls. Prefetches are logged to the trace with "prefetched": true.
"""

import json
import os
import sys

root, trace = sys.argv[1], sys.argv[2]
pred = None
if len(sys.argv) > 3:
    import ast
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tau_replay as T
    pred = json.load(open(sys.argv[3]))
    keep = {ast.literal_eval(k): tuple(v) for k, v in pred["keep"].items()}
    task = {"instruction": pred["instruction"], "lower": pred["instruction"].lower()}
    seen, results = set(), []
D = {n: json.load(open(os.path.join(root, "tau_bench/envs/retail/data", f"{n}.json"))) for n in ("users", "orders", "products")}
S = {"type": "string"}
L = {"type": "array", "items": S}
ADDR = {k: S for k in ("address1", "address2", "city", "state", "country", "zip")}


def find_email(email):
    return next((u for u, p in D["users"].items() if p["email"].lower() == email.lower()), "Error: user not found")


def find_name_zip(first_name, last_name, zip):
    return next((u for u, p in D["users"].items() if p["name"]["first_name"].lower() == first_name.lower() and p["name"]["last_name"].lower() == last_name.lower() and p["address"]["zip"] == zip), "Error: user not found")


def lookup(table, key):
    return D[table].get(key) or f"Error: {table[:-1]} not found"


def write(order_field=None):
    def run(**kw):
        if order_field and kw.get(order_field) not in D["orders"]:
            return "Error: order not found"
        return {"status": "ok", "applied": kw}
    return run


TOOLS = {
    "find_user_id_by_email": ("Find user id by email.", {"email": S}, find_email),
    "find_user_id_by_name_zip": ("Find user id by first name, last name, and zip code.", {"first_name": S, "last_name": S, "zip": S}, find_name_zip),
    "get_user_details": ("Get the details of a user, including their orders.", {"user_id": S}, lambda user_id: lookup("users", user_id)),
    "get_order_details": ("Get the status and details of an order. Order ids look like '#W0000000'.", {"order_id": S}, lambda order_id: lookup("orders", order_id)),
    "get_product_details": ("Get the inventory details of a product.", {"product_id": S}, lambda product_id: lookup("products", product_id)),
    "list_all_product_types": ("List the name and product id of all product types.", {}, lambda: {p["name"]: p["product_id"] for p in D["products"].values()}),
    "calculate": ("Calculate the result of a mathematical expression.", {"expression": S}, lambda expression: "ok"),
    "cancel_pending_order": ("Cancel a pending order. Reason is 'no longer needed' or 'ordered by mistake'.", {"order_id": S, "reason": S}, write("order_id")),
    "exchange_delivered_order_items": ("Exchange items in a delivered order for new items of the same product type.", {"order_id": S, "item_ids": L, "new_item_ids": L, "payment_method_id": S}, write("order_id")),
    "modify_pending_order_address": ("Modify the shipping address of a pending order.", {"order_id": S, **ADDR}, write("order_id")),
    "modify_pending_order_items": ("Modify items in a pending order to new items of the same product type. Can only be called once per order.", {"order_id": S, "item_ids": L, "new_item_ids": L, "payment_method_id": S}, write("order_id")),
    "modify_pending_order_payment": ("Modify the payment method of a pending order.", {"order_id": S, "payment_method_id": S}, write("order_id")),
    "modify_user_address": ("Modify the default address of a user.", {"user_id": S, **ADDR}, write()),
    "return_delivered_order_items": ("Return some items of a delivered order.", {"order_id": S, "item_ids": L, "payment_method_id": S}, write("order_id")),
    "transfer_to_human_agents": ("Transfer the user to a human agent with a summary of the issue.", {"summary": S}, write()),
}


def reply(id_, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    method, id_ = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(id_, {"protocolVersion": msg["params"].get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "tau", "version": "1"}})
    elif method == "tools/list":
        reply(id_, {"tools": [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": p, "required": list(p)}} for n, (d, p, _) in TOOLS.items()]})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        with open(trace, "a") as f:
            f.write(json.dumps({"name": name, "args": args}) + "\n")
        try:
            out = TOOLS[name][2](**args)
        except Exception as e:  # bad arguments from the model are part of the benchmark
            out = f"Error: {e}"
        text = out if isinstance(out, str) else json.dumps(out)
        if pred and name in T.READS["retail"] and not text.startswith("Error"):
            seen.add(T.call_key(name, args))
            results.append((name, out))
            extra = []
            for _ in range(3):  # chain: user -> orders -> products
                new = sorted(T.predict(keep, task, results, pred["cfg"]) - seen)[:8 - len(extra)]
                if not new:
                    break
                for k in new:
                    n, a = k[:k.index("{")], json.loads(k[k.index("{"):])
                    seen.add(k)
                    r = TOOLS[n][2](**a)
                    if isinstance(r, str) and r.startswith("Error"):
                        continue
                    results.append((n, r))
                    extra.append(f"{n}({json.dumps(a)}) -> {json.dumps(r)}")
                    with open(trace, "a") as f:
                        f.write(json.dumps({"name": n, "args": a, "prefetched": True}) + "\n")
            if extra:
                text += "\n\n[prefetched: results of reads you are likely to need next; use them instead of calling these tools again]\n" + "\n".join(extra)
        reply(id_, {"content": [{"type": "text", "text": text}]})
    elif id_ is not None:
        reply(id_, {})
