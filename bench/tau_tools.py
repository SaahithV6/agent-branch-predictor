"""tau-bench retail tools over the benchmark's own data, shared by the tool servers (tau_mcp.py,
tau_chat_mcp.py). Reads return what tau-bench returns; writes are checked loosely and return "ok"
without changing the data. TOOL_LATENCY adds a simulated network delay to every call.
"""

import json
import math
import os
import random
import time
import zlib

root = os.environ.get("TAU_ROOT") or os.path.expanduser("~/Documents/tau-bench")
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


LAT = float(os.environ.get("TOOL_LATENCY", "0"))  # median seconds per call; 0 = tau's in-memory speed
WAIT = float(os.environ.get("PREFETCH_WAIT", "1.5"))  # max seconds a read waits for its prefetches
READ = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details", "get_product_details", "list_all_product_types"}


def key(name, args):
    return name + json.dumps(args, sort_keys=True)


def execute(name, args):
    """Run a tool with simulated network latency: lognormal around LAT, seeded by the call itself so
    every arm of an A/B sees the same delay for the same call."""
    t0 = time.time()
    if LAT:
        time.sleep(LAT * math.exp(0.5 * random.Random(zlib.crc32(key(name, args).encode())).gauss(0, 1)))
    try:
        out = TOOLS[name][2](**args)
    except Exception as e:  # bad arguments from the model are part of the benchmark
        out = f"Error: {e}"
    return out, t0, round((time.time() - t0) * 1000, 2)


