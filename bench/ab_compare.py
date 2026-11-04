"""Compare two live runs of the same tasks (baseline vs prefetch).

Usage: python3 -I bench/ab_compare.py <tau-bench> base.jsonl prefetch.jsonl

Per task: wall seconds, model turns, tool calls the agent issued, output tokens, input tokens
(fresh + cache read + cache write), and whether the agent's write actions match tau-bench's answer
key (same names and arguments, list arguments order-insensitive) as a correctness proxy.
"""
import json, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__))
import tau_replay as T

key = T.load_tasks(os.path.join(sys.argv[1], "tau_bench/envs/retail/tasks_test.py"))
WRITES = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
          "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"}


def norm(acts):
    return sorted(json.dumps([n, {k: sorted(v) if isinstance(v, list) else v for k, v in a.items()}], sort_keys=True) for n, a in acts if n in WRITES)


def rows(path):
    return {r["i"]: r for r in map(json.loads, open(path)) if not r.get("error") and r["actions"]}


def metrics(r):
    u = r["usage"] or {}
    return {"seconds": r["seconds"], "turns": r["num_turns"], "tool_calls": len(r["actions"]), "out_tokens": u.get("output_tokens", 0),
            "in_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
            "correct": norm(r["actions"]) == norm(key[r["i"]]["actions"])}


a, b = rows(sys.argv[2]), rows(sys.argv[3])
ids = sorted(set(a) & set(b))
A, B = [metrics(a[i]) for i in ids], [metrics(b[i]) for i in ids]
print(f"paired tasks: {len(ids)}")
rng = random.Random(0)
for m in ("seconds", "turns", "tool_calls", "out_tokens", "in_tokens"):
    xa, xb = [x[m] for x in A], [x[m] for x in B]
    d = [y - x for x, y in zip(xa, xb)]
    boot = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(2000))
    print(f"{m:11} base mean {st.mean(xa):9.1f}  prefetch mean {st.mean(xb):9.1f}  change {st.mean(d) / st.mean(xa):+6.1%}  "
          f"(95% CI {boot[50] / st.mean(xa):+.1%}..{boot[1949] / st.mean(xa):+.1%})  median {st.median(xa):.1f} -> {st.median(xb):.1f}")
print(f"correct    base {sum(x['correct'] for x in A)}/{len(A)}  prefetch {sum(x['correct'] for x in B)}/{len(B)}")
