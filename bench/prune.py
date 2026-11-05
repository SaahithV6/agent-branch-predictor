"""Learned field pruning for attached (prefetched) records: send only the fields agents use.

Usage: python3 -I bench/prune.py <tau-bench> [out.json]

Every token attached to the agent's context is written to the prompt cache and re-read on every
later turn, so the cheapest token is one never sent. From real transcripts we learn, per tool and
per field path, how often a field's value is used later in the conversation: passed as an argument
to a later tool call, or repeated in the agent's own replies (which is how decision fields like an
order's status show up: "your order is still pending"). Fields used in fewer than KEEP of results are
dropped from attachments. The agent's own tool calls are never pruned.

Scored on held-out customers (20%, by hash) per domain: size kept (characters of JSON) and recall of
used values (share of values the agent later used that survive pruning).
"""

import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tau_replay as T  # noqa: E402  (walk / idlike; reads sys.argv[1] = tau-bench root)
import traces as TR  # noqa: E402


def leaves(res):
    return [(pat, v) for pat, v, _ in T.walk(res)] if isinstance(res, (dict, list)) else []


def usage(trs):
    """(tool, path) -> [results containing the field, results where its value was used later]."""
    stats = {}
    for t in trs:
        said = t.get("agent_said", "").lower()
        for i, s in enumerate(t["steps"]):
            if not (s["read"] and s["ok"]):
                continue
            later_args = {str(v) for x in t["steps"][i + 1:] if isinstance(x["args"], dict) for v in x["args"].values()}
            later_said = said[s.get("said_at", 0):]
            seen = set()
            for pat, v in leaves(s["result"]):
                k = (s["name"], pat)
                used = v in later_args or (len(v) >= 3 and v.lower() in later_said)
                st = stats.setdefault(k, [0, 0])
                if k not in seen:
                    st[0] += 1
                    seen.add(k)
                st[1] += used  # counts uses; a list field can be used several times per result
    return stats


def keep_set(stats, keep):
    return {f"{tool}|{'/'.join(pat)}" for (tool, pat), (n, u) in stats.items() if n and u / n >= keep}


def project(tool, obj, keep, path=()):
    """obj with only kept leaves (structure preserved); None when nothing under it is kept."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            step = "{k}" if T.idlike(k) else k
            sub = project(tool, v, keep, path + (step,))
            if sub is not None or f"{tool}|{'/'.join(path + ('{k}',))}" in keep and step == "{k}":
                out[k] = sub if sub is not None else v
        return out or None
    if isinstance(obj, list):
        out = [x for x in (project(tool, v, keep, path + ("*",)) for v in obj) if x is not None]
        return out or None
    return obj if f"{tool}|{'/'.join(path)}" in keep else None


def evaluate(train, test, keep):
    ks = keep_set(usage(train), keep)
    full = kept = used = used_kept = 0
    for t in test:
        said = t.get("agent_said", "").lower()
        for i, s in enumerate(t["steps"]):
            if not (s["read"] and s["ok"] and isinstance(s["result"], (dict, list))):
                continue
            p = project(s["name"], s["result"], ks) or {}
            full += len(json.dumps(s["result"], separators=(",", ":")))
            kept += len(json.dumps(p, separators=(",", ":")))
            later_args = {str(v) for x in t["steps"][i + 1:] if isinstance(x["args"], dict) for v in x["args"].values()}
            later_said = said[s.get("said_at", 0):]
            pv = {v for _, v in leaves(p)}
            for _, v in leaves(s["result"]):
                if v in later_args or (len(v) >= 3 and v.lower() in later_said):
                    used += 1
                    used_kept += v in pv
    return {"size_kept": round(kept / max(full, 1), 3), "recall_used": round(used_kept / max(used, 1), 4), "fields_kept": len(ks)}


if __name__ == "__main__":
    report = {}
    for d in ("retail", "airline", "telecom"):
        trs = TR.load(d)
        test = [t for t in trs if zlib.crc32(t["group"].encode()) % 5 == 0]
        train = [t for t in trs if zlib.crc32(t["group"].encode()) % 5 != 0]
        report[d] = {}
        for keep in (0.005, 0.01, 0.02, 0.05, 0.1):
            r = evaluate(train, test, keep)
            report[d][keep] = r
            print(f"{d:8} keep>={keep:<6} attachment size {r['size_kept']:.0%} of full, used values retained {r['recall_used']:.1%}, fields {r['fields_kept']}", flush=True)
    if len(sys.argv) > 2:
        json.dump(report, open(sys.argv[2], "w"), indent=1)
