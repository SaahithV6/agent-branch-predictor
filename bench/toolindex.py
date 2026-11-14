"""Shortlist for a large tool catalog: which (tool, argument) slots could take a given value.

Scoring every tool for every prediction is impossible at a million tools. Instead each catalog
argument is indexed under (value class, name token): the class says what kind of value it accepts
(email, date, uuid, url, integer, id, code, word), the tokens come from its name (orderId -> order).
A candidate value is looked up with its own class and the tokens of where it came from: the result
field it sat in (order_id), or the words right before it in the customer's text ("my order #W123").
Lookups cost one dict probe per (class, token), so prediction cost follows the conversation, not the
catalog size.

Usage: python3 -I bench/toolindex.py <tools.jsonl> [N]   (build, report size and lookup cost)
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict

GENERIC = {"id", "ids", "get", "list", "the", "of", "by", "for", "a", "an", "and", "to", "in", "on", "is", "my", "your",
           "value", "values", "param", "params", "data", "request", "input", "body", "string", "number", "type", "info"}
CAP = 64  # slots per (class, token): beyond that a token says nothing about which tool is meant
TOP = 16  # slots returned per value


def tokens(name):
    """camelCase, snake_case and dotted names -> lower-case word stems."""
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))
    return {p.lower().rstrip("s") for p in parts if len(p) > 1} - GENERIC


def value_class(v):
    v = str(v)
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.\w+", v):
        return "email"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ][\d:.]+Z?)?", v):
        return "date"
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", v):
        return "uuid"
    if re.match(r"https?://", v):
        return "url"
    if re.fullmatch(r"-?\d+", v):
        return "integer"
    if re.fullmatch(r"[A-Z]{2,5}", v):
        return "code"
    if re.search(r"\d", v) and re.fullmatch(r"[#\w.:/-]+", v):
        return "id"
    if re.fullmatch(r"[A-Za-z][a-z]+", v):
        return "word"
    return None


def arg_classes(a):
    """Value classes one catalog argument accepts, from its schema, else its name."""
    fmt, typ, name = str(a.get("format", "")).lower(), str(a.get("type", "string")).lower(), str(a["name"]).lower()
    if a.get("enum"):
        return {value_class(e) for e in a["enum"] if value_class(e)} or {"word"}
    if "email" in fmt or name.endswith("email"):
        return {"email"}
    if "date" in fmt or name.endswith(("date", "time", "_at")) or name in ("date", "since", "until"):
        return {"date"}
    if fmt == "uuid" or name.endswith("uuid") or name.endswith("guid"):
        return {"uuid"}
    if fmt in ("uri", "url") or name.endswith(("url", "uri", "link")):
        return {"url"}
    if typ in ("integer", "number") or fmt in ("int32", "int64"):
        return {"integer"}
    if typ in ("boolean", "array", "object"):
        return set()
    if name.endswith(("id", "key", "number", "num", "code", "sku", "ref", "token")):
        return {"id", "integer", "uuid", "code"}
    if name.endswith(("name", "city", "country", "state", "status", "category")):
        return {"word", "code"}
    return set()  # free text: never a prefetch target on its own


class ToolIndex:
    def __init__(self, tools):
        """tools: iterable of catalog records (bench/catalog.py). Only read tools are indexed."""
        self.args, self.server = {}, {}
        self.index = defaultdict(list)
        for t in tools:
            if not t["read"]:
                continue
            tid = t["id"]
            need = [a for a in t["args"] if a.get("required")] or t["args"][:1]  # a call needs its required arguments
            self.args[tid] = tuple(a["name"] for a in need)
            self.server[tid] = t["server"]
            ttoks = tokens(t["name"])
            for a in need:
                classes = arg_classes(a)
                for c in classes:
                    for tok in tokens(a["name"]) or ttoks:
                        slots = self.index[(c, tok)]
                        if len(slots) < CAP:
                            slots.append((tid, a["name"]))

    def lookup(self, value, context, k=TOP):
        """Up to k (tool, argument) slots that could take value, most context tokens matched first;
        context: names/words describing where the value came from."""
        c = value_class(value)
        if not c:
            return []
        hits = Counter()
        for tok in tokens(" ".join(context)):
            hits.update(self.index.get((c, tok), ()))
        return [slot for slot, _ in hits.most_common(k)]


if __name__ == "__main__":
    import resource
    path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    tools = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        tools.append(json.loads(line))
    t0 = time.time()
    ix = ToolIndex(tools)
    build = time.time() - t0
    probes = [("#W5918442", ["order", "id"]), ("sofia@example.com", ["email"]), ("2024-05-20", ["flight", "date"]),
              ("JFK", ["origin"]), ("12345", ["zip", "code"]), ("yusuf_rossi_9620", ["user", "id"])]
    t0, hits = time.perf_counter(), []
    for _ in range(1000):
        hits = [len(ix.lookup(v, c)) for v, c in probes]
    us = (time.perf_counter() - t0) / 1000 / len(probes) * 1e6
    print(json.dumps({"tools": len(tools), "read_tools_indexed": len(ix.args), "keys": len(ix.index), "build_s": round(build, 1),
                      "lookup_us": round(us, 1), "hits_per_probe": hits,
                      "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}))
