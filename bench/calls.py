"""Tool-call examples from public agent trajectories, keyed to the 1M-tool catalog.

Each example is one real call and what came before it: the user's request, the calls already made
(with their results), and the call that came next. Tools are named by catalog id, so a model trained
here can be asked to pick the next call out of the whole catalog, and test splits can hold out
whole servers (APIs) the model has never seen called.

    python3 -I bench/calls.py toolbench    -> data/calls/toolbench.jsonl
"""

import ast
import json
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "calls"
CATALOG = ROOT / "data" / "catalogs" / "tools_toolbench.jsonl"
RESULT_CHARS = 2000  # a result is kept up to this long: enough for its ids and names


def norm(s):
    # ToolBench's own standardize(): lower case, non-alphanumerics -> "_", collapsed
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", str(s).lower())).strip("_")


def stream(path, chunk=1 << 22):
    """Objects of a top-level JSON array, read a chunk at a time (the ToolBench file is 2 GB)."""
    dec, buf, start = json.JSONDecoder(), "", True
    with open(path) as fh:
        while True:
            more = fh.read(chunk)
            buf += more
            if start:
                buf = buf.lstrip().lstrip("[")
                start = False
            while True:
                buf = buf.lstrip().lstrip(",").lstrip()
                if not buf or buf[0] == "]":
                    break
                try:
                    obj, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break  # incomplete object: read more
                yield obj
                buf = buf[end:]
            if not more:
                return


def split_of(server):
    """Stable 80/10/10 train/dev/test by server: a held-out API is never seen in training."""
    h = zlib.crc32(server.encode()) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def toolbench_index():
    idx = {}
    for line in open(CATALOG):
        t = json.loads(line)
        k = f"{norm(t['name'])}_for_{norm(t['server'])}"
        idx.setdefault(k, t["id"])
        idx.setdefault(k[:64], t["id"])  # OpenAI function names stop at 64 characters
    return idx


SPEC = re.compile(r'This is the subfunction for tool "([^"]+)"[^"]*?The description of this function is: "(.*)"$', re.S)


def specs(system):
    """The trajectory's own tool definitions, from its system prompt: [(fn name, server, desc, args)]."""
    i = system.find("[", system.find("you have access to the following APIs"))
    try:
        fns = ast.literal_eval(system[i:system.rindex("]") + 1])
    except (ValueError, SyntaxError):
        return []
    out = []
    for f in fns:
        m = SPEC.match(f.get("description", ""))
        if not m:
            continue
        props = f.get("parameters", {}).get("properties", {}) or {}
        req = set(f.get("parameters", {}).get("required", []) or [])
        args = [{"name": k, "type": v.get("type", "string"), "required": k in req,
                 **({"example": str(v["example_value"])} if "example_value" in v else {})} for k, v in props.items()]
        out.append((f["name"], m.group(1), m.group(2), args))
    return out


ACTION = re.compile(r"Action:\s*(\S+)\s*\nAction Input:\s*(.*)", re.S)


def toolbench():
    idx = toolbench_index()
    best = {}  # query -> longest record (the file holds one record per step of each trajectory)
    for path in sorted(DATA.glob("Yhyu13_ToolBench_*_train.json")) + sorted(DATA.glob("Yhyu13_ToolBench_*_eval.json")):
        for rec in stream(path):
            conv = rec["conversations"]
            q = next((c["value"] for c in conv if c["from"] == "user"), "").replace("Begin!", "").strip()
            if q and len(conv) > len(best.get(q, ())):
                best[q] = conv
    n = {"trajectories": len(best), "calls": 0, "mapped": 0, "added_tools": 0}
    out = open(DATA / "toolbench.jsonl", "w")
    added = open(DATA / "tools_toolbench_traj.jsonl", "w")  # trajectory tools the catalog snapshot lacks
    examples = {}  # tool id -> {arg: documented example value}: part of the tool's definition, like its description
    for q, conv in best.items():
        for fn, server, desc, args in specs(conv[0]["value"]):
            if fn in idx:
                examples.setdefault(idx[fn], {a["name"]: a["example"] for a in args if "example" in a})
            if fn not in idx:
                api = fn[:-len(f"_for_{norm(server)}")] if fn.endswith(f"_for_{norm(server)}") else fn
                t = {"source": "toolbench_traj", "server": server, "name": api, "read": True, "args": args,
                     "desc": desc[:1000], "id": f"toolbench/{server}/{api}"}
                idx[fn] = t["id"]
                examples[t["id"]] = {a["name"]: a["example"] for a in args if "example" in a}
                added.write(json.dumps(t) + "\n")
                n["added_tools"] += 1
        session = sorted({idx[fn] for fn, *_ in specs(conv[0]["value"]) if fn in idx})
        prior = []
        for i, c in enumerate(conv):
            if c["from"] != "assistant":
                continue
            m = ACTION.search(c["value"])
            if not m or m.group(1) == "Finish":
                continue
            name = m.group(1).strip()
            try:
                args = json.loads(m.group(2).strip() or "{}")
            except json.JSONDecodeError:
                args = None
            result = conv[i + 1]["value"][:RESULT_CHARS] if i + 1 < len(conv) and conv[i + 1]["from"] == "function" else ""
            tool = idx.get(name) or idx.get(name[:64])
            n["calls"] += 1
            if tool and isinstance(args, dict):
                n["mapped"] += 1
                server = tool.split("/")[1]
                out.write(json.dumps({"src": "toolbench", "conv": zlib.crc32(q.encode()), "split": split_of(server),
                                      "query": q, "session": session, "prior": list(prior), "tool": tool, "args": args}) + "\n")
            prior.append({"tool": tool or name, "args": args if isinstance(args, dict) else {}, "result": result})
    out.close()
    added.close()
    json.dump({k: v for k, v in examples.items() if v}, open(DATA / "tool_examples.json", "w"))
    return n


def py_calls(s):
    """'[A B(x="1", y=2), C()]' -> [("A B", {"x": "1", "y": 2}), ("C", {})]; None if it does not parse."""
    s, out, i = s.strip()[1:-1], [], 0
    while i < len(s):
        j = s.find("(", i)
        if j < 0:
            break
        name, depth, q, k = s[i:j].strip(" ,"), 0, None, j
        for k in range(j, len(s)):  # the matching ")", skipping quoted text
            ch = s[k]
            if q:
                q = None if ch == q and s[k - 1] != "\\" else q
            elif ch in "\"'":
                q = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if not depth:
                    break
        src = s[j + 1:k]
        for txt in (src, re.sub(r"\b(true|false|null)\b", lambda m: {"null": "None"}.get(m[0], m[0].title()), src)):
            try:
                out.append((name, {a.arg: ast.literal_eval(a.value)
                                   for a in ast.parse(f"f({txt})", mode="eval").body.keywords}))
                break
            except (SyntaxError, ValueError):
                pass
        else:
            return None
        i = k + 1
    return out


def toolace():
    """ToolACE (Team-ACE): 11k dialogs over ~26k synthetic APIs, one API per server. Mostly single calls."""
    d = json.load(open(DATA / "Team-ACE_ToolACE_data.json"))
    tools, n = {}, {"dialogs": len(d), "calls": 0, "mapped": 0}
    out = open(DATA / "toolace.jsonl", "w")
    for di, x in enumerate(d):
        s = x["system"]
        i = s.find("[", s.find("can invoke:"))
        try:
            fns = json.loads(s[i:s.rindex("]. \nShould") + 1])
        except ValueError:
            continue
        ids = {}
        for f in fns:
            props = (f.get("parameters") or {}).get("properties") or {}
            req = set((f.get("parameters") or {}).get("required") or [])
            tid = f"toolace/{f['name']}/{f['name']}"
            ids[f["name"]] = tid
            tools.setdefault(tid, {"source": "toolace", "server": f["name"], "name": f["name"], "read": True,
                                   "args": [{"name": k, "type": v.get("type", "string"), "required": k in req}
                                            for k, v in props.items() if isinstance(v, dict)],
                                   "desc": str(f.get("description", ""))[:1000], "id": tid})
        session, prior, users = sorted(ids.values()), [], []
        conv = x["conversations"]
        for ci, c in enumerate(conv):
            if c["from"] == "user":
                users.append(c["value"])
            if c["from"] != "assistant" or not c["value"].startswith("["):
                continue
            calls = py_calls(c["value"]) or []
            result = conv[ci + 1]["value"][:RESULT_CHARS] if ci + 1 < len(conv) and conv[ci + 1]["from"] == "tool" else ""
            step = []
            for name, args in calls:
                n["calls"] += 1
                if name not in ids:
                    continue
                n["mapped"] += 1
                # parallel calls in one turn share the same prior
                out.write(json.dumps({"src": "toolace", "conv": zlib.crc32(f"toolace{di}".encode()),
                                      "split": split_of(name), "query": "\n".join(users), "session": session,
                                      "prior": list(prior), "tool": ids[name], "args": args}) + "\n")
                step.append({"tool": ids[name], "args": args, "result": result})
            prior += step
    out.close()
    with open(DATA / "tools_toolace.jsonl", "w") as f:
        for t in tools.values():
            f.write(json.dumps(t) + "\n")
    n["tools"] = len(tools)
    return n


if __name__ == "__main__":
    assert py_calls('[A B(x="a, (b)", y=2), C(), D(z=[1, 2], w=true)]') == [("A B", {"x": "a, (b)", "y": 2}), ("C", {}),
                                                                         ("D", {"z": [1, 2], "w": True})]
    print({"toolbench": toolbench, "toolace": toolace}[sys.argv[1]]())
