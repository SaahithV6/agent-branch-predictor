import json, os, re, sqlite3, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

TOKEN = open(os.path.join(os.path.dirname(__file__), "..", "claude.token")).read().strip()
SYSTEM = ("You predict tool calls. Given an assistant's tools and a user request, output the read-only tool calls the "
          "assistant will make to answer it, with exact argument values. Output only a JSON array of {\"name\":..., \"args\":{...}}. "
          "Omit optional arguments unless the request needs them. Include only calls you are confident about.")


def key(name, args):
    lower = lambda v: v.lower() if isinstance(v, str) else v
    return name + json.dumps({k: lower(v) for k, v in args.items()}, sort_keys=True)


def traces(db, prefix):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for (b,) in c.execute("select body from records where kind='tool-trace' order by rowid"):
        d = json.loads(b)
        calls = {key(x["name"], x["args"]) for x in d["calls"] if x["name"].startswith(prefix) and x["name"] not in ("sim_page_oncall", "sim_create_ticket", "sim_post_status_update")}
        if calls:
            out.append((d["query"], calls))
    return out


def draft(model, tools, query):
    prompt = f"Tools:\n{json.dumps(tools)}\n\nUser request: {query}"
    for attempt in range(3):
        r = subprocess.run(["claude", "-p", "--setting-sources", "", "--strict-mcp-config", "--model", model, "--tools", "", "--no-session-persistence", "--system-prompt", SYSTEM, prompt],
                           capture_output=True, text=True, timeout=120, env={**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": TOKEN}, cwd="/tmp")
        m = re.search(r"\[.*\]", r.stdout, re.S)
        if m:
            try:
                return {key(c["name"], c.get("args", {})) for c in json.loads(m.group(0))}
            except Exception:
                pass
    print(f"drafter failed: {r.stderr[:200]!r} {r.stdout[:200]!r}", file=sys.stderr)
    return set()


def main():
    model = sys.argv[1]
    sets = [("real", traces(os.path.expanduser("~/Documents/qm-native/real2.db"), "real_")[:30], json.load(open("tools-real.json"))),
            ("sim", traces(os.path.expanduser("~/Documents/qm-native/bench2.db"), "sim_")[:30], json.load(open("tools-sim.json")))]
    for name, ts, tools in sets:
        with ThreadPoolExecutor(4) as ex:
            preds = list(ex.map(lambda t: draft(model, tools, t[0]), ts))
        P = sum(len(p) for p in preds); H = sum(len(p & t[1]) for p, t in zip(preds, ts)); A = sum(len(t[1]) for t in ts)
        print(f"{model} {name}: n={len(ts)} precision={H / max(P, 1):.0%} coverage={H / max(A, 1):.0%} (predicted {P}, hit {H}, actual {A})", flush=True)
        json.dump([{"query": t[0], "actual": sorted(t[1]), "pred": sorted(p)} for p, t in zip(preds, ts)], open(f"drafter-{model}-{name}.json", "w"), indent=1)


main()
