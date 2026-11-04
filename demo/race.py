import json, sys, urllib.request
q = sys.argv[1]
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/api/ask", data=json.dumps({"q": q}).encode(), headers={"content-type": "application/json"})))
done, first, tools = set(), {}, {}
for raw in urllib.request.urlopen(f"http://127.0.0.1:8090/api/events?id={r['id']}", timeout=300):
    line = raw.decode().strip()
    if not line.startswith("data:"): continue
    ev = json.loads(line[5:]); a = ev["arm"]
    if ev["type"] == "delta": first.setdefault(a, ev["t"])
    elif ev["type"] == "tool": tools[a] = tools.get(a, 0) + 1
    elif ev["type"] == "done": print(f"{a:5s} done {ev['t']/1000:6.2f}s first {first.get(a,0)/1000:5.2f}s tools {tools.get(a,0)}")
    elif ev["type"] == "trace":
        u = ev["trace"]["usage"] or {}
        print(f"{a:5s} fast={u.get('fastPath')} memo={u.get('memo')} tokens_in={u.get('input',0)+u.get('cacheRead',0)+u.get('cacheWrite',0)}")
        done.add(a)
        if len(done) == 2: break
