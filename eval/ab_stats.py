import json, math, statistics as st, sys
R = [json.loads(l) for l in open(sys.argv[1])]
arms = list(dict.fromkeys(r["arm"] for r in R))
ok = lambda r: r and r["status"] == "RUN_FINISHED" and r["total"]
by = {}
for r in R: by.setdefault(r["i"], {})[r["arm"]] = r
for a in arms:
    rows = [r for r in R if r["arm"] == a]; g = [r for r in rows if ok(r)]
    print(f"{a:26s} ok={len(g)}/{len(rows)} ttft median={st.median(r['ttft'] for r in g if r['ttft']):5.1f}s | total median={st.median(r['total'] for r in g):5.1f}s mean={st.mean(r['total'] for r in g):5.1f}s | correct={sum(r.get('correct', False) for r in rows)}/{len(rows)} | chars={st.median(r['chars'] for r in g):.0f}")
base = arms[0]
for b in arms[1:]:
    for k in ("ttft", "total"):
        d = [v[b][k] - v[base][k] for v in by.values() if ok(v.get(base)) and ok(v.get(b)) and v[b][k] and v[base][k]]
        se = st.stdev(d) / math.sqrt(len(d))
        print(f"  {b} - {base} {k:5s} n={len(d)}: mean {st.mean(d):+.2f}s (95% CI {st.mean(d)-1.96*se:+.2f}..{st.mean(d)+1.96*se:+.2f}) median {st.median(d):+.2f}s faster {sum(x<0 for x in d)}/{len(d)}; ratio of medians {st.median(v[b][k] for v in by.values() if ok(v.get(b)) and v[b][k]) / st.median(v[base][k] for v in by.values() if ok(v.get(base)) and v[base][k]):.2f}")
