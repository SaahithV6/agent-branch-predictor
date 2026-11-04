import json, math, statistics as st, sys
R = [json.loads(l) for l in open(sys.argv[1])]
arms = list(dict.fromkeys(r["arm"] for r in R))
by = {}
for r in R: by.setdefault(r["i"], {})[r["arm"]] = r
ok = lambda r: r and r["status"] == "RUN_FINISHED" and r["ttft"]
for a in arms:
    rows = [r for r in R if r["arm"] == a]; g = [r for r in rows if ok(r)]
    print(f"{a:10s} n={len(rows)} ok={len(g)} ttft median={st.median(r['ttft'] for r in g):5.1f}s mean={st.mean(r['ttft'] for r in g):5.1f}s | total median={st.median(r['total'] for r in g):5.1f}s | chars={st.median(r['chars'] for r in g):.0f}")
base, test = arms[0], arms[1]
for k in ("ttft", "total"):
    d = [v[test][k] - v[base][k] for v in by.values() if ok(v.get(base)) and ok(v.get(test))]
    se = st.stdev(d) / math.sqrt(len(d))
    print(f"paired {k} {test}-{base} n={len(d)}: mean {st.mean(d):+.2f}s (95% CI {st.mean(d)-1.96*se:+.2f}..{st.mean(d)+1.96*se:+.2f}), median {st.median(d):+.2f}s, {test} faster {sum(x<0 for x in d)}/{len(d)}")
