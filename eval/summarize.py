import json, sqlite3, statistics as st, sys
prefix, a, b, db = sys.argv[1:5]
L = lambda p: [json.loads(l) for l in open(f"{prefix}-{p}.jsonl")]
A, B = L(a), L(b)
assert [r["text"] for r in A] == [r["text"] for r in B]
def summ(rows, label):
    s = [r["seconds"] for r in rows]
    print(f"  {label:10s} median={st.median(s):5.1f}s mean={st.mean(s):5.1f}s calls/turn={st.mean(r['calls'] for r in rows):.2f} errors={sum(r['status'] != 'ok' for r in rows)}")
for split in sorted({r["split"] for r in A}) + [None]:
    f = lambda rows: [r for r in rows if split is None or r["split"] == split]
    print(f"--- {split or 'ALL'} (n={len(f(A))})"); summ(f(A), a); summ(f(B), b)
d = [y["seconds"] - x["seconds"] for x, y in zip(A, B)]
print(f"paired {b}-{a}: median {st.median(d):.1f}s mean {st.mean(d):.1f}s; {b} faster on {sum(x < 0 for x in d)}/{len(d)}")
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
P = H = 0
for (body,) in c.execute("select body from records where kind='tool-trace'"):
    x = json.loads(body)
    if x.get("mode") != b: continue
    keys = {k["key"] for k in x["calls"]}; P += len(x["predicted"]); H += sum(p in keys for p in x["predicted"])
print(f"{b} traces: predicted={P}, re-called by model={H}")
