import json, os, statistics as st, sys, time, urllib.request

BRIDGE, QM = "http://127.0.0.1:8787", "http://127.0.0.1:8080"
MODEL = os.environ.get("BENCH_MODEL")
ASKS = [f"Prep me for the QBR with {c}." for c in ["Nike", "Boeing", "Oracle", "Visa"]] + [f"Any overdue or disputed invoices for {c}?" for c in ["Intel", "eBay", "Zoom", "Target"]] + \
       [f"What's Hacker News saying about {t}? Read the two most relevant linked articles. Answer in exactly 5 bullet points." for t in ["zig", "nixos", "htmx", "duckdb", "ebpf", "wayland", "passkeys", "heat pumps"]]
rows = []
for i, text in enumerate(ASKS):
    t0 = time.time()
    body = {"surface": "api", "actor": {"externalId": "saahith"}, "conversation": {"kind": "dm", "threadRef": f"prof-{i}-{int(t0)}"}, "text": text, **({"model": MODEL} if MODEL else {})}
    r = json.load(urllib.request.urlopen(urllib.request.Request(QM + "/v1/turns", data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=400))
    t1 = time.time()
    tl = [x for x in json.load(urllib.request.urlopen(BRIDGE + "/timelines")) if x["query"] == text][-1]
    pre = tl["at"] / 1000 - t0
    calls = sorted(tl["timeline"], key=lambda c: c["start"])
    batches = []
    for c in calls:
        if batches and c["start"] - batches[-1]["start"] < 400:
            batches[-1]["end"] = max(batches[-1]["end"], c["end"]); batches[-1]["n"] += 1
        else:
            batches.append({"start": c["start"], "end": c["end"], "n": 1})
    tool = sum(b["end"] - b["start"] for b in batches) / 1000
    first = (batches[0]["start"] / 1000) if batches else 0
    between = sum(b["start"] - a["end"] for a, b in zip(batches, batches[1:])) / 1000
    final = (t1 - t0) - pre - (batches[-1]["end"] / 1000 if batches else 0)
    row = {"text": text[:50], "total": t1 - t0, "pre_recall": pre, "first_step": first, "tool": tool, "between_steps": between, "final_answer": final, "steps": len(batches), "calls": len(calls), "reply_len": len(r.get("reply") or "")}
    rows.append(row)
    print(" ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
json.dump(rows, open(f"profile-{os.environ.get('PROFILE_TAG', 'run')}.json", "w"), indent=1)
for group, sel in (("business", rows[:8]), ("research", rows[8:])):
    print(f"--- {group} (medians)", {k: round(st.median(r[k] for r in sel), 1) for k in ("total", "pre_recall", "first_step", "tool", "between_steps", "final_answer", "steps", "reply_len")})
