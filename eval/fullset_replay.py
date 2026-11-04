import json, os, pickle, sqlite3, sys
from collections import Counter
sys.argv = ["x"]
import train as T

db = sqlite3.connect(f"file:{os.path.expanduser('~/Documents/qm-native/biz-native2.db')}?mode=ro", uri=True)
traces = []
for (b,) in db.execute("select body from records where kind='tool-trace' order by rowid"):
    d = json.loads(b)
    if d.get("mode") != "shadow":
        continue
    calls = [c for c in d["calls"] if c["name"].startswith("biz_") and c["name"] not in ("biz_create_task", "biz_draft_email")]
    if calls:
        traces.append((d["query"], calls))
ACCTS = [r[0] for r in sqlite3.connect(os.path.expanduser("~/Documents/qm-native/biz.db")).execute("select name from accounts")]
T.ENTS = sorted({a.lower() for a in ACCTS} | set(T.ENTS), key=len, reverse=True)
T.SIM_ENT = T.SIM_ENT
model, mlb = pickle.load(open("call-classifier.pkl", "rb"))
STOP = set("the and for with that this from have what your about into when were will would there their them they then than which while".split())
words = lambda q: {w for w in __import__("re").findall(r"[a-z0-9][a-z0-9._-]{2,}", q.lower()) if w not in STOP}
key = lambda n, a: n + json.dumps({k: v.lower() if isinstance(v, str) else v for k, v in a.items()}, sort_keys=True)


def ent(q):
    low = q.lower()
    hits = [a for a in ACCTS if a.lower() in low]
    return max(hits, key=len) if hits else None


def hist_pred(hist, q, e, theta, tau=0.5):
    w = words(q) - set(e.lower().split())
    nb = [h for h in hist if len(w & h["w"]) / max(len(w | h["w"]), 1) >= tau]
    if len(nb) < 2:
        return set()
    cnt = Counter(t for h in nb for t in h["t"])
    return {t.replace('"$e"', json.dumps(e.lower())) for t, n in cnt.items() if n >= 2 and n / len(nb) >= theta}


def clf_pred(q, e, cth):
    import re
    probs = model.predict_proba([re.sub(re.escape(e), " ENTITY ", q, flags=re.I)])[0]
    return {key(t, {"account": e}) for t, p in zip(mlb.classes_, probs) if t.startswith("biz_") and p >= cth}


rows = []
for theta, cth, mode in [(0.7, 0.5, "fallback"), (0.5, 0.5, "fallback"), (0.3, 0.5, "fallback"), (0.7, 0.5, "union"), (0.5, 0.3, "union"), (0.3, 0.3, "union"), (0.3, 0.2, "union"), (0.2, 0.15, "union")]:
    hist, P = [], [0, 0, 0, 0, 0]
    for i, (q, calls) in enumerate(traces):
        e = ent(q)
        actual = {key(c["name"], c["args"]) for c in calls}
        if i >= 8 and e:
            hp = hist_pred(hist, q, e, theta)
            cp = clf_pred(q, e, cth)
            pred = (hp | cp) if mode == "union" else (hp or cp)
            P[0] += len(pred); P[1] += len(pred & actual); P[2] += len(actual); P[3] += actual <= pred; P[4] += 1
        hist.append({"w": words(q) - set((e or "").lower().split()), "t": {key(c["name"], {k: ("$E" if isinstance(v, str) and e and v.lower() == e.lower() else v) for k, v in c["args"].items()}) for c in calls}})
    print(f"{mode:8s} theta={theta} clf={cth}: precision={P[1]/max(P[0],1):4.0%} coverage={P[1]/max(P[2],1):4.0%} FULL-SET turns={P[3]}/{P[4]} ({P[3]/max(P[4],1):.0%}) predicted/turn={P[0]/max(P[4],1):.1f}")


import re
import numpy as np
feats = model.steps[0][1] if hasattr(model, "steps") else model.estimators_[0]
vec = lambda q, e: feats.transform([re.sub(re.escape(e), " ENTITY ", q, flags=re.I)])
def cos(a, b):
    return float((a @ b.T).toarray()[0, 0] / (np.sqrt((a.multiply(a)).sum() * (b.multiply(b)).sum()) or 1))

def task_pred(hist, q, e, theta, tau):
    v = vec(q, e)
    nb = [h for h in hist if cos(v, h["v"]) >= tau]
    if len(nb) < 2:
        return set()
    cnt = Counter(t for h in nb for t in h["t"])
    return {t.replace('"$e"', json.dumps(e.lower())) for t, n in cnt.items() if n >= 2 and n / len(nb) >= theta}

print("--- task-level neighbors (cosine on classifier features)")
for tau in (0.2, 0.3, 0.4, 0.5):
    for theta, cth in ((0.5, 0.5), (0.3, 0.3), (0.2, 0.3)):
        hist, P = [], [0, 0, 0, 0, 0]
        for i, (q, calls) in enumerate(traces):
            e = ent(q)
            actual = {key(c["name"], c["args"]) for c in calls}
            if i >= 8 and e:
                pred = task_pred(hist, q, e, theta, tau) | clf_pred(q, e, cth)
                P[0] += len(pred); P[1] += len(pred & actual); P[2] += len(actual); P[3] += actual <= pred; P[4] += 1
            if e:
                hist.append({"v": vec(q, e), "t": {key(c["name"], {k: ("$E" if isinstance(v, str) and v.lower() == e.lower() else v) for k, v in c["args"].items()}) for c in calls}})
        print(f"tau={tau} theta={theta} clf={cth}: precision={P[1]/max(P[0],1):4.0%} coverage={P[1]/max(P[2],1):4.0%} FULL-SET={P[3]}/{P[4]} ({P[3]/max(P[4],1):.0%}) predicted/turn={P[0]/max(P[4],1):.1f}")
