import json, os, pickle, re, sqlite3, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MultiLabelBinarizer

REAL_ENT = {"nvidia", "together ai", "lunar outpost", "bedrock robotics", "zipline", "verkada", "honeywell", "waymo", "neuralink", "hermeus", "lyft", "cohere",
            "the exploration company", "revel", "modal", "workiva", "qm", "native", "cortex", "scrubline", "jepa-propulsion", "jepa-cad", "ycoyi9_27"}
SIM_ENT = {"auth", "billing", "search", "checkout", "notify", "ingest", "gateway", "reports", "acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka"}
ENTS = sorted(REAL_ENT | SIM_ENT, key=len, reverse=True)
PARAM = {t["name"]: next(iter(t["params"])) for f in ("tools-real.json", "tools-sim.json", "tools-biz.json") for t in json.load(open(f))}


def entity(q):
    low = f" {re.sub(r'[^a-z0-9_-]+', ' ', q.lower())} "
    return next((e for e in ENTS if f" {e} " in low), None)


def mask(q, e):
    return re.sub(re.escape(e), " ENTITY ", q, flags=re.I) if e else q


def key(name, args):
    return name + json.dumps({k: v.lower() if isinstance(v, str) else v for k, v in args.items()}, sort_keys=True)


def traces():
    out = []
    global RAW
    RAW = []
    for db, prefix in (("real2.db", "real_"), ("bench2.db", "sim_")):
        c = sqlite3.connect(f"file:{os.path.expanduser('~/Documents/qm-native/' + db)}?mode=ro", uri=True)
        for (b,) in c.execute("select body from records where kind='tool-trace' order by rowid"):
            d = json.loads(b)
            calls = [x for x in d["calls"] if x["name"] in PARAM]
            if calls:
                RAW.append(calls)
                out.append((d["query"], {key(x["name"], x["args"]) for x in calls}, {x["name"] for x in calls}))
    return out


def train():
    import glob
    rows = [r for f in sorted(glob.glob("distill-data*.json")) for r in json.load(open(f))]
    X = [r["text"].replace("{E}", " ENTITY ") for r in rows]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([r["tools"] for r in rows])
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
    model = make_pipeline(feats, OneVsRestClassifier(LogisticRegression(C=8, max_iter=2000, class_weight="balanced")))
    model.fit(X, Y)
    return model, mlb


def arg_templates(exclude=None):
    from collections import Counter
    seen = {}
    for i, calls in enumerate(RAW):
        if i == exclude:
            continue
        for c in calls:
            e = c["args"].get(PARAM[c["name"]], "")
            slotted = {k: ("$E" if isinstance(v, str) and v.lower() == str(e).lower() else v) for k, v in c["args"].items()}
            seen.setdefault(c["name"], Counter())[json.dumps(slotted, sort_keys=True)] += 1
    return {t: json.loads(c.most_common(1)[0][0]) for t, c in seen.items()}


def fill(tool, e, templates):
    t = templates.get(tool, {PARAM[tool]: "$E"})
    return {k: (e if v == "$E" else v) for k, v in t.items()}


def predict(model, mlb, q, threshold, templates=None):
    e = entity(q)
    if not e:
        return set()
    probs = model.predict_proba([mask(q, e)])[0]
    group = "real_" if e in REAL_ENT else "sim_"
    kind_ok = lambda t: t.startswith(group) and ((e in SIM_ENT and (("customer" in PARAM[t]) == (e in {"acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka"})))
                                                 or (e in REAL_ENT and (("repo" == PARAM[t]) == (e in {"qm", "native", "cortex", "scrubline", "jepa-propulsion", "jepa-cad", "ycoyi9_27"}))))
    return {key(t, fill(t, e, templates or {})) for t, p in zip(mlb.classes_, probs) if p >= threshold and kind_ok(t)}


if __name__ == "__main__":
    model, mlb = train()
    pickle.dump((model, mlb), open("call-classifier.pkl", "wb"))
    ts = traces()
    for threshold in (0.3, 0.5, 0.7, 0.9):
        for label, sel in (("real", lambda t: next(iter(t[2])).startswith("real_")), ("sim", lambda t: next(iter(t[2])).startswith("sim_"))):
            idx = [i for i, t in enumerate(ts) if sel(t)]
            sub = [ts[i] for i in idx]
            preds = [predict(model, mlb, ts[i][0], threshold, arg_templates(exclude=i)) for i in idx]
            P = sum(map(len, preds)); H = sum(len(p & t[1]) for p, t in zip(preds, sub)); A = sum(len(t[1]) for t in sub)
            nameP = sum(len({k.split("{")[0] for k in p} & t[2]) for p, t in zip(preds, sub)); nameA = sum(len(t[2]) for t in sub)
            print(f"threshold={threshold} {label:4s} n={len(sub):3d} exact precision={H / max(P, 1):4.0%} coverage={H / max(A, 1):4.0%} | tool-name coverage={nameP / max(nameA, 1):4.0%}")
