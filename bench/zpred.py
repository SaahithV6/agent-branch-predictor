"""Zero-shot next-call prediction over a million tools: no per-tool history needed.

A tool's identity is never a feature. Every candidate is described by how it relates to the
conversation (lexical match of its definition, its server versus recent calls, whether its required
arguments can be filled from what is on screen), so a tool that was never called in training is
scored exactly like one that was. Test examples come from servers held out of training entirely
(bench/calls.py splits by server).

  stage 1  retrieve: BM25 over every catalog definition -> top K (or: the session's own tool set)
  stage 2  rank:     LightGBM on candidate features, trained on the train split
  stage 3  fill:     each argument from a prior result field (by key) or the request (by value class)
  stage 4  gate:     LightGBM on dev predicts "whole call exactly right"; thresholds picked on dev

    python3 -I bench/zpred.py run [--train N] [--k 50] [--workers 8]
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import re
import sys
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolindex import tokens as name_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CATALOGS = [ROOT / "data/catalogs/tools_1m.jsonl", ROOT / "data/calls/tools_toolbench_traj.jsonl"]
EXAMPLES = ROOT / "data/calls/toolbench.jsonl"
OUT = ROOT / "bench/results/zpred"
EMB = ROOT / "data/calls/emb"  # optional: tools.npy (aligned with catalog order) + ctx_<split>.npy (aligned with examples)

STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "this", "that", "with", "by", "as",
        "it", "at", "from", "you", "your", "i", "me", "my", "we", "our", "can", "could", "would", "please", "also", "want",
        "like", "some", "any", "get", "about", "what", "which", "how", "need", "know", "give", "provide", "information",
        "api", "endpoint", "use", "will", "if", "so", "do", "all", "s", "t"}
WORD = re.compile(r"[a-z0-9]+")
DF_CAP = 0.05  # query words in more than this share of tools carry ~no evidence: skip their postings
BM25_K1, BM25_B = 1.2, 0.75


def words(s):
    return [w.rstrip("s") if len(w) > 3 else w for w in WORD.findall(str(s).lower()) if w not in STOP]


def tool_words(t):
    ws = list(name_tokens(t["name"])) * 2 + list(name_tokens(t["server"])) * 2 + words(t.get("desc", ""))[:80]
    for a in t.get("args", []):
        ws += list(name_tokens(a["name"]))
    return ws


# ---------------------------------------------------------------------------------- catalog + BM25
class Catalog:
    def __init__(self, paths=CATALOGS):
        t0 = time.time()
        self.tools, self.pos = [], {}
        vocab, rows, cols, lens = {}, [], [], []
        for p in paths:
            for line in open(p):
                t = json.loads(line)
                if t["id"] in self.pos:
                    continue
                d = len(self.tools)
                self.pos[t["id"]] = d
                self.tools.append({"id": t["id"], "server": t["server"], "name": t["name"],
                                   "args": t.get("args", []), "nt": name_tokens(t["name"]), "st": name_tokens(t["server"])})
                ws = tool_words(t)
                lens.append(len(ws))
                for w, c in collections.Counter(ws).items():
                    rows.append(vocab.setdefault(w, len(vocab)))
                    cols.append(d)
                    # tf stored inline: (word, doc, count) triples
                    cols.append(c)
        n = len(self.tools)
        rows = np.array(rows, np.int32)
        dc = np.array(cols, np.int32).reshape(-1, 2)
        order = np.argsort(rows, kind="stable")
        self.post_doc, self.post_tf = dc[order, 0], dc[order, 1].astype(np.float32)
        self.ptr = np.searchsorted(rows[order], np.arange(len(vocab) + 1))
        self.vocab = vocab
        self.dl = np.array(lens, np.float32)
        df = np.diff(self.ptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.cap = DF_CAP * n
        self.norm = (BM25_K1 * (1 - BM25_B + BM25_B * self.dl / self.dl.mean())).astype(np.float32)
        self.n = n
        self.emb = np.load(EMB / "tools.npy", mmap_mode="r") if (EMB / "tools.npy").exists() else None
        print(f"catalog: {n:,} tools, {len(vocab):,} words, {len(self.post_doc):,} postings, {time.time() - t0:.0f}s",
              flush=True)

    def bm25(self, qwords, docs=None):
        """Scores for every tool (docs=None) or for the given doc indices."""
        ids = [self.vocab[w] for w in set(qwords) if w in self.vocab]
        if docs is not None:
            docs = np.asarray(docs)
            out = np.zeros(len(docs), np.float32)
            where = {d: i for i, d in enumerate(docs)}
            for v in ids:
                s, e = self.ptr[v], self.ptr[v + 1]
                pd_ = self.post_doc[s:e]
                hit = np.isin(pd_, docs)
                for d, tf in zip(pd_[hit], self.post_tf[s:e][hit]):
                    out[where[d]] += self.idf[v] * tf * (BM25_K1 + 1) / (tf + self.norm[d])
            return out
        sc = np.zeros(self.n, np.float32)
        for v in ids:
            s, e = self.ptr[v], self.ptr[v + 1]
            if e - s > self.cap:
                continue
            d, tf = self.post_doc[s:e], self.post_tf[s:e]
            sc[d] += self.idf[v] * tf * (BM25_K1 + 1) / (tf + self.norm[d])
        return sc


# ---------------------------------------------------------------------------------- context values
KV = re.compile(r"""['"]([A-Za-z_][\w\-]{0,40})['"]\s*:\s*(?:['"]([^'"]{1,120})['"]|(-?\d+(?:\.\d+)?)|(True|False|true|false))""")
QUOTED = re.compile(r"""["'“]([^"'”]{2,80})["'”]""")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"https?://\S+")
CAPS = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*")


CODE = re.compile(r"(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{5,}(?![\w-])")  # mixed letters+digits: ids, VINs, hashes
PHRASE = re.compile(r"\b(?:for|about|called|named|titled|of|on)\s+([a-z][\w']*(?:\s+[a-z][\w']*){0,3}?)(?=[,.?!;]|\s(?:and|in|with|from|to|by|at)\b|$)")
OPENERS = STOP | {"please", "additionally", "furthermore", "moreover", "also", "then", "next", "finally", "help", "hey", "hi",
                  "thanks", "could", "can", "i'm", "im", "let", "tell", "show", "find", "fetch", "check"}
KINDS = ["email", "url", "quoted", "code", "num", "caps", "phrase", "span", "result", "args", "example"]
EXAMPLES_DOC = ROOT / "data/calls/tool_examples.json"
# ponytail: documented example values exist only for ToolBench tools here (the other catalogs were parsed without
# them), so in the 1M setting this favours ToolBench definitions; parse OpenAPI examples/defaults to even it out.
TOOL_EXAMPLES = json.load(open(EXAMPLES_DOC)) if EXAMPLES_DOC.exists() else {}
ARGCATS = ["id", "url", "email", "name", "query", "q", "term", "keyword", "search", "city", "country", "date", "lang",
           "language", "limit", "page", "symbol", "code", "type", "category", "user", "username", "title", "text"]


def context_values(ex):
    """[(value, key tokens, kind, words before it, recency)] from the request and the prior results."""
    out = []
    q = ex["query"]
    for rx, kind in ((EMAIL, "email"), (URL, "url"), (QUOTED, "quoted"), (CODE, "code"), (NUM, "num"), (CAPS, "caps"),
                     (PHRASE, "phrase")):
        for m in rx.finditer(q):
            v = (m.group(1) if rx in (QUOTED, PHRASE) else m.group(0)).strip().rstrip(".,;:!?")
            if kind == "caps":
                ws = v.split()
                while ws and ws[0].lower() in OPENERS:
                    ws = ws[1:]
                v = " ".join(ws)
            if len(v) < 1:
                continue
            before = frozenset(words(q[max(0, m.start() - 40):m.start()])[-4:])
            out.append((v, frozenset(), kind, before, 0))
    seen = {v.lower() for v, *_ in out}
    toks = [(m.group(0), m.start()) for m in re.finditer(r"[a-z][a-z0-9']*", q)]
    for n_ in (1, 2, 3):  # lowercase spans the patterns above miss ("taylor swift"): the slot model sorts them out
        for i in range(len(toks) - n_ + 1):
            span = toks[i:i + n_]
            if span[0][0] in STOP or span[-1][0] in STOP:
                continue
            v = q[span[0][1]:span[-1][1] + len(span[-1][0])]
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append((v, frozenset(), "span", frozenset(words(q[max(0, span[0][1] - 40):span[0][1]])[-4:]), 0))
    n = len(ex["prior"])
    for j, p in enumerate(ex["prior"][-4:]):
        rec = min(n, 4) - j
        for m in KV.finditer(p["result"]):
            v = m.group(2) or m.group(3) or m.group(4)
            out.append((v, frozenset(name_tokens(m.group(1)) | {m.group(1).lower()}), "result", frozenset(), rec))
        for k, v in p["args"].items():  # what was passed before is often passed again
            if not isinstance(v, (dict, list)):
                out.append((str(v), frozenset(name_tokens(k) | {k.lower()}), "args", frozenset(), rec))
    return out


def vclass(v):
    s = str(v)
    if re.fullmatch(r"-?\d+", s):
        return "int"
    if re.fullmatch(r"-?\d+\.\d+", s):
        return "float"
    if EMAIL.fullmatch(s):
        return "email"
    if s.startswith("http"):
        return "url"
    if re.fullmatch(r"(true|false)", s, re.I):
        return "bool"
    return "str"


SLOT_FEATS = ["key_hit", "key_exact", "before_hit", *[f"k_{k}" for k in KINDS], "t_num", "t_bool", "required",
              "v_int", "v_float", "v_url", "v_email", "v_bool", "v_len", "v_words", "v_digits", "v_lower", "recency",
              "repeats", "n_args", *[f"a_{c}" for c in ARGCATS]]
SLOT = None  # LightGBM slot model, trained in run() before workers fork


def slot_rows(arg, values):
    at = name_tokens(arg["name"]) | {arg["name"].lower()}
    typ = arg.get("type", "string")
    cnt = collections.Counter(str(v).lower() for v, *_ in values)
    rows = []
    for v, src, kind, before, rec in values:
        s, c = str(v), vclass(v)
        rows.append([bool(src & at), arg["name"].lower() in src, bool(before & at), *[kind == k for k in KINDS],
                     typ in ("integer", "number"), typ == "boolean", bool(arg.get("required")),
                     c == "int", c == "float", c == "url", c == "email", c == "bool", min(len(s), 200), len(s.split()),
                     sum(ch.isdigit() for ch in s) / max(len(s), 1), s.islower(), rec, cnt[s.lower()], 0,
                     *[c_ in at for c_ in ARGCATS]])
    return rows


def with_examples(tool, values):
    ex = TOOL_EXAMPLES.get(tool["id"], {})
    return values + [(v, frozenset(name_tokens(k) | {k.lower()}), "example", frozenset(), 0) for k, v in ex.items()]


def fill(tool, values, slot=None):
    """Arguments for this tool from context values -> (args, n required, n required filled, min filled score)."""
    slot = slot or SLOT
    args, req, got, low = {}, 0, 0, 1.0
    values = with_examples(tool, values)
    for a in tool["args"]:
        req += bool(a.get("required"))
        if not values:
            continue
        R = np.array(slot_rows(a, values), np.float32)
        R[:, SLOT_FEATS.index("n_args")] = len(tool["args"])
        sc = slot.predict(R)
        i = int(np.argmax(sc))
        if sc[i] >= (0.05 if a.get("required") else 0.8):  # optional args only on strong evidence (grid on dev)
            args[a["name"]] = values[i][0]
            got += bool(a.get("required"))
            low = min(low, float(sc[i]))
    return args, req, got, low


def train_slots(cat, examples, workers):
    X, y = [], []
    for ex in examples:
        t = cat.tools[cat.pos[ex["tool"]]]
        vals = with_examples(t, context_values(ex))
        if not vals:
            continue
        tgt = {k: str(v).strip().lower() for k, v in ex["args"].items()}
        for a in t["args"]:
            if a["name"] not in tgt:
                continue
            R = slot_rows(a, vals)
            for r in R:
                r[SLOT_FEATS.index("n_args")] = len(t["args"])
            X += R
            y += [str(v).strip().lower() == tgt[a["name"]] for v, *_ in vals]
    X, y = np.array(X, np.float32), np.array(y, np.int8)
    m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1,
                   "num_threads": workers}, lgb.Dataset(X, y, feature_name=SLOT_FEATS), 300)
    print(f"slot model: {len(y):,} (arg, value) pairs, {y.mean():.3f} positive", flush=True)
    return m


def same_args(a, b):
    norm = lambda d: {str(k): str(v).strip().lower() for k, v in d.items()}  # noqa: E731
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------------- features
FEATS = ["bm25", "rank", "rel_top", "gap_next", "same_server_last", "server_seen", "called_before", "is_last_tool",
         "q_name_overlap", "q_server_overlap", "n_args", "n_req", "req_fillable", "n_key_hits", "n_prior", "emb_cos",
         "emb_rank", "session"]


def candidates(cat, ex, k, session_only, ctx_vec=None):
    """Candidate doc indices and their BM25 scores."""
    q = words(ex["query"])
    if ex["prior"]:
        last = ex["prior"][-1]
        q += words(" ".join(re.split(r"[/_]", last["tool"]))) + words(last["result"][:400])
    if session_only:
        docs = np.array([cat.pos[t] for t in ex["session"] if t in cat.pos], np.int64)
        return docs, cat.bm25(q, docs), None
    sc = cat.bm25(q)
    esc = None
    if cat.emb is not None and ctx_vec is not None:
        esc = np.asarray(cat.emb @ ctx_vec, np.float32)
        # union of the two retrievers' top K
        top = np.unique(np.concatenate([np.argpartition(-sc, k)[:k], np.argpartition(-esc, k)[:k]]))
    else:
        top = np.argpartition(-sc, k)[:k]
    return top, sc[top], (esc[top] if esc is not None else None)


def fillable(tool, values):
    """Cheap stand-in for fill() while ranking 50 candidates: share of required args with a plausible value,
    and how many args have a value under the same key name. The learned slot model runs on the winner only."""
    keys = set().union(*(src for _, src, *_ in values)) if values else set()
    q_num = any(k in ("num",) for _, _, k, *_ in values)
    q_str = any(k in ("quoted", "caps", "code", "email", "url") for _, _, k, *_ in values)
    req = hits = ok = 0
    for a in tool["args"]:
        at = name_tokens(a["name"]) | {a["name"].lower()}
        hit = bool(at & keys)
        hits += hit
        if a.get("required"):
            req += 1
            ok += hit or (q_num if a.get("type") in ("integer", "number") else q_str)
    return req, ok / req if req else 1.0, hits


def featurize(cat, ex, docs, bm, emb, session_only):
    values = context_values(ex)
    qw = set(words(ex["query"]))
    last = ex["prior"][-1]["tool"] if ex["prior"] else ""
    last_server = last.split("/")[1] if last.count("/") >= 2 else ""
    servers = {p["tool"].split("/")[1] for p in ex["prior"] if p["tool"].count("/") >= 2}
    called = collections.Counter(p["tool"] for p in ex["prior"])
    order = np.argsort(-bm)
    rank = np.empty(len(docs), np.int32)
    rank[order] = np.arange(len(docs))
    top = bm[order[0]] if len(bm) else 1.0
    nxt = np.sort(bm)[::-1]
    erank = np.empty(len(docs), np.int32)
    if emb is not None:
        erank[np.argsort(-emb)] = np.arange(len(docs))
    rows = []
    for i, d in enumerate(docs):
        t = cat.tools[d]
        req, frac, hits = fillable(t, values)
        rows.append([bm[i], rank[i], bm[i] / (top + 1e-6), bm[i] - (nxt[1] if len(nxt) > 1 else 0),
                     t["server"] == last_server, t["server"] in servers, called[t["id"]], t["id"] == last,
                     len(t["nt"] & qw) / (len(t["nt"]) or 1), len(t["st"] & qw) / (len(t["st"]) or 1),
                     len(t["args"]), req, frac, hits, len(ex["prior"]),
                     emb[i] if emb is not None else np.nan, erank[i] if emb is not None else -1, session_only])
    return np.array(rows, np.float32)


# ---------------------------------------------------------------------------------- per-example work (forked workers)
CAT = None


def work(job):
    ex, k, session_only, ctx = job
    docs, bm, emb = candidates(CAT, ex, k, session_only, ctx)
    if not len(docs):
        return None
    X = featurize(CAT, ex, docs, bm, emb, session_only)
    nxt = CAT.pos[ex["tool"]]
    later = {CAT.pos[t] for t, _ in ex["future"]}
    # relevance: 2 = the very next call, 1 = called later in the task, 0 = never
    y = np.array([2 if d == nxt else 1 if d in later else 0 for d in docs], np.int8)
    return X, y, docs, int((y > 0).any())


def build(examples, k, session_only, workers, ctx=None):
    jobs = [(ex, k, session_only, None if ctx is None else ctx[i]) for i, ex in enumerate(examples)]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(work, jobs, chunksize=64)


def stack(res):
    keep = [r for r in res if r is not None]
    X = np.concatenate([r[0] for r in keep])
    y = np.concatenate([r[1] for r in keep])
    g = np.array([len(r[1]) for r in keep])
    return X, y, g, keep


# ---------------------------------------------------------------------------------- train / evaluate
def top1(model, res, examples):
    """Per example: (gate features, tool right, call right, target retrieved). The call is the top-ranked tool
    with its arguments filled by the slot model; it is right if the agent makes exactly that call later on."""
    out = []
    for r, ex in zip(res, examples):
        if r is None:
            out.append(None)
            continue
        X, y, docs, hit = r
        s = model.predict(X)
        i = int(np.argmax(s))
        srt = np.sort(s)[::-1]
        t = CAT.tools[docs[i]]
        args, req, got, low = fill(t, context_values(ex))
        right = y[i] > 0 and any(t["id"] == tid and same_args(args, a) for tid, a in ex["future"])
        g = [srt[0], srt[0] - (srt[1] if len(srt) > 1 else 0), low, got / req if req else 1.0, len(args), *X[i]]
        out.append((g, bool(y[i] > 0), bool(right), hit))
    return out


def gate_X(t):
    return np.array([x[0] for x in t], np.float32)


def report(name, t, gate, th):
    n = len(t)
    tt = [x for x in t if x is not None]
    p = gate.predict(gate_X(tt))
    rows = {"examples": n, "retrieved": round(sum(x[3] for x in tt) / n, 4),
            "tool_top1": round(sum(x[1] for x in tt) / n, 4), "call_top1": round(sum(x[2] for x in tt) / n, 4)}
    for target, thr in th.items():
        fire = p >= thr
        right = np.array([x[2] for x in tt])
        rows[f"@{target}"] = {"threshold": round(float(thr), 4), "coverage": round(float(fire.sum() / n), 4),
                              "precision": round(float(right[fire].mean()) if fire.any() else 0.0, 4)}
    print(name, json.dumps(rows), flush=True)
    return rows


def thresholds(p, right, targets=(0.5, 0.6, 0.7, 0.8, 0.9)):
    order = np.argsort(-p)
    cum = np.cumsum(right[order]) / np.arange(1, len(p) + 1)
    out = {}
    for tg in targets:
        ok = np.nonzero(cum >= tg)[0]
        # the largest prefix that still meets the target; skip tiny prefixes that are pure noise
        ok = ok[ok >= 50]
        out[tg] = float(p[order[ok[-1]]]) if len(ok) else 1.01
    return out


def load(split, limit=None, seed=0):
    xs = [json.loads(line) for line in open(EXAMPLES) if f'"split": "{split}"' in line]
    # every call still to come in the same trajectory: a prefetch pays off if the agent makes it at any later step
    calls = collections.defaultdict(list)
    for x in xs:
        calls[x["conv"]].append((len(x["prior"]), x["tool"], x["args"]))
    for x in xs:
        x["future"] = [(t, a) for n, t, a in sorted(calls[x["conv"]], key=lambda c: c[0]) if n >= len(x["prior"])]
    if limit and len(xs) > limit:
        rng = np.random.default_rng(seed)
        xs = [xs[i] for i in sorted(rng.choice(len(xs), limit, replace=False))]
    return xs


def ctx_for(split, examples):
    p = EMB / f"ctx_{split}.npy"
    if not p.exists():
        return None
    m = json.load(open(EMB / f"ctx_{split}.json"))  # example key -> row
    arr = np.load(p, mmap_mode="r")
    return [np.asarray(arr[m[f"{ex['conv']}:{len(ex['prior'])}"]]) for ex in examples]


def run(a):
    global CAT
    OUT.mkdir(parents=True, exist_ok=True)
    CAT = Catalog()
    global SLOT
    tr, dv, te = load("train", a.train), load("dev", a.dev), load("test", a.test)
    SLOT = train_slots(CAT, tr, a.workers)
    SLOT.save_model(str(OUT / "slots.txt"))
    summary = {"config": vars(a), "catalog_tools": CAT.n, "embeddings": CAT.emb is not None}
    for setting, session_only in (("session", True), ("catalog", False)):
        t0 = time.time()
        R = {s: build(xs, a.k, session_only, a.workers, None if session_only else ctx_for(s, xs))
             for s, xs in (("train", tr), ("dev", dv), ("test", te))}
        X, y, g, _ = stack(R["train"])
        ranker = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1], "learning_rate": 0.05,
                            "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1, "num_threads": a.workers},
                           lgb.Dataset(X, y, group=g, feature_name=FEATS), 400)
        dt, tt = top1(ranker, R["dev"], dv), top1(ranker, R["test"], te)
        keep = [i for i, x in enumerate(dt) if x is not None]
        dk = [dt[i] for i in keep]
        GX, Gy = gate_X(dk), np.array([x[2] for x in dk], np.int8)
        gp = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
              "verbose": -1, "num_threads": a.workers}
        # thresholds from out-of-fold gate scores: a gate scored on its own training rows looks perfect
        # folds by server: steps of one conversation (or one API) on both sides would be memorized
        fold = np.array([zlib.crc32(dv[i]["tool"].split("/")[1].encode()) // 10 % 2 for i in keep])  # % 10 is the split itself
        oof = np.zeros(len(dk))
        for f in (0, 1):
            oof[fold == f] = lgb.train(gp, lgb.Dataset(GX[fold != f], Gy[fold != f]), 300).predict(GX[fold == f])
        th = thresholds(oof, Gy.astype(np.float32))
        gate = lgb.train(gp, lgb.Dataset(GX, Gy), 300)
        summary[setting] = {"dev": report(f"{setting} dev ", dt, gate, th), "test": report(f"{setting} test", tt, gate, th),
                            "seconds": round(time.time() - t0),
                            "ranker_gain": dict(zip(FEATS, map(float, ranker.feature_importance("gain").round(1))))}
        ranker.save_model(str(OUT / f"ranker_{setting}.txt"))
        gate.save_model(str(OUT / f"gate_{setting}.txt"))
    tag = "emb" if CAT.emb is not None else "bm25"
    json.dump(summary, open(OUT / f"run_{tag}_k{a.k}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--dev", type=int, default=8000)
    ap.add_argument("--test", type=int, default=0, help="0 = every test example")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.test = a.test or None
    run(a)
