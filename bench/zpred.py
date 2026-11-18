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
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
import zlib
from array import array
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolindex import tokens as name_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CATALOGS = [ROOT / "data/catalogs/tools_1m.jsonl", ROOT / "data/calls/tools_toolbench_traj.jsonl",
            ROOT / "data/calls/tools_toolace.jsonl", ROOT / "data/calls/tools_taumt.jsonl"]
SOURCES = ("toolbench", "toolace", "taumt")  # bench/calls.py <source> -> data/calls/<source>.jsonl
WHOLE = ("taumt",)  # small sources: every example is used, never sampled down with the big ones
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
    def __init__(self, paths=CATALOGS, size=None, keep=(), emb=True):
        """size: scale the catalog to this many tools. Below the real count, a random subset that always holds
        `keep` (every tool the evaluated examples touch); above it, padded with mirror servers: exact copies of
        random tools under a new server name, the hardest lookalike there is (only the server tells them apart)."""
        t0 = time.time()
        self.tools, self.pos = [], {}
        vocab, rows, cols, lens = {}, array("i"), array("i"), []
        total = sum(1 for p in paths for _ in open(p)) if size else 0
        p_in = min(1.0, (size - len(keep)) / max(1, total - len(keep))) if size else 1.0
        extra = max(0.0, (size - total) / total) if size else 0.0
        rng = random.Random(0)
        for p in paths:
            for line in open(p):
                t = json.loads(line)
                if t["id"] in self.pos or (t["id"] not in keep and rng.random() > p_in):
                    continue
                rec = {"id": t["id"], "server": t["server"], "name": t["name"],
                       "args": t.get("args", []), "nt": name_tokens(t["name"]), "st": name_tokens(t["server"])}
                ws = collections.Counter(tool_words(t)).items()
                copies = int(extra) + (rng.random() < extra % 1)
                for j in range(1 + copies):
                    if j:
                        rec = {**rec, "id": f"{t['id']}~m{j}", "server": f"{t['server']}~m{j}"}
                    d = len(self.tools)
                    self.pos[rec["id"]] = d
                    self.tools.append(rec)
                    lens.append(sum(c for _, c in ws))
                    for w, c in ws:
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
        # server -> its tools, and server-name words -> servers: retrieval by server (servers already called,
        # servers the request names) is what scopes a million tools down to the few that matter
        self.by_server = collections.defaultdict(list)
        for d, t in enumerate(self.tools):
            self.by_server[t["server"]].append(d)
        self.servers = list(self.by_server)
        self.server_words = collections.defaultdict(list)
        for i, sv in enumerate(self.servers):
            for w in set(words(" ".join(name_tokens(sv)))):
                self.server_words[w].append(i)
        self.server_idf = {w: math.log(1 + len(self.servers) / len(v)) for w, v in self.server_words.items()}
        # server-level BM25: a server's document is every word of its tools (tf log-damped), so a request that
        # describes what an API does, not what one endpoint is called, still finds the server
        ns = len(self.servers)
        srv = np.empty(n, np.int64)
        for i, sv in enumerate(self.servers):
            srv[self.by_server[sv]] = i
        self.srv = srv
        key = np.repeat(np.arange(len(vocab), dtype=np.int64), np.diff(self.ptr)) * ns + srv[self.post_doc]
        uk, inv = np.unique(key, return_inverse=True)
        self.s_doc = (uk % ns).astype(np.int32)
        self.s_tf = np.log1p(np.bincount(inv, weights=self.post_tf)).astype(np.float32)
        self.s_ptr = np.searchsorted(uk // ns, np.arange(len(vocab) + 1))
        sdl = np.bincount(self.s_doc, weights=self.s_tf, minlength=ns)
        sdf = np.diff(self.s_ptr)
        self.s_idf = np.log(1 + (ns - sdf + 0.5) / (sdf + 0.5)).astype(np.float32)
        self.s_norm = (BM25_K1 * (1 - BM25_B + BM25_B * sdl / sdl.mean())).astype(np.float32)
        del key, uk, inv
        # tools.npy is aligned with the full catalog's order; a resized catalog runs BM25-only
        # float32 in RAM (1 GB at 1M x 256): numpy has no float16 BLAS, a float16 mat-vec is ~10x slower
        self.emb = np.load(EMB / "tools.npy").astype(np.float32) if emb and (EMB / "tools.npy").exists() and not size else None
        if self.emb is not None and len(self.emb) < n:  # tools appended after tools.npy was made (later catalog files)
            # ponytail: they get zero vectors (BM25 only); rerun bench/embed.py tools to embed them
            self.emb = np.concatenate([self.emb, np.zeros((n - len(self.emb), self.emb.shape[1]), np.float32)])
        assert self.emb is None or len(self.emb) == n, "tools.npy is stale: rerun bench/embed.py tools"
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

    def server_bm25(self, qwords):
        """Scores for every server."""
        sc = np.zeros(len(self.servers), np.float32)
        for v in {self.vocab[w] for w in qwords if w in self.vocab}:
            s, e = self.s_ptr[v], self.s_ptr[v + 1]
            if e - s > DF_CAP * len(self.servers):
                continue
            d, tf = self.s_doc[s:e], self.s_tf[s:e]
            sc[d] += self.s_idf[v] * tf * (BM25_K1 + 1) / (tf + self.s_norm[d])
        return sc


# ---------------------------------------------------------------------------------- context values
KV = re.compile(r"""['"]([A-Za-z_][\w\-]{0,40})['"]\s*:\s*(?:['"]([^'"]{1,120})['"]|(-?\d+(?:\.\d+)?)|(True|False|true|false))""")
QUOTED = re.compile(r"""["'“]([^"'”]{2,80})["'”]""")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"https?://\S+")
CAPS = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*")


CODE = re.compile(r"(?<![\w#-])#?(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{5,}(?![\w-])")  # mixed letters+digits: ids, VINs, hashes
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


def json_scalars(obj, key):
    """(key, value) for list items and a bare scalar: the values KV misses. A bare result is keyed by its tool name."""
    if isinstance(obj, dict):
        return [kv for k, v in obj.items() for kv in json_scalars(v, k) if isinstance(v, (dict, list))]
    if isinstance(obj, list):
        return [kv for v in obj for kv in json_scalars(v, key)]
    return [] if obj is None else [(key, str(obj))]


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
        try:  # JSON results: list items and a bare scalar result carry no "key": "value" pair for KV to find
            obj = json.loads(p["result"])
        except ValueError:
            obj = None
        for k, v in json_scalars(obj, p["tool"].rsplit("/", 1)[-1])[:200]:
            out.append((v, frozenset(name_tokens(k) | {k.lower()}), "result", frozenset(), rec))
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


# whether the agent passes an argument at all: "required" in a schema is a weak hint (agents skip required args
# and pass optional ones), so it is learned
INC_FEATS = ["required", "t_num", "t_bool", "n_args", "arg_idx", "has_example", "best", "second", "n_vals",
             "best_key_hit", *[f"b_{k}" for k in KINDS], *[f"a_{c}" for c in ARGCATS]]
INC = None  # LightGBM include model, trained with SLOT
VALS = 3  # candidate values kept per argument; a fan-out (open every order) needs as many as are prefetched


def arg_options(tool, values, slot=None, inc=True):
    """Per argument (name, P(passed), [(value, P(right | passed))] best 3), and the include-model rows."""
    slot = slot or SLOT
    values = with_examples(tool, values)
    ex_keys = TOOL_EXAMPLES.get(tool["id"], {})
    out, rows = [], []
    for j, a in enumerate(tool["args"]):
        sc, o = np.zeros(0), []
        if values:
            R = np.array(slot_rows(a, values), np.float32)
            R[:, SLOT_FEATS.index("n_args")] = len(tool["args"])
            sc = slot.predict(R, num_threads=1)
            o, got = [], set()
            for i in np.argsort(-sc):  # best VALS *distinct* values: one value is often found under several keys
                v = str(values[i][0]).strip().lower()
                if v not in got:
                    got.add(v)
                    o.append(i)
                    if len(o) == VALS:
                        break
        b = o[0] if len(o) else None
        at = name_tokens(a["name"]) | {a["name"].lower()}
        typ = a.get("type", "string")
        rows.append([bool(a.get("required")), typ in ("integer", "number"), typ == "boolean", len(tool["args"]), j,
                     a["name"] in ex_keys, sc[b] if b is not None else 0.0, sc[o[1]] if len(o) > 1 else 0.0,
                     len(values), b is not None and bool(values[b][1] & at),
                     *[b is not None and values[b][2] == k for k in KINDS], *[c in at for c in ARGCATS]])
        out.append((a["name"], [(values[i][0], float(sc[i])) for i in o]))
    if not inc:
        return None, rows
    pinc = INC.predict(np.array(rows, np.float32), num_threads=1) if rows else []
    return [(n, float(p), vs) for (n, vs), p in zip(out, pinc)], rows


def configs(opts, m):
    """The m most likely argument sets -> [(args, log P(all exactly right))]. Each argument is left out or passed with
    one of its best values; the best set plus every one-change alternative to it."""
    choice = [sorted([(math.log(max(1 - p, 1e-6)), None)] + [(math.log(max(p * v, 1e-6)), val) for val, v in vs],
                     key=lambda c: -c[0]) for _, p, vs in opts]
    base = sum(c[0][0] for c in choice)
    cands = [(base, -1, 0)] + [(base - c[0][0] + c[j][0], i, j) for i, c in enumerate(choice) for j in range(1, len(c))]
    out = []
    for lp, i, j in sorted(cands, key=lambda c: -c[0])[:m]:
        pick = [0] * len(choice)
        if i >= 0:
            pick[i] = j
        args = {opts[k][0]: choice[k][q][1] for k, q in enumerate(pick) if choice[k][q][1] is not None}
        out.append((args, lp))
    return out


def slot_data(cat, exs):
    X, y = [], []
    for ex in exs:
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
    return np.array(X, np.float32), np.array(y, np.int8)


def train_slots(cat, examples, workers):
    """Slot model (which value fills an argument) and include model (is the argument passed at all)."""

    def fit(X, y, feats):
        return lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                          "verbose": -1, "num_threads": workers}, lgb.Dataset(X, y, feature_name=feats), 300)

    X, y = slot_data(cat, examples)
    slot = fit(X, y, SLOT_FEATS)
    print(f"slot model: {len(y):,} (arg, value) pairs, {y.mean():.3f} positive", flush=True)
    # include features carry slot scores from a slot model that never saw the example (2 folds)
    halves, Xi, yi = (examples[0::2], examples[1::2]), [], []
    for h in (0, 1):
        m = fit(*slot_data(cat, halves[1 - h]), SLOT_FEATS)
        for ex in halves[h]:
            t = cat.tools[cat.pos[ex["tool"]]]
            Xi += arg_options(t, context_values(ex), slot=m, inc=False)[1]
            yi += [a["name"] in ex["args"] for a in t["args"]]
    inc = fit(np.array(Xi, np.float32), np.array(yi, np.int8), INC_FEATS)
    print(f"include model: {len(yi):,} arguments, {np.mean(yi):.3f} passed", flush=True)
    return slot, inc


def same_args(a, b):
    norm = lambda d: {str(k): str(v).strip().lower() for k, v in d.items()}  # noqa: E731
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------------- features
FEATS = ["bm25", "rank", "rel_top", "gap_next", "same_server_last", "server_seen", "called_before", "is_last_tool",
         "q_name_overlap", "q_server_overlap", "n_args", "n_req", "req_fillable", "n_key_hits", "n_prior", "emb_cos",
         "emb_rank", "session", "server_named", "srv_bm25", "srv_rank", "srv_rel_top"]


PER_SERVER = 20  # tools taken per retrieved server, best by BM25: big servers (Azure: 1000s of ops) stay bounded
SRV_K, SRV_TOOLS = 5, 10  # servers taken by server-level BM25, and tools from each


def named_servers(cat, query, top=10):
    """Servers whose name the request spells out: {server: share of its name's idf the query covers}."""
    q = set(words(query))
    hit = collections.Counter()
    for w in q:
        if len(cat.server_words.get(w, ())) <= 2000:  # "api", "service": names everything, means nothing
            for i in cat.server_words.get(w, ()):
                hit[i] += cat.server_idf[w]
    out = {}
    for i, got in hit.most_common(top * 5):
        sv = cat.servers[i]
        cov = got / sum(cat.server_idf.get(w, 0) for w in set(words(" ".join(name_tokens(sv)))))
        if cov >= 0.5:
            out[sv] = cov
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:top])


def candidates(cat, ex, k, session_only, ctx_vec=None):
    """Candidate doc indices and their BM25 scores. Also leaves the server scores in ex["_srv"] for featurize."""
    qr = words(ex["query"])
    q = list(qr)
    if ex["prior"]:
        last = ex["prior"][-1]
        q += words(" ".join(re.split(r"[/_]", last["tool"]))) + words(last["result"][:400])
    ssc = cat.server_bm25(qr)
    ex["_srv"] = ssc
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
    parts = [top]
    if ex["prior"]:  # a later step: the last result's words can crowd the request out of the top K
        scr = cat.bm25(qr)
        parts.append(np.argpartition(-scr, k // 2)[:k // 2])
    servers = {p["tool"].split("/")[1] for p in ex["prior"] if p["tool"].count("/") >= 2} | set(named_servers(cat, ex["query"]))
    for sv in servers:
        d = np.array(cat.by_server.get(sv, ()), np.int64)
        if len(d):
            parts.append(d[np.argsort(-sc[d])[:PER_SERVER]])
    for i in np.argpartition(-ssc, SRV_K)[:SRV_K]:  # servers whose tools, taken together, match the request
        d = np.array(cat.by_server[cat.servers[i]], np.int64)
        parts.append(d[np.argsort(-sc[d])[:SRV_TOOLS]])
    top = np.unique(np.concatenate(parts))
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
    named = named_servers(cat, ex["query"])
    order = np.argsort(-bm)
    rank = np.empty(len(docs), np.int32)
    rank[order] = np.arange(len(docs))
    top = bm[order[0]] if len(bm) else 1.0
    nxt = np.sort(bm)[::-1]
    erank = np.empty(len(docs), np.int32)
    if emb is not None:
        erank[np.argsort(-emb)] = np.arange(len(docs))
    ssc = ex["_srv"]
    sid = cat.srv[docs]
    cs = ssc[sid]  # each candidate's server score, and its rank among all servers
    us, inv = np.unique(sid, return_inverse=True)
    srank = np.array([(ssc > ssc[u]).sum() for u in us])[inv]
    stop = float(ssc.max()) or 1.0
    rows = []
    for i, d in enumerate(docs):
        t = cat.tools[d]
        req, frac, hits = fillable(t, values)
        rows.append([bm[i], rank[i], bm[i] / (top + 1e-6), bm[i] - (nxt[1] if len(nxt) > 1 else 0),
                     t["server"] == last_server, t["server"] in servers, called[t["id"]], t["id"] == last,
                     len(t["nt"] & qw) / (len(t["nt"]) or 1), len(t["st"] & qw) / (len(t["st"]) or 1),
                     len(t["args"]), req, frac, hits, len(ex["prior"]),
                     emb[i] if emb is not None else np.nan, erank[i] if emb is not None else -1, session_only,
                     named.get(t["server"], 0.0), cs[i], srank[i], cs[i] / stop])
    return np.array(rows, np.float32)


# ---------------------------------------------------------------------------------- per-example work (forked workers)
CAT = None


def work(job):
    ex, k, session_only, ctx = job
    t0 = time.perf_counter()
    docs, bm, emb = candidates(CAT, ex, k, session_only, ctx)
    if not len(docs):
        return None
    X = featurize(CAT, ex, docs, bm, emb, session_only)
    nxt = CAT.pos[ex["tool"]]
    later = {CAT.pos[t] for t, _ in ex["future"]}
    # relevance: 2 = the very next call, 1 = called later in the task, 0 = never
    y = np.array([2 if d == nxt else 1 if d in later else 0 for d in docs], np.int8)
    return X, y, docs, int((y > 0).any()), (time.perf_counter() - t0) * 1000


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
SPEC = 1  # calls prefetched per prediction: the SPEC most likely (tool, arguments) pairs
TK = 5  # ranked tools whose argument sets compete for those slots
TEMP = 1.0  # ranker score -> log P(tool) by softmax at this temperature, picked on dev


def bucket(ex):
    """Reporting only: first call / next server already called / a server new to this conversation."""
    if not ex["prior"]:
        return "first"
    return "same" if ex["tool"].split("/")[1] in {p["tool"].split("/")[1] for p in ex["prior"] if p["tool"].count("/") >= 2} else "new"


def speculate(model, X, docs, ex):
    """Gate features and the calls to prefetch: [(row, tool, args)], the SPEC most likely whole calls among the top TK
    tools, P(call) = P(tool) * P(its argument set). One tool may take several slots with different arguments."""
    s = model.predict(X, num_threads=1)
    order = np.argsort(-s)
    srt = s[order]
    lt = srt[:TK] / TEMP - np.logaddexp.reduce(s / TEMP)
    values = context_values(ex)
    pool = []
    for j, i in enumerate(order[:TK]):
        t = CAT.tools[docs[i]]
        for args, lp in configs(arg_options(t, values)[0], SPEC):
            pool.append((lt[j] + lp, i, t, args, j))
    picks = sorted(pool, key=lambda c: -c[0])[:SPEC]
    top = picks[0]
    g = [srt[0], srt[0] - (srt[1] if len(srt) > 1 else 0), top[0], np.logaddexp.reduce([c[0] for c in picks]),
         len(top[3]), top[4], len({c[1] for c in picks}), *X[top[1]]]
    return g, [(i, t, args) for _, i, t, args, _ in picks]


def top1(model, res, examples):
    """Per example: (gate features, tool right, call right, target retrieved, ms, bucket). The prediction is the
    top SPEC ranked tools with arguments filled by the slot model; it is right if the agent makes any of those
    exact calls later on (a speculative prefetch that pays off)."""
    global RK
    RK = model
    with mp.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(_top1, list(zip(res, examples)), chunksize=64)


RK, WORKERS = None, 8


def _top1(job):
    r, ex = job
    if r is None:
        return None
    X, y, docs, hit, ms = r
    t0 = time.perf_counter()
    g, picks = speculate(RK, X, docs, ex)
    tool_right = right = False
    for i, t, args in picks:
        tool_right |= bool(y[i] > 0)
        right |= bool(y[i] > 0) and any(t["id"] == tid and same_args(args, a) for tid, a in ex["future"])
    # ms: retrieve + featurize + rank + fill, one thread (labeling included; it is a few set lookups)
    return g, tool_right, right, hit, ms + (time.perf_counter() - t0) * 1000, bucket(ex)


def gate_X(t):
    return np.array([x[0] for x in t], np.float32)


def report(name, t, gate, th):
    n = len(t)
    tt = [x for x in t if x is not None]
    p = gate.predict(gate_X(tt))
    rows = {"examples": n, "retrieved": round(sum(x[3] for x in tt) / n, 4),
            "tool_top1": round(sum(x[1] for x in tt) / n, 4), "call_top1": round(sum(x[2] for x in tt) / n, 4),
            "ms_p50": round(float(np.percentile([x[4] for x in tt], 50)), 2),
            "ms_p99": round(float(np.percentile([x[4] for x in tt], 99)), 2), "spec": SPEC,
            **{f"{b}": {"share": round(sum(x[5] == b for x in tt) / n, 3),
                        "retrieved": round(float(np.mean([x[3] for x in tt if x[5] == b])), 3),
                        "tool": round(float(np.mean([x[1] for x in tt if x[5] == b])), 3),
                        "call": round(float(np.mean([x[2] for x in tt if x[5] == b])), 3)}
               for b in ("first", "same", "new") if any(x[5] == b for x in tt)}}
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


def load(split, limit=None, seed=0, sources=SOURCES):
    xs = [json.loads(line) for src in sources for line in open(ROOT / f"data/calls/{src}.jsonl")
          if f'"split": "{split}"' in line]
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
    if not p.exists() or CAT.emb is None:
        return None
    m = json.load(open(EMB / f"ctx_{split}.json"))  # example key -> row
    arr = np.load(p, mmap_mode="r")
    keys = [f"{ex['src']}:{ex['conv']}:{len(ex['prior'])}" for ex in examples]
    print(f"ctx {split}: {sum(k in m for k in keys)} of {len(keys)} examples embedded", flush=True)
    return [np.asarray(arr[m[k]]) if k in m else None for k in keys]  # not embedded: BM25 only


def run(a):
    global CAT
    OUT.mkdir(parents=True, exist_ok=True)
    global SLOT, INC, SPEC, TEMP, WORKERS
    SPEC, WORKERS = a.spec, a.workers
    big = tuple(x for x in SOURCES if x not in WHOLE)
    tr = load("train", a.train, sources=big) + load("train", sources=WHOLE)
    dv = load("dev", a.dev, sources=big) + load("dev", sources=WHOLE)
    # test per source: ToolBench (multi-step, real results) stays comparable with earlier runs
    tests = {f"test_{src}": load("test", a.test, sources=(src,)) for src in SOURCES}
    keep = {t for xs in (tr, dv, *tests.values()) for x in xs for t in (*x["session"], x["tool"], *(f for f, _ in x["future"]))}
    CAT = Catalog(size=a.size, keep=keep, emb=not a.no_emb)
    SLOT, INC = train_slots(CAT, tr, a.workers)
    if not a.size:  # the committed models are the full-catalog ones
        SLOT.save_model(str(OUT / "slots.txt"))
        INC.save_model(str(OUT / "include.txt"))
    summary = {"config": vars(a), "catalog_tools": CAT.n, "embeddings": CAT.emb is not None}
    for setting, session_only in (("session", True), ("catalog", False)):
        if setting not in a.settings:
            continue
        t0 = time.time()
        R = {s: build(xs, a.k, session_only, a.workers, None if session_only else ctx_for(s.split("_")[0], xs))
             for s, xs in (("train", tr), ("dev", dv), *tests.items())}
        X, y, g, _ = stack(R["train"])
        ranker = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1], "learning_rate": 0.05,
                            "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1, "num_threads": a.workers},
                           lgb.Dataset(X, y, group=g, feature_name=FEATS), 400)
        best = None
        for TEMP in (0.5, 1.0, 2.0, 4.0):  # how much tool doubt may trade against argument doubt
            dt = top1(ranker, R["dev"], dv)
            hit = sum(x[2] for x in dt if x is not None)
            print(f"{setting} temp {TEMP}: dev calls right {hit / len(dv):.4f}", flush=True)
            if best is None or hit > best[0]:
                best = (hit, TEMP, dt)
        _, TEMP, dt = best
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
        summary[setting] = {"dev": report(f"{setting} dev ", dt, gate, th),
                            **{k: report(f"{setting} {k}", top1(ranker, R[k], xs), gate, th) for k, xs in tests.items()},
                            "temp": TEMP, "seconds": round(time.time() - t0),
                            "ranker_gain": dict(zip(FEATS, map(float, ranker.feature_importance("gain").round(1))))}
        if not a.size:
            ranker.save_model(str(OUT / f"ranker_{setting}.txt"))
            gate.save_model(str(OUT / f"gate_{setting}.txt"))
            json.dump({"spec": SPEC, "k": a.k, "temp": TEMP, "tk": TK, "thresholds": th},
                      open(OUT / f"gate_{setting}.json", "w"))
    tag = ("emb" if CAT.emb is not None else "bm25") + (f"_n{CAT.n}" if a.size else "") + (f"_spec{SPEC}" if SPEC > 1 else "")
    json.dump(summary, open(OUT / f"run_{tag}_k{a.k}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--dev", type=int, default=8000)
    ap.add_argument("--test", type=int, default=0, help="0 = every test example")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, help="resize the catalog to this many tools (see Catalog)")
    ap.add_argument("--settings", default="session,catalog")
    ap.add_argument("--no-emb", action="store_true", help="BM25 + server retrieval only")
    ap.add_argument("--spec", type=int, default=1, help="calls prefetched per prediction")
    a = ap.parse_args()
    a.test = a.test or None
    run(a)
