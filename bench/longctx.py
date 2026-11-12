"""Train the predictor on what prefetchd actually sees in a long session.

Usage: python3 -I bench/longctx.py <tau-bench> <out_dir> [domain ...]

gpred trains on single chats (~7-20 calls) and sees everything since the chat began. A long
session that never says where a ticket ends hands it stale results and another customer's words,
which it never saw in training: on telecom, readiness fell from 97% (memory reset per ticket) to
11%. Here every prediction sees exactly the service's view (the last WINDOW reads, the last
TEXT_CHARS of text), and the training set adds chained sessions: each chat also appears after 1-3
other chats of the same customer fold, with no marker between them. Held-out folds stay clean.
Writes the same per-fold files as gpred.export, for prefetchd --models.
"""

import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402
import traces as TR  # noqa: E402

WINDOW = int(os.environ.get("WINDOW", "8"))
TEXT_CHARS = int(os.environ.get("TEXT_CHARS", "1500"))
PREFIX = int(os.environ.get("PREFIX", "3"))
WORKERS = int(os.environ.get("WORKERS", "3"))


def view_steps(t):
    """G.steps, seen through the service's bounded window."""
    reads = [s for s in t["steps"] if s["read"] and s["ok"] and G.scalar_args(s["args"])]
    for i, s in enumerate(reads):
        avail = [x for x in (reads[i - 1]["t_result"] if i else None, s.get("t_text")) if x is not None]
        win = reads[max(0, i - WINDOW):i]
        yield (s["text"][-TEXT_CHARS:], [(r["name"], r["result"]) for r in win], {G.key(r["name"], r["args"]) for r in win},
               {G.key(r["name"], r["args"]) for r in reads[i:]}, G.key(s["name"], s["args"]), max(avail) if avail else None, s["t_call"])


G.steps = view_steps  # dataset() and trace() look it up at call time; worker processes import this module too


def chain(ts, rng):
    """One session: 1-PREFIX other chats, then t[-1]. Text accumulates across them, as in prefetchd."""
    text, steps = "", []
    for t in ts:
        base = ""
        for s in t["steps"]:
            if s["text"] != base:
                text += s["text"][len(base):] if s["text"].startswith(base) else s["text"]
                base = s["text"]
            steps.append({**s, "text": text})
    last = ts[-1]
    return {**last, "steps": steps, "ncalls": len(steps), "chained": len(ts) - 1}


def augment(data, fold, rng):
    by = {}
    for t in data:
        by.setdefault(fold[t["group"]], []).append(t)
    out = list(data)
    for t in data:
        pool = by[fold[t["group"]]]
        out.append(chain(rng.sample(pool, rng.randint(1, PREFIX)) + [t], rng))
    return out


DATA = None  # set before the pool forks: workers share it copy-on-write instead of each unpickling a copy


def job(args):
    return G.one_fold((DATA, *args))


def export(here, domain):
    data = TR.load(domain)
    groups = sorted({t["group"] for t in data})
    fold = {g: i % 5 for i, g in enumerate(groups)}
    global DATA
    DATA = augment(data, fold, random.Random(0))
    os.makedirs(here, exist_ok=True)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as ex:
        for f, i, c in ex.map(job, [(fold, f, here, domain, 0.85, 0.5) for f in range(5)]):
            print(f"{domain} fold {f} inject={i} cache={c}", flush=True)
    json.dump({"fold_of_group": fold, "window": WINDOW, "text_chars": TEXT_CHARS}, open(os.path.join(here, f"gpred_{domain}_folds.json"), "w"))


if __name__ == "__main__":
    for d in sys.argv[3:] or ("retail", "airline", "telecom"):
        export(sys.argv[2], d)
