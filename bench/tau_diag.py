"""Where v2 loses on the tuning half: misses and wasted guesses per tool, and the rules behind waste.
Usage: python3 -I bench/tau_diag.py <tau-bench>"""
import os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import tau_replay as T

cfg = dict(text=True, levels=(1, 2), mention=True, theta=0.3)
for dom, tasks in T.suites().items():
    tune, _ = T.halves(tasks)
    users = sorted({t["user"] for t in tune}); fold = {u: i % 5 for i, u in enumerate(users)}
    miss, waste, waste_rule = Counter(), Counter(), Counter()
    for f in range(5):
        keep = T.fit([t for t in tune if fold[t["user"]] != f], dom, cfg)
        for t in [t for t in tune if fold[t["user"]] == f]:
            results, pred = [], set()
            actual = {T.call_key(n, a) for n, a in t["actions"] if n in T.READS[dom]}
            for n, a in t["actions"]:
                if n in T.READS[dom]:
                    p = T.predict(keep, t, results, cfg); pred |= p
                    if T.call_key(n, a) not in p:
                        miss[(n, "first call" if not results else "missed")] += 1
                    results.append((n, T.READS[dom][n](**a)))
            idx = T.sources(results)
            for r, b in keep:
                for k, bb in T.candidates(r, t, idx, cfg):
                    if bb == b and k not in actual:
                        waste_rule[(r[0], tuple((a, s[0], s[-1]) for a, s in r[1]), b)] += 1
            for k in pred - actual: waste[k.split("{")[0]] += 1
    print(f"== {dom}\n  misses: {dict(miss.most_common())}\n  waste: {dict(waste.most_common())}")
    for k, v in waste_rule.most_common(8): print("   ", v, k)
