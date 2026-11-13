"""Latency ablation: prefetchd whose shards never rebuild a tau2 environment between tickets.

Results are wrong on purpose (every session shares one dirty environment per domain); only the
latency numbers mean anything. Same arguments as prefetchd.py.

Usage: python3 -I bench/ablate_nobuild.py <tau-bench> serve [--shards N] [--port P] [--models DIR]
"""

import multiprocessing
import os
import runpy
import sys

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B)
import realtools as RT  # noqa: E402

_build, _envs = RT.make_env, {}


def make_env(domain, task=None):
    if domain not in _envs:
        _envs[domain] = _build(domain, task)
    return _envs[domain]


RT.make_env = make_env
multiprocessing.set_start_method("fork")  # 3.14 defaults to forkserver, whose children would re-import unpatched
sys.argv[0] = os.path.join(B, "prefetchd.py")
runpy.run_path(sys.argv[0], run_name="__main__")
