"""Threshold/level sweep on the tuning half only. Usage: python3 -I bench/sweep.py <tau-bench>"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import tau_replay as T
for dom, tasks in T.suites().items():
    tune, _ = T.halves(tasks)
    for text in (False, True):
        for z in (0.5, 1.0, 1.5):
            for theta in (0.3, 0.4, 0.5):
                s = T.cv(tune, dom, dict(text=text, levels=(1, 2), mention=text, theta=theta, stump=3, z=z))
                print(f"{dom:8} text={text!s:5} z={z} theta={theta}  precision={s['precision']:.0%} coverage={s['coverage']:.0%} calls_saved={s['calls_saved']:.0%}", flush=True)
