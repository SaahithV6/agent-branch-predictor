"""Self-check for gpred's sentence-context features and saved-model compatibility. No training.

Usage: python3 -I bench/test_gpred.py <tau-bench>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpred as G  # noqa: E402

TEXT = "Sure, my name is Ivan Hernandez and I live in San Diego, 92133."


def fz(context):
    f = G.Featurizer([])
    f.args = {"find_user_id_by_name_zip": ("first_name", "last_name", "zip")}
    f.tools = sorted(f.args)
    f.shapes = {"find_user_id_by_name_zip|first_name": G.Counter({"[A-Z][a-z]+": 1}),
                "find_user_id_by_name_zip|last_name": G.Counter({"[A-Z][a-z]+": 1}),
                "find_user_id_by_name_zip|zip": G.Counter({r"\d+": 1})}
    f.config["context"] = context
    return f


def ctx(f, arg, value):
    row = next(r for r in f.rows(TEXT, [], set()) if r[1] == arg and r[2] == value)
    return row[3][-4:]


def test_context_tells_first_name_from_last():
    f = fz(True)
    assert len(f.names()) == len(f.rows(TEXT, [], set())[0][3])
    _, prev_same, next_same, _ = ctx(f, "first_name", "Ivan")
    assert (prev_same, next_same) == (0, 1)  # "is Ivan Hernandez": a capitalised word follows
    _, prev_same, next_same, _ = ctx(f, "last_name", "Hernandez")
    assert (prev_same, next_same) == (1, 0)
    assert ctx(f, "first_name", "Ivan")[0] != ctx(f, "last_name", "Ivan")[0]  # word before is hashed per argument


def test_models_saved_before_config_load_without_context():
    meta = fz(True).meta()
    del meta["config"]
    old = G.Featurizer.load(meta)
    assert old.config == G.SAVED_BEFORE_CONFIG and not old.config["context"]
    assert len(old.names()) == len(fz(True).names()) - len(G.OPTIONAL["context"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"{len(tests)} passed")
