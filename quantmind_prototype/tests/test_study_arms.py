"""Pipeline check for the arm comparison (run_study_arms.py).

The rows below are SYNTHETIC and exist only to exercise the code path. They are
built in a temporary directory, never written to study/, and never reported.
"""
from __future__ import annotations

import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_study_arms import analyse, load  # noqa: E402
from src import study_log  # noqa: E402

FIELDS = ["code", "build", "background", "task_status", "sus_items", "pre1", "pre2",
          "post1", "post2", "post3", "understood_reason", "comment_helped",
          "comment_hurt", "sus"]


def _write(path, build, n, sus_items, post):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i in range(n):
            w.writerow({"code": f"P-{build}{i}", "build": build, "background": "synthetic",
                        "task_status": json.dumps(["unaided", "partial", "failed"]),
                        "sus_items": json.dumps(sus_items), "pre1": 2, "pre2": 2,
                        "post1": post, "post2": post, "post3": 0 if build == "B" else 3,
                        "understood_reason": "yes", "comment_helped": "", "comment_hurt": "",
                        "sus": study_log.sus_score(sus_items)})


def test_missing_or_empty_arm_computes_nothing(tmp_path):
    a = tmp_path / "a.csv"
    _write(a, "A", 4, [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], 4)
    assert analyse(str(a), str(tmp_path / "missing.csv"))["n_B"] == 0


def test_arm_difference_direction_and_common_items(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, "A", 6, [4, 2, 4, 2, 4, 2, 4, 2, 4, 2], 4)   # SUS 75, trust +2
    _write(b, "B", 6, [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3)   # SUS 50, trust +1
    res = analyse(str(a), str(b))
    assert res["n_A"] == res["n_B"] == 6
    assert res["sus"]["diff_B_minus_A"] == -25.0
    assert res["trust_change"]["diff_B_minus_A"] == -1.0
    assert load(str(b)).trust_post.iloc[0] == 3           # post3 (absent in B) not used
    assert res["power"]["min_detectable_d_80pct"] > 1.5    # small n -> large MDE
