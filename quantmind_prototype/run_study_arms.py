"""User-study arm comparison: explanations shown (A) vs hidden (B).

Reads two response files in the format written by ``src/study_log.record_session``
and compares them on the outcome measures fixed in advance:

  * SUS score (0-100), standard scoring
  * trust change: mean(post1, post2) - mean(pre1, pre2). POST3 exists only in
    build A, so it is excluded here to keep the two arms comparable
  * task completion: share of tasks completed unaided, per task
  * understood_reason (task 3): share answering "yes"

Uncertainty is a 10,000-resample bootstrap 95% interval on the difference in
means (B minus A), with a permutation p-value alongside. The study is small, so
the minimum detectable effect at 80% power is reported and the result is to be
read as directional. Nothing is computed unless both files hold real rows.

Run:  python run_study_arms.py [--a study/responses_main.csv] [--b study/responses_b.csv]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src import config

STUDY = os.path.join(config.ROOT, "study")
OUT = os.path.join(config.RESULTS_DIR, "user_study_arms.json")
N_BOOT = 10_000
SEED = 20260924


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["task_status"] = df["task_status"].map(json.loads)
    df["trust_pre"] = (df.pre1 + df.pre2) / 2
    df["trust_post"] = (df.post1 + df.post2) / 2      # common items only
    df["trust_change"] = df.trust_post - df.trust_pre
    df["unaided_share"] = df.task_status.map(lambda s: np.mean([x == "unaided" for x in s]))
    return df


def diff_ci(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """B - A difference in means: bootstrap 95% interval and permutation p."""
    obs = b.mean() - a.mean()
    boots = np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
                      for _ in range(N_BOOT)])
    pool, na = np.concatenate([a, b]), len(a)
    perm = np.empty(N_BOOT)
    for i in range(N_BOOT):
        p = rng.permutation(pool)
        perm[i] = p[na:].mean() - p[:na].mean()
    return {"diff_B_minus_A": round(float(obs), 3),
            "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                     round(float(np.percentile(boots, 97.5)), 3)],
            "perm_p": round(float((np.abs(perm) >= abs(obs)).mean()), 3)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return round(float((b.mean() - a.mean()) / sp), 2) if sp > 0 else 0.0


def analyse(a_path: str, b_path: str) -> dict:
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return {"n_A": 0, "n_B": 0, "note": "arm files missing; nothing computed"}
    A, B = load(a_path), load(b_path)
    if len(B) == 0 or len(A) == 0:
        return {"n_A": len(A), "n_B": len(B), "note": "an arm is empty; nothing computed"}
    rng = np.random.default_rng(SEED)
    out = {"n_A": len(A), "n_B": len(B),
           "builds_in_files": {"A": sorted(set(A.build)), "B": sorted(set(B.build))}}
    for name, col in [("sus", "sus"), ("trust_change", "trust_change"),
                      ("trust_post", "trust_post"), ("unaided_share", "unaided_share")]:
        a, b = A[col].to_numpy(float), B[col].to_numpy(float)
        out[name] = {"mean_A": round(float(a.mean()), 3), "sd_A": round(float(a.std(ddof=1)), 3),
                     "mean_B": round(float(b.mean()), 3),
                     "sd_B": round(float(b.std(ddof=1)), 3) if len(b) > 1 else None,
                     "cohens_d": cohens_d(a, b) if len(b) > 1 else None,
                     **diff_ci(a, b, rng)}
    n_tasks = len(A.task_status.iloc[0])
    out["per_task_unaided"] = {
        f"task{i + 1}": {"A": round(float(np.mean([s[i] == "unaided" for s in A.task_status])), 2),
                         "B": round(float(np.mean([s[i] == "unaided" for s in B.task_status])), 2)}
        for i in range(n_tasks)}
    out["understood_reason_yes"] = {"A": round(float((A.understood_reason == "yes").mean()), 2),
                                    "B": round(float((B.understood_reason == "yes").mean()), 2)}
    sd_pool = float(np.sqrt((A.sus.var(ddof=1) + B.sus.var(ddof=1)) / 2))
    out["power"] = {"min_detectable_d_80pct": round(2.8 * np.sqrt(1 / len(A) + 1 / len(B)), 2),
                    "min_detectable_sus_points": round(2.8 * np.sqrt(1 / len(A) + 1 / len(B)) * sd_pool, 1),
                    "note": "two-sided alpha 0.05; the study is underpowered, read directionally"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(STUDY, "responses_main.csv"))
    ap.add_argument("--b", default=os.path.join(STUDY, "responses_b.csv"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    res = analyse(args.a, args.b)
    if res.get("n_B", 0) > 0 and res.get("n_A", 0) > 0:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
