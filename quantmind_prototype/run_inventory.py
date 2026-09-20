"""Run inventory: every trained run, and which ones count as candidate strategies.

The Deflated Sharpe Ratio discounts for the number of strategy configurations tried.
Candidates are configurations evaluated as strategies on the held-out window:
price-only and sentiment agents on the core universe, and price-only agents on the
extended universe. Placebo runs are controls, and walk-forward refits repeat one
configuration on earlier windows, so neither is a candidate. Baselines are fixed
rules and are not trained.

Writes results/run_inventory.json. Reads locked artifacts only.
Run:  python run_inventory.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src import config, stats

R = os.path.join(config.ROOT, "results")


def _runs(sub: str, variant: str):
    out = []
    for p in sorted(glob.glob(os.path.join(R, sub, "runs", f"{variant}_seed*"))):
        m = json.load(open(os.path.join(p, "metrics.json")))["metrics"]["Sharpe"]
        r = pd.read_csv(os.path.join(p, "returns.csv"), index_col=0)["port_ret"].values
        out.append((m, r))
    return out


def main():
    groups = [("core", "price", "candidate"), ("core", "sentiment", "candidate"),
              ("core", "placebo", "control"), ("extended", "price", "candidate")]
    rows, cand = [], []
    for sub, variant, role in groups:
        runs = _runs("" if sub == "core" else "extended", variant)
        rows.append({"universe": sub, "variant": variant, "runs": len(runs), "role": role,
                     "window": "train 2015-2021, test 2022-2024", "counts_toward_trials": role == "candidate"})
        if role == "candidate":
            cand += [(sub, variant, s, r) for s, r in runs]
    for sub in ("core", "extended"):
        p = os.path.join(R, "" if sub == "core" else "extended", "walkforward.json")
        if os.path.exists(p):
            w = json.load(open(p))
            rows.append({"universe": sub, "variant": "price (walk-forward refits)",
                         "runs": len(w["windows"]) * len(w["seeds"]), "role": "robustness refit",
                         "window": "three expanding windows", "counts_toward_trials": False})
    sharpes = np.array([c[2] for c in cand])
    n_all = len(cand)
    out = {"runs": rows, "candidate_trials": n_all, "candidate_breakdown":
           {f"{u}/{v}": sum(1 for c in cand if c[0] == u and c[1] == v) for u, v, _ in groups if _ == "candidate"}}
    core_price = [c for c in cand if c[0] == "core" and c[1] == "price"]
    best = max(core_price, key=lambda c: c[2])
    for label, n in (("core_best_price_seed_N10", 10), ("core_best_price_seed_all_candidates", n_all)):
        d = stats.deflated_sharpe_ratio(best[3], n_trials=n, all_trial_sharpes=sharpes[:n] if n <= n_all else sharpes)
        out[label] = {"n_trials": n, "deflated_sharpe": d["deflated_sharpe"],
                      "expected_max_sharpe_ann": d["expected_max_sharpe_ann"]}
    ext = [c for c in cand if c[0] == "extended"]
    if ext:
        eb = max(ext, key=lambda c: c[2])
        d = stats.deflated_sharpe_ratio(eb[3], n_trials=n_all, all_trial_sharpes=sharpes)
        out["extended_best_price_seed_all_candidates"] = {"n_trials": n_all, "sharpe": eb[2],
                                                          "deflated_sharpe": d["deflated_sharpe"]}
    json.dump(out, open(os.path.join(R, "run_inventory.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
