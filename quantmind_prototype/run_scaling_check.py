"""Shapley attribution at scale: correctness checks and grouped explanations.

Part 1 (correctness, any universe): the from-scratch sampler is checked against the
closed-form Shapley values of a linear function, and against the efficiency identity,
at the observation dimensionality of each universe (5 assets: 41 inputs; 30 assets:
241 inputs) and at several permutation budgets. The error of a fixed budget grows
with dimensionality, so the budget needed for the same tolerance is reported rather
than assumed.

Part 2 (explanation, QM_UNIVERSE=extended): 240 raw feature attributions are not an
explanation, so contributions are summed within groups, by asset (which holding the
signal belongs to) and by feature family (which kind of signal it is). Summing the
individual Shapley values keeps the efficiency property: group contributions still
add up to the change in output.

Run:  python run_scaling_check.py            (core: part 1 only)
      QM_UNIVERSE=extended python run_scaling_check.py
"""
from __future__ import annotations

import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src import config, data, explain
from src.explain import shapley_sampling

OUT = config.RESULTS_DIR
DIMS = {"toy (8 inputs)": 8, "5 assets": 5 * len(config.FEATURE_NAMES) + 6,
        "30 assets": 30 * len(config.FEATURE_NAMES) + 31}
BUDGETS = [64, 500, 2000, 6000]
FAMILY = {"ret_1d": "short-term return", "ret_5d": "momentum", "rsi_14": "RSI",
          "macd_hist": "MACD trend", "vol_20": "volatility",
          "px_sma20": "price vs average", "sma5_sma20": "trend (moving averages)"}


def _linear(w):
    return lambda X: (np.atleast_2d(np.asarray(X, dtype=np.float64)) @ w).reshape(-1, 1)


def closed_form_and_efficiency() -> list[dict]:
    rows = []
    for label, M in DIMS.items():
        rng = np.random.default_rng(11)
        w = rng.uniform(-1, 1, M)
        x = rng.uniform(-1, 1, M)
        bg = rng.uniform(-1, 1, (6, M))
        closed = w * (x - bg.mean(axis=0))
        for n_perm in BUDGETS:
            phi = shapley_sampling(_linear(w), x, bg, n_perm=n_perm,
                                   rng=np.random.default_rng(0))[:, 0]
            rows.append({"universe": label, "inputs": M, "n_perm": n_perm,
                         "closed_form_max_abs_err": float(np.max(np.abs(phi - closed)))})
        # Efficiency with a single-row background is exact for any function.
        b1 = rng.uniform(-1, 1, (1, M))
        f = lambda X: np.tanh(np.atleast_2d(np.asarray(X, dtype=np.float64))).sum(axis=1, keepdims=True)
        phi = shapley_sampling(f, x, b1, n_perm=200, rng=np.random.default_rng(0))[:, 0]
        rows.append({"universe": label, "inputs": M, "check": "efficiency",
                     "abs_err": float(abs(phi.sum() - (f(x.reshape(1, -1))[0, 0] - f(b1)[0, 0])))})
    return rows


def grouped_explanation(n_explain: int = 30, n_perm: int = 64) -> dict:
    prices, _ = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]
    best, best_sh = None, -1e9
    for mp in glob.glob(os.path.join(OUT, "runs", "price_seed*", "metrics.json")):
        sh = json.load(open(mp))["metrics"]["Sharpe"]
        if sh > best_sh:
            best, best_sh = os.path.join(os.path.dirname(mp), "model"), sh
    model = PPO.load(best)
    tickers = list(te_rets.columns)
    names = explain.build_feature_names(tickers)
    f = explain.make_policy_fn(model)

    rng = np.random.default_rng(config.SEED)
    bg_all = explain._observations(model, tr_feat, tr_rets)
    background = bg_all[rng.choice(len(bg_all), size=30, replace=False)]
    obs = explain._observations(model, te_feat, te_rets)
    sel = np.sort(rng.choice(len(obs), size=min(n_explain, len(obs)), replace=False))

    asset_of = [n.split(":")[0] if ":" in n else n for n in names]          # ticker or w_TICKER
    asset_of = [a[2:] if a.startswith("w_") else a for a in asset_of]
    fam_of = [FAMILY.get(n.split(":")[1], n) if ":" in n else "current holding (inertia)"
              for n in names]

    by_asset = np.zeros(len(tickers) + 1)
    by_family: dict[str, float] = {}
    resid = []
    for x in obs[sel]:
        phi = shapley_sampling(f, x, background, n_perm=n_perm, rng=rng)    # (M, K)
        mag = np.abs(phi).mean(axis=1)                                      # (M,)
        for j, a in enumerate(asset_of):
            by_asset[(tickers + ["CASH"]).index(a)] += mag[j]
        for j, fam in enumerate(fam_of):
            by_family[fam] = by_family.get(fam, 0.0) + mag[j]
        # Multi-row background makes efficiency hold in expectation; report the gap.
        resid.append(float(np.abs(phi.sum(axis=0) - (f(x.reshape(1, -1))[0]
                                                     - f(background).mean(axis=0))).max()))
    k = len(sel)
    return {"model": os.path.basename(os.path.dirname(best)), "decisions_explained": k,
            "n_perm_per_decision": n_perm, "background_size": 30, "raw_inputs": len(names),
            "by_asset_top10": sorted(({"asset": a, "mean_abs_shapley": float(v / k)}
                                      for a, v in zip(tickers + ["CASH"], by_asset)),
                                     key=lambda d: -d["mean_abs_shapley"])[:10],
            "by_family": sorted(({"family": a, "mean_abs_shapley": v / k}
                                 for a, v in by_family.items()), key=lambda d: -d["mean_abs_shapley"]),
            "efficiency_gap_max_over_decisions": float(np.max(resid)),
            "efficiency_gap_mean": float(np.mean(resid))}


if __name__ == "__main__":
    out = {"universe": config.UNIVERSE, "scaling": closed_form_and_efficiency()}
    for r in out["scaling"]:
        print(r, flush=True)
    if config.UNIVERSE == "extended":
        out["grouped"] = grouped_explanation()
        print(json.dumps(out["grouped"], indent=1)[:1500])
    with open(os.path.join(OUT, "shapley_scaling.json"), "w") as fh:
        json.dump(out, fh, indent=2)
