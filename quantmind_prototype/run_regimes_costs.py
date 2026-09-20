"""Regime split and transaction-cost sensitivity, from locked artifacts.

Two questions a single aggregate hides:

  * Regimes. Does the agent's result hold in each sub-period of the held-out
    window, or is it carried by one? The window is split by calendar year, fixed
    before looking at results: 2022 (drawdown), 2023 (recovery), 2024 (trend).
    Agent rows use the locked per-seed return series; baselines are recomputed on
    the same aligned window.
  * Costs. The agent trained at a 0.1% cost. Its locked models are replayed under
    0, 0.05%, 0.1%, 0.25% and 0.5% to see how quickly the edge (or its absence)
    changes with friction. Buy-and-hold never rebalances, so it pays no cost.

No model is retrained. Works for either universe (QM_UNIVERSE=core|extended).

Writes results/regimes.json and results/cost_sweep.json.
Run:  python run_regimes_costs.py
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

from src import backtest, baselines, config, data, metrics
from src.env import PortfolioEnv

OUT = config.RESULTS_DIR
RUNS_DIR = os.path.join(OUT, "runs")
REGIMES = [("2022 drawdown", "2022-01-01", "2023-01-01"),
           ("2023 recovery", "2023-01-01", "2024-01-01"),
           ("2024 trend", "2024-01-01", "2025-01-01")]
COSTS = [0.0, 0.0005, 0.001, 0.0025, 0.005]


def _sharpe(x):
    return float(metrics.sharpe(np.asarray(x), config.RISK_FREE_DAILY))


def _roll(model, features, log_rets, cost):
    env = PortfolioEnv(features, log_rets, cost=cost)
    obs, _ = env.reset()
    rets, turns, done = [], [], False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        rets.append(info["port_ret"])
        turns.append(info["turnover"])
    return np.array(rets), float(np.mean(turns))


def main() -> int:
    prices, _ = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    _, tr_rets, _ = split["train"]
    te_feat, te_rets_full, te_prices_full = split["test"]
    te_rets = backtest.align_to_tradeable(te_rets_full)
    te_prices = te_prices_full.iloc[1:]
    test_simple, train_simple = np.exp(te_rets.values) - 1.0, np.exp(tr_rets.values) - 1.0
    idx = te_rets.index

    price_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "price_seed*")))
    if not price_dirs:
        raise SystemExit("no locked price runs found; run run_experiments_w3.py first")

    seed_rets = {os.path.basename(d): pd.read_csv(os.path.join(d, "returns.csv"),
                                                  index_col=0, parse_dates=True)["port_ret"]
                 for d in price_dirs}
    baseline_rets = {
        "Buy & Hold (1/N)": backtest.buy_and_hold(te_rets).values,
        "Markowitz (train-fit)": baselines.markowitz_series(train_simple, test_simple),
        "12-1 momentum": baselines.momentum_series(te_prices, test_simple),
    }

    # ---- regimes ----------------------------------------------------------
    regimes = []
    for label, lo, hi in REGIMES:
        m = np.asarray((idx >= pd.Timestamp(lo)) & (idx < pd.Timestamp(hi)))
        seed_sh = np.array([_sharpe(s.reindex(idx).values[m]) for s in seed_rets.values()])
        row = {"regime": label, "days": int(m.sum()),
               "agent_sharpe_mean": float(seed_sh.mean()),
               "agent_sharpe_sd": float(seed_sh.std(ddof=1)),
               "agent_sharpes": seed_sh.tolist()}
        for name, r in baseline_rets.items():
            row[name] = _sharpe(np.asarray(r)[m])
        row["seeds_beating_1N"] = int((seed_sh > row["Buy & Hold (1/N)"]).sum())
        row["agent_total_return_mean"] = float(np.mean(
            [np.prod(1 + s.reindex(idx).values[m]) - 1 for s in seed_rets.values()]))
        row["1N_total_return"] = float(np.prod(1 + baseline_rets["Buy & Hold (1/N)"][m]) - 1)
        regimes.append(row)
    with open(os.path.join(OUT, "regimes.json"), "w") as fh:
        json.dump({"universe": config.UNIVERSE, "n_assets": len(config.TICKERS),
                   "n_seeds": len(seed_rets), "regimes": regimes}, fh, indent=2)

    # ---- cost sweep -------------------------------------------------------
    sweep = []
    bh_sharpe = _sharpe(baseline_rets["Buy & Hold (1/N)"])
    models = {os.path.basename(d): PPO.load(os.path.join(d, "model")) for d in price_dirs}
    for c in COSTS:
        sh, tu = [], []
        for model in models.values():
            r, t = _roll(model, te_feat, te_rets_full, c)
            sh.append(_sharpe(r)); tu.append(t)
        sh = np.array(sh)
        sweep.append({"cost": c, "agent_sharpe_mean": float(sh.mean()),
                      "agent_sharpe_sd": float(sh.std(ddof=1)),
                      "seeds_beating_1N": int((sh > bh_sharpe).sum()),
                      "mean_daily_turnover": float(np.mean(tu))})
        print(f"cost {c:.4f}: Sharpe {sh.mean():.3f} +/- {sh.std(ddof=1):.3f}", flush=True)
    with open(os.path.join(OUT, "cost_sweep.json"), "w") as fh:
        json.dump({"universe": config.UNIVERSE, "buy_hold_sharpe": bh_sharpe,
                   "trained_at_cost": config.TRANSACTION_COST, "sweep": sweep}, fh, indent=2)
    for r in regimes:
        print(f"{r['regime']}: agent {r['agent_sharpe_mean']:.2f} +/- {r['agent_sharpe_sd']:.2f}  "
              f"1/N {r['Buy & Hold (1/N)']:.2f}  beating {r['seeds_beating_1N']}/{len(seed_rets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
