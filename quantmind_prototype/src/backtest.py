"""Backtest the trained agent on the held-out test period and compare baselines.

Baselines:
  * Buy & Hold (equal weight): invest 1/N in each asset on day 1, never rebalance.
  * Best single asset (ex-post): an optimistic reference, not investable.

Outputs: a metrics table (CSV), an equity-curve figure and a weight-allocation
figure, all written to results/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, metrics
from .env import PortfolioEnv


def run_agent(model, features, log_rets, deterministic=True):
    """Roll the policy through the env, recording daily returns and weights."""
    env = PortfolioEnv(features, log_rets)
    obs, _ = env.reset()
    rets, weights, dates = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rets.append(info["port_ret"])
        weights.append(info["target_weights"])
        dates.append(info["date"])
    idx = pd.DatetimeIndex(dates)
    weights = pd.DataFrame(weights, index=idx, columns=list(log_rets.columns) + ["CASH"])
    return pd.Series(rets, index=idx, name="QuantMind PPO"), weights


def buy_and_hold(log_rets) -> pd.Series:
    """Equal-weight, drifting buy & hold: invest 1/N on day one, never rebalance.

    Earns simple_rets[t] on each date t (same realised-return days as the agent),
    so the comparison is like-for-like once reindexed to the agent's dates.
    """
    simple = np.exp(log_rets.values) - 1.0
    n = simple.shape[1]
    w = np.ones(n) / n
    port_rets = np.empty(len(simple))
    for t in range(len(simple)):
        r = simple[t]
        port_rets[t] = float(np.dot(w, r))
        w = w * (1.0 + r)
        w = w / (w.sum() + 1e-12)
    return pd.Series(port_rets, index=log_rets.index, name="Buy & Hold (1/N)")


def best_single_asset(log_rets) -> pd.Series:
    simple = np.exp(log_rets) - 1.0
    totals = (1.0 + simple).prod()
    best = totals.idxmax()
    return simple[best].rename(f"Best asset ({best})")


def make_plots(strategies: dict[str, pd.Series], weights: pd.DataFrame, source: str):
    # --- Equity curves -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rets in strategies.items():
        eq = metrics.equity_curve(rets.values, initial=config.INITIAL_CASH)
        lw = 2.4 if "QuantMind" in name else 1.4
        ax.plot(rets.index, eq, label=name, linewidth=lw)
    ax.set_title(f"QuantMind backtest — portfolio value on held-out test period\n(data source: {source})")
    ax.set_ylabel("Portfolio value (£)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "equity_curve.png"), dpi=150)
    plt.close(fig)

    # --- Allocation over time ---------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns, alpha=0.85)
    ax.set_title("QuantMind PPO — portfolio allocation over time")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=len(weights.columns), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "allocation.png"), dpi=150)
    plt.close(fig)


def evaluate(model, test_features, test_rets, source="unknown"):
    agent_rets, weights = run_agent(model, test_features, test_rets)
    bh = buy_and_hold(test_rets).reindex(agent_rets.index).fillna(0.0)
    best = best_single_asset(test_rets).reindex(agent_rets.index).fillna(0.0)

    strategies = {agent_rets.name: agent_rets, bh.name: bh, best.name: best}
    table = metrics.summary_table(strategies, rf_daily=config.RISK_FREE_DAILY)
    table.to_csv(os.path.join(config.RESULTS_DIR, "metrics.csv"))

    extra = pd.Series({
        "Avg daily turnover": float(weights.diff().abs().sum(axis=1).mean()),
    })
    make_plots(strategies, weights, source)
    return table, weights, agent_rets, extra
