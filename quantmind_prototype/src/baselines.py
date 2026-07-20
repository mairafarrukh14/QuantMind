"""Reference strategies for the evaluation (Work Order W4).

Every strategy is run through the same cost-aware simulator as the agent -- 0.1%
on turnover, the same 40% concentration cap where it applies -- so the comparison
is like-for-like. The set spans a floor and a ceiling and the standard yardsticks:

  * equal-weight 1/N buy-and-hold        -- the hard baseline (DeMiguel et al.)
  * best single asset in hindsight        -- an un-investable upper reference
  * Markowitz mean-variance (train-fit)   -- the classical optimised alternative
  * 12-1 momentum, monthly rebalanced     -- a standard active rule
  * random-weight policy (mean of >=100)  -- the floor any skill must clear

The 1/N and best-asset baselines already live in ``backtest.py``; this module
adds the optimised, momentum and random references and the shared simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, metrics
from .env import apply_position_cap

TRADING_DAYS = metrics.TRADING_DAYS


# --------------------------------------------------------------------------- #
#  Shared cost-aware simulator                                                 #
# --------------------------------------------------------------------------- #
def simulate(target_weights: np.ndarray, simple_rets: np.ndarray,
             cost: float = config.TRANSACTION_COST) -> np.ndarray:
    """Daily net returns for a target-weight path.

    ``target_weights`` is (T, n_assets) over the risky assets (cash is the
    remainder); ``simple_rets`` is (T, n_assets) realised same-day returns. On each
    day we pay ``cost`` times the turnover from the drifted previous weights to the
    new target, then earn the target's return. Mirrors the agent's accounting.
    """
    T, n = target_weights.shape
    prev = np.zeros(n)  # start in cash
    out = np.empty(T)
    for t in range(T):
        tgt = target_weights[t]
        turnover = np.abs(tgt - prev).sum()
        r = float(np.dot(tgt, simple_rets[t]))
        out[t] = r - cost * turnover
        grown = tgt * (1.0 + simple_rets[t])
        s = grown.sum()
        prev = grown / s if s > 1e-12 else tgt
    return out


# --------------------------------------------------------------------------- #
#  Markowitz mean-variance (train-fit, long-only, capped)                      #
# --------------------------------------------------------------------------- #
def markowitz_weights(train_simple: np.ndarray, cap: float = config.MAX_WEIGHT) -> np.ndarray:
    """Long-only tangency (max-Sharpe) weights from the training window.

    Uses the unconstrained tangency solution w ~ Sigma^{-1} mu, then projects to
    long-only and applies the concentration cap -- a transparent, scipy-free
    approximation of the constrained optimiser adequate for a five-asset baseline.
    """
    mu = train_simple.mean(axis=0)
    cov = np.cov(train_simple, rowvar=False)
    n = len(mu)
    cov += 1e-6 * np.eye(n)  # ridge for a well-conditioned inverse
    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        raw = mu.copy()
    raw = np.clip(raw, 0, None)             # long-only
    if raw.sum() <= 1e-12:
        raw = np.ones(n)
    w = raw / raw.sum()
    capped = apply_position_cap(np.append(w, 0.0), cap, n)  # cap, cash slot spare
    return capped[:n] / capped[:n].sum()


def markowitz_series(train_simple: np.ndarray, test_simple: np.ndarray,
                     rebalance: int = 21) -> np.ndarray:
    """Hold the train-fit weights over the test window, rebalancing monthly."""
    w = markowitz_weights(train_simple)
    T, n = test_simple.shape
    targets = np.tile(w, (T, 1))
    # Rebalance monthly: between rebalances, let weights drift (handled by the
    # simulator's own drift), so we only re-assert the target every `rebalance` days.
    mask = np.zeros(T, dtype=bool)
    mask[::rebalance] = True
    # simulate() re-targets every day; to mimic monthly rebalancing we hold the
    # last target constant between rebalance days by forward-filling.
    for t in range(1, T):
        if not mask[t]:
            targets[t] = targets[t - 1]
    return simulate(targets, test_simple)


# --------------------------------------------------------------------------- #
#  12-1 momentum, monthly                                                      #
# --------------------------------------------------------------------------- #
def momentum_series(prices_test: pd.DataFrame, test_simple: np.ndarray,
                    rebalance: int = 21) -> np.ndarray:
    """Equal-weight the assets with positive 12-1 momentum, rebalanced monthly.

    12-1 momentum is the trailing 12-month return skipping the most recent month;
    with no positive names the strategy sits in cash.
    """
    px = prices_test.values
    T, n = test_simple.shape
    lookback, skip = 252, 21
    targets = np.zeros((T, n))
    current = np.zeros(n)
    for t in range(T):
        if t % rebalance == 0:
            if t - skip - 1 >= 0 and t - lookback >= 0:
                mom = px[t - skip] / px[t - lookback] - 1.0
            else:
                mom = np.zeros(n)
            winners = mom > 0
            current = (winners / winners.sum()) if winners.any() else np.zeros(n)
        targets[t] = current
    return simulate(targets, test_simple)


# --------------------------------------------------------------------------- #
#  Random-weight policy (the floor)                                            #
# --------------------------------------------------------------------------- #
def random_series(test_simple: np.ndarray, n_runs: int = 100,
                  cap: float = config.MAX_WEIGHT, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Mean daily returns across ``n_runs`` random-weight policies.

    Each run draws a fresh capped simplex every day; the average across runs is the
    floor a skilful policy must clear. Returns (mean_series, summary_of_run_sharpes).
    """
    T, n = test_simple.shape
    rng = np.random.default_rng(seed)
    all_series = np.empty((n_runs, T))
    sharpes = np.empty(n_runs)
    for k in range(n_runs):
        targets = np.empty((T, n))
        for t in range(T):
            w = rng.random(n + 1)
            w = w / w.sum()
            w = apply_position_cap(w, cap, n)
            targets[t] = w[:n]
        s = simulate(targets, test_simple)
        all_series[k] = s
        sharpes[k] = metrics.sharpe(s)
    mean_series = all_series.mean(axis=0)
    return mean_series, {"mean_sharpe": float(sharpes.mean()),
                         "std_sharpe": float(sharpes.std(ddof=1)), "n_runs": n_runs}
