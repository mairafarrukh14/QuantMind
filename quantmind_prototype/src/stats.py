"""Honest-evaluation statistics (Work Order W4, contribution C4).

Two tools that turn a single backtest number into a claim that survives scrutiny:

* a **bootstrap** confidence interval on the Sharpe difference against a baseline
  (Efron and Tibshirani, 1993), which needs no assumption that returns are normal;
* the **Deflated Sharpe Ratio** (Bailey and Lopez de Prado, 2014), which discounts
  the best Sharpe for the number of trials run and for non-normal returns, so a
  best-of-ten result is not mistaken for a single confident one.

Implemented from scratch in NumPy (no scipy), consistent with the rest of the
project: the normal CDF uses ``math.erf`` and the inverse-normal uses Acklam's
rational approximation.
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics

TRADING_DAYS = metrics.TRADING_DAYS
_EULER = 0.5772156649015329


# --------------------------------------------------------------------------- #
#  Normal CDF / inverse-CDF (no scipy)                                         #
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _skew(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    m = r.mean()
    s = r.std()
    return float(np.mean(((r - m) / s) ** 3)) if s > 0 else 0.0


def _kurtosis(r: np.ndarray) -> float:
    """Non-excess (Pearson) kurtosis."""
    r = np.asarray(r, float)
    m = r.mean()
    s = r.std()
    return float(np.mean(((r - m) / s) ** 4)) if s > 0 else 3.0


# --------------------------------------------------------------------------- #
#  Bootstrap CI on the Sharpe difference                                       #
# --------------------------------------------------------------------------- #
def bootstrap_sharpe_diff_ci(agent: np.ndarray, baseline: np.ndarray,
                             n_boot: int = 10_000, alpha: float = 0.05,
                             seed: int = 0) -> dict:
    """Bootstrap CI for Sharpe(agent) - Sharpe(baseline).

    Resamples the daily-return pairs with replacement (paired, so the market-wide
    days line up), recomputing the Sharpe difference each time.
    """
    agent = np.asarray(agent, float)
    baseline = np.asarray(baseline, float)
    n = min(len(agent), len(baseline))
    agent, baseline = agent[:n], baseline[:n]
    rng = np.random.default_rng(seed)

    point = metrics.sharpe(agent) - metrics.sharpe(baseline)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metrics.sharpe(agent[idx]) - metrics.sharpe(baseline[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe_diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "prob_positive": float(np.mean(diffs > 0)), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
#  Deflated Sharpe Ratio                                                        #
# --------------------------------------------------------------------------- #
def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int,
                          all_trial_sharpes: np.ndarray | None = None) -> dict:
    """Deflated Sharpe Ratio of one (best) track record (Bailey and Lopez de Prado
    2014): the observed Sharpe is judged against the maximum an analyst would expect
    from ``n_trials`` random strategies, adjusted for skew, kurtosis and length."""
    r = np.asarray(returns, float)
    T = len(r)
    sr = metrics.sharpe(r) / np.sqrt(TRADING_DAYS)  # per-period (undo annualisation)

    if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
        var_sr = float(np.var(np.asarray(all_trial_sharpes, float) / np.sqrt(TRADING_DAYS),
                              ddof=1))
    else:
        var_sr = 1.0 / T  # asymptotic variance of the SR estimator under the null
    e_max = math.sqrt(var_sr) * ((1 - _EULER) * _norm_ppf(1 - 1.0 / n_trials)
                                 + _EULER * _norm_ppf(1 - 1.0 / (n_trials * math.e)))

    skew = _skew(r) if T > 2 else 0.0
    kurt = _kurtosis(r) if T > 3 else 3.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    dsr = _norm_cdf(((sr - e_max) * math.sqrt(T - 1)) / denom)
    return {"sharpe_ann": float(sr * np.sqrt(TRADING_DAYS)),
            "expected_max_sharpe_ann": float(e_max * np.sqrt(TRADING_DAYS)),
            "n_trials": int(n_trials), "deflated_sharpe": float(dsr)}
