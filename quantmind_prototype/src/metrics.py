"""Financial performance metrics for evaluating the agent vs baselines.

These are the standard risk/return statistics used to judge an active strategy
(the same ones named in the PPT aims): CAGR, annualised volatility, Sharpe,
Sortino, and maximum drawdown. Implemented directly so the report can state the
exact formulae used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    return initial * np.cumprod(1.0 + np.asarray(returns, dtype=float))


def cagr(returns: np.ndarray) -> float:
    eq = equity_curve(returns)
    years = len(returns) / TRADING_DAYS
    if years <= 0 or eq[-1] <= 0:
        return 0.0
    return float(eq[-1] ** (1.0 / years) - 1.0)


def annual_vol(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    excess = np.asarray(returns, dtype=float) - rf_daily
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    excess = np.asarray(returns, dtype=float) - rf_daily
    downside = excess[excess < 0]
    dd = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def summary(returns: np.ndarray, rf_daily: float = 0.0) -> dict:
    returns = np.asarray(returns, dtype=float)
    eq = equity_curve(returns)
    return {
        "Total Return": float(eq[-1] - 1.0),
        "CAGR": cagr(returns),
        "Ann. Volatility": annual_vol(returns),
        "Sharpe": sharpe(returns, rf_daily),
        "Sortino": sortino(returns, rf_daily),
        "Max Drawdown": max_drawdown(returns),
    }


def summary_table(strategies: dict[str, np.ndarray], rf_daily: float = 0.0) -> pd.DataFrame:
    """strategies: {name: daily_returns} -> tidy metrics DataFrame."""
    rows = {name: summary(rets, rf_daily) for name, rets in strategies.items()}
    df = pd.DataFrame(rows).T
    return df
