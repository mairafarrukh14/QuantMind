"""Central configuration for the QuantMind RL prototype.

Keeping every tunable in one place makes the experiments in the report
reproducible and easy to cite (e.g. "we used a 0.1% transaction cost").
"""
from __future__ import annotations

import os

# --- Project paths -----------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which asset universe a run uses. "core" is the original five-name universe and
# keeps every existing path. "extended" is the 30-name, all-sector universe and
# writes only to extended/ subdirectories, so the locked core artifacts are never
# touched. Select with the environment variable QM_UNIVERSE=extended.
UNIVERSE = os.environ.get("QM_UNIVERSE", "core")
if UNIVERSE not in ("core", "extended"):
    raise ValueError(f"QM_UNIVERSE must be 'core' or 'extended', got {UNIVERSE!r}")
_SUB = "" if UNIVERSE == "core" else "extended"
DATA_DIR = os.path.join(ROOT, "data", _SUB)
RESULTS_DIR = os.path.join(ROOT, "results", _SUB)
MODELS_DIR = os.path.join(ROOT, "models", _SUB)
for _d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Asset universe ----------------------------------------------------------
# A small, liquid, diversified slice of the S&P 500. Kept small (a) because the
# template/PPT notes RL "generalises poorly to small portfolios" -- so part of
# the prototype's job is to test exactly that, and (b) so the SHAP explanations
# stay legible for a non-technical user.
CORE_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]

# Extended universe: the five core names plus 24 further US large caps covering all
# eleven GICS sectors, plus one non-US equity ETF (iShares MSCI Japan). Every name
# has daily history from the start of the window.
EXTENDED_TICKERS = CORE_TICKERS + [
    "NVDA", "ORCL",                    # information technology
    "GOOGL", "VZ", "DIS",              # communication services
    "AMZN", "HD", "MCD", "NKE",        # consumer discretionary
    "PG", "KO", "WMT",                 # consumer staples
    "UNH", "PFE", "MRK",               # health care
    "BAC", "GS",                       # financials
    "CAT", "HON", "UPS",               # industrials
    "CVX",                             # energy
    "SHW",                             # materials
    "NEE",                             # utilities
    "AMT",                             # real estate
    "EWJ",                             # non-US: MSCI Japan ETF
]
TICKERS = CORE_TICKERS if UNIVERSE == "core" else EXTENDED_TICKERS

# Backtest window. Training on the bulk of the history, holding out the most
# recent stretch (incl. the 2022 drawdown + 2023-24 recovery) as an unseen test.
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
TRAIN_TEST_SPLIT = "2022-01-01"  # everything before -> train, on/after -> test

# --- Environment parameters --------------------------------------------------
TRANSACTION_COST = 0.001   # 0.1% proportional cost on turnover (realistic retail)
RISK_FREE_DAILY = 0.0      # daily risk-free rate used in reward shaping
RISK_PENALTY = 0.15        # weight on the rolling-volatility penalty in reward
TURNOVER_PENALTY = 0.02    # explicit churn penalty (beyond the trading cost)
# Concentration limit: twice the equal weight, so the rule is the same in both
# universes (2/5 = 40% for the core five, 2/30 = 6.67% for the extended thirty).
MAX_WEIGHT = 2.0 / len(TICKERS)
INITIAL_CASH = 100_000.0

# --- Training parameters -----------------------------------------------------
SEED = 42
TOTAL_TIMESTEPS = 150_000  # PPO training budget
EVAL_EPISODES = 1          # deterministic single pass over the test set

# Feature names per asset (order matters: used for SHAP labelling).
FEATURE_NAMES = [
    "ret_1d",      # 1-day log return
    "ret_5d",      # 5-day log return (momentum)
    "rsi_14",      # Relative Strength Index, scaled to [-1, 1]
    "macd_hist",   # MACD histogram, volatility-scaled
    "vol_20",      # 20-day realised volatility, scaled
    "px_sma20",    # price / 20-day SMA - 1 (mean-reversion signal)
    "sma5_sma20",  # 5-day SMA / 20-day SMA - 1 (trend signal)
]
