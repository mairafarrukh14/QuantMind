"""Central configuration for the QuantMind RL prototype.

Keeping every tunable in one place makes the experiments in the report
reproducible and easy to cite (e.g. "we used a 0.1% transaction cost").
"""
from __future__ import annotations

import os

# --- Project paths -----------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
for _d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Asset universe ----------------------------------------------------------
# A small, liquid, diversified slice of the S&P 500. Kept small (a) because the
# template/PPT notes RL "generalises poorly to small portfolios" -- so part of
# the prototype's job is to test exactly that, and (b) so the SHAP explanations
# stay legible for a non-technical user.
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]

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
MAX_WEIGHT = 0.40          # concentration limit: max fraction in any single asset
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
