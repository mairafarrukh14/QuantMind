# QuantMind — RL Feature Prototype

Feature prototype for the CM3070 preliminary report (Template 4.2, *Financial
Advisor Bot*). It implements the single most important and most technically
challenging feature of QuantMind: a **reinforcement-learning active
portfolio-management agent** with an **explainability (XAI) layer**, evaluated
against a buy-and-hold baseline.

This is intentionally *one* feature of the full QuantMind design (RL + FinBERT
sentiment + XAI + LLaMA-3 narration + React UI). The prototype proves the
hardest part — the RL decision engine and its explanations — is feasible.

## What it does

```
prices (yfinance, cached)  ->  technical features  ->  PortfolioEnv (Gymnasium)
        ->  PPO agent (Stable-Baselines3)  ->  backtest vs Buy&Hold
        ->  SHAP feature attribution  ->  plain-English rationale
```

* **Data** (`src/data.py`): daily adjusted close for 5 diversified S&P 500
  names (AAPL, MSFT, JPM, XOM, JNJ), 2015–2024, cached to `data/prices.csv`.
  Falls back to a correlated GBM simulator if Yahoo is unreachable (flagged in
  output) so the demo always runs.
* **Environment** (`src/env.py`): a custom Gymnasium env framing active
  portfolio management as decision-making under uncertainty (Jiang et al. 2017
  style). Continuous action = target weight vector over assets + cash; reward =
  log return net of transaction costs minus a volatility penalty.
* **Agent** (`src/train.py`): PPO (Schulman et al. 2017) via Stable-Baselines3.
* **Backtest** (`src/backtest.py`): held-out test period (2022-01 onward).
  Metrics: total return, CAGR, volatility, Sharpe, Sortino, max drawdown,
  turnover. Baselines: equal-weight buy & hold, best ex-post single asset.
* **XAI** (`src/explain.py`): **Shapley-value sampling** (Strumbelj &
  Kononenko, 2014 — the estimator underlying SHAP) implemented from scratch in
  NumPy, treating the policy as `obs -> weights`. Attributes each allocation to
  its drivers; produces a global importance chart and a per-decision
  natural-language rationale. Implemented directly (no `shap`/`numba`
  dependency) to stay lightweight and keep the algorithm transparent.

## Run

```bash
cd quantmind_prototype
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_prototype.py            # full run (~200k PPO steps + Shapley XAI)
python run_prototype.py --quick    # fast smoke test
python run_prototype.py --no-shap  # skip the (slower) attribution stage
```

Artefacts written to `results/`:
`equity_curve.png`, `allocation.png`, `shap_importance.png`,
`metrics.csv`, `rationale.txt`, `run_summary.json`.

## Reproducibility
Single seed (`config.SEED = 42`) across NumPy, the env and PPO. All tunables
live in `src/config.py`.
