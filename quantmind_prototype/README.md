# QuantMind — Explainable AI Financial Advisor

CM3070 Final Year Project, Template 4.2 (*Financial Advisor Bot*). QuantMind is a
reinforcement-learning portfolio advisor that runs an active strategy, attaches a
faithful, audited explanation to every decision, and serves it through a web
dashboard for a non-technical user. Everything below reproduces from a fresh clone;
all reported numbers are read from locked artifacts in `results/`.

## Pipeline

```
prices (yfinance, cached) + FinBERT news sentiment
   -> PortfolioEnv (Gymnasium)  -> PPO agent (Stable-Baselines3)   [train once, lock]
   -> Shapley attribution (from scratch)  -> audited LLM rationale (Ollama)
   -> evaluation (baselines, bootstrap CI, deflated Sharpe, deletion test)
   -> FastAPI backend + static dashboard
```

Key modules (`src/`): `data.py` (prices + 7 technical features), `env.py`
(environment + reward, no-look-ahead), `train.py` (PPO), `explain.py` (Shapley
sampling), `sentiment.py` (FinBERT, pre-close leakage filter), `audit.py` +
`rationale.py` (rephrase-only LLM + automatic audit), `baselines.py` + `stats.py`
(evaluation), `study_log.py` (user study).

## Setup

```bash
cd quantmind_prototype
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional (for the sentiment and LLM components — not needed to reproduce
training, evaluation, the tests or the dashboard, which read the already-locked
`data/news/sentiment_cache.csv` and `results/`):
```bash
pip install -r requirements-optional.txt
```
- **Sentiment**: needs the FNSPID news dataset. `python fetch_news.py` downloads and
  filters it (a Hugging Face token in `HF_TOKEN` avoids rate limits), then
  `python build_sentiment.py` scores it with FinBERT.
- **LLM rationale**: install [Ollama](https://ollama.com), then `ollama pull llama3.2:1b`.

## Reproduce every result (from locked artifacts)

```bash
python -m pytest                     # W1: 8 property tests (pass table -> results/)
python run_experiments_w3.py         # W3: 10 locked runs (5 seeds x sentiment off/on)
python run_experiments_placebo.py    # CO-17: 5 seeds, sentiment date-shuffled (control)
python run_evaluation.py             # W4: main table, baselines, bootstrap CI, DSR
python run_faithfulness.py           # W5: explanation deletion test + figure
python run_walkforward.py            # Table 5: three-window walk-forward retraining
python run_audit_eval.py             # W6: LLM audit pass-rate (needs Ollama)
python run_reconcile.py              # W8: one source of truth, zero mismatches
python export_frontend_data.py       # build the locked dashboard payload + data.js
```

Training is deterministic (fixed seeds, cached data, pinned versions); nothing is
retrained after `run_experiments_w3.py`, and every table/figure/endpoint reads the
same locked files.

## Dashboard

Two equivalent modes, both reading the one locked payload. Quote paths that
contain spaces or parentheses (Windows and some macOS setups).

```bash
# 1. Live API + dashboard
uvicorn serve:app --port 8000        # then open http://localhost:8000/

# 2. No server at all (static fallback) -- cross-platform
cd ../frontend && python -m http.server 8080   # then open http://localhost:8080/
# macOS shortcut:   open "../frontend/index.html"
# Windows shortcut: start "" "..\frontend\index.html"
```

## Tests

```bash
python -m pytest                     # runs tests/test_properties.py
```

The eight properties: no look-ahead, valid simplex, cap enforced, exact cost,
Shapley closed-form, Shapley efficiency, sentiment pre-close no-leakage, and audit
rejects corrupted rationales. All pass on a fresh clone.

## Configuration
All tunables live in `src/config.py` (asset universe, dates, costs, PPO budget).
