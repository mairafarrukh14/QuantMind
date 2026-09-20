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
python -m pytest                     # test suite: 8 property tests + recommendation and study-analysis tests
python run_experiments_w3.py         # 10 locked runs (5 seeds x sentiment off/on)
python run_experiments_placebo.py    # 5 seeds, sentiment dates shuffled (control)
python run_evaluation.py             # main table, baselines, bootstrap CI, deflated Sharpe
python run_regimes_costs.py          # per-year regime split and transaction-cost sweep
python run_faithfulness.py           # explanation deletion test + figure
python run_walkforward.py            # three-window walk-forward retraining
python run_audit_eval.py             # LLM audit pass-rate (needs Ollama)
python run_reconcile.py              # one source of truth, zero mismatches
python export_frontend_data.py       # build the locked dashboard payload + data.js
python run_study_arms.py             # user-study arm comparison (needs the local response files, which are not published)
python run_persona_survey.py         # persona-survey summary (needs the local survey file, not published)
python run_scaling_check.py          # Shapley error by input dimension (QM_UNIVERSE=extended adds grouped attribution)
python run_faithfulness_comparison.py  # deletion area with interval, against a from-scratch LIME
python run_inventory.py              # every trained run and the deflated-Sharpe trial count
python run_latency.py                # recommendation latency and test-suite time
python run_contrast_check.py         # WCAG contrast of the dashboard colours
python run_report_numbers.py         # report numbers, generated from the locked artifacts
python run_report_tables.py          # report tables, generated from the locked artifacts
```

### Extended universe

Set `QM_UNIVERSE=extended` to run the same scripts on the 30-asset, all-sector
universe (29 US large caps plus one Japan ETF, with a 2/N concentration cap). Its
data and results live under `data/extended/` and `results/extended/`, so the core
five-asset artifacts are never touched:

```bash
QM_UNIVERSE=extended python run_experiments_w3.py
QM_UNIVERSE=extended python run_walkforward.py
QM_UNIVERSE=extended python run_evaluation.py
QM_UNIVERSE=extended python run_regimes_costs.py
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

### Build a portfolio (live advice)

With the server running, the **Build a portfolio** screen takes an amount and an
as-of date and returns a suggested split, computed at request time from the locked
best-seed policy (nothing is trained). It starts from an all-cash portfolio and
flags dates that fall inside the training period. The same path is available as
`POST /api/recommend`; `GET /api/recommend/range` gives the dates it accepts.

### User-study builds

`index.html` is build A (explanations shown). `index.html?build=B` is build B, the
same dashboard with every explanation surface removed, used for the study's
comparison arm. `study/arm_b_session_guide.md` describes running a session.

## Tests

```bash
python -m pytest                     # runs tests/test_properties.py
```

The eight properties: no look-ahead, valid simplex, cap enforced, exact cost,
Shapley closed-form, Shapley efficiency, sentiment pre-close no-leakage, and audit
rejects corrupted rationales. All pass on a fresh clone.

## Configuration
All tunables live in `src/config.py` (asset universe, dates, costs, PPO budget).
