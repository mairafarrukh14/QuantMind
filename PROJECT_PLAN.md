# QuantMind — Requirements & Build Plan

> Working reference for the CM3070 Final Project preliminary submission.
> Project: **QuantMind — An Explainable AI Financial Advisor Bot for Intelligent
> Active Portfolio Management.** Template **4.2 — Financial Advisor Bot**
> (originates from CM3020 Artificial Intelligence).

## 1. What the project is

A bot that analyses financial data and produces recommendations for a *dynamic /
active* investment strategy ("active portfolio management"), usable by a
non-technical user through a web interface, presenting analysis and
recommendations **with explanations**.

QuantMind's distinctive angle: fuse three ML streams that existing tools treat
in isolation —
* **RL** — a PPO agent for active allocation (decision-making under uncertainty);
* **NLP** — FinBERT news/sentiment signals fused into the strategy;
* **XAI** — SHAP / LIME feature attribution + a local LLaMA-3 (Ollama)
  natural-language rationale.

Three-tier architecture: **Data layer → AI/ML engine → Web interface (React)**.

## 2. This submission = the PRELIMINARY report (staff-graded)

Four chapters, **6000-word total cap** (per-chapter maxes are strict; the
chapter maxes sum to 7000 to allow balancing):

| # | Chapter | Max words | Status |
|---|---------|-----------|--------|
| 1 | Introduction (concept + motivation + **must state Template 4.2**) | 1000 | to write (later) |
| 2 | Literature Review | 2500 | to write (later) |
| 3 | Design | 2000 | to write (later) |
| 4 | **Feature Prototype** (only *new* element) + evaluation + improvements | 1500 | **prototype built; chapter to write** |

Plus: a **3–5 minute MP4 video** demonstrating the prototype (user records it; a
script will be provided). LaTeX write-up deferred per user (do later).

### Source mapping (PPT → chapters)
* Slide 2 "The Problem" → Introduction / motivation.
* Slides 3–6 Related Work (Robo-advisors; Deep RL; FinBERT/LLM; XAI & trust) →
  Literature Review (4–6 pieces — hits Rubric 2/3 "4–6 pieces" band).
* Slides 7–8 (gap table + 3-tier architecture) + Slide 9 (aims & eval) → Design.
* This prototype + its evaluation → Chapter 4.

## 3. Marking criteria to optimise for
Instructions list 14 criteria; rubric weights that matter most for us:
* **Rubric 11 — prototype technically challenging (up to 8 pts):** custom RL env
  + PPO + SHAP-on-policy is genuinely advanced → target the 6–8 band.
* **Rubric 12 / criterion 13 — evaluate prototype + improvements (up to 3 pts):**
  use *ML/finance-appropriate* evaluation (Sharpe, drawdown, baseline
  comparison, turnover) and a concrete improvement plan.
* Rubrics 2–4 (literature + citations), 5–7 (design quality), 8–10 (workplan +
  evaluation strategy) — covered by the three earlier chapters.
* Criterion 12 — effective video demo.

## 4. Feature prototype (DECIDED scope: RL agent + backtest + XAI)
Implemented in `quantmind_prototype/` (see its README). Deliberately *one*
feature of the full design — the hardest one — to prove feasibility. Excludes
FinBERT fusion, LLaMA narration and the React UI (those are final-project scope;
noted as "future work / improvements" in the chapter).

**Pipeline:** yfinance data (cached, GBM fallback) → engineered technical
features → custom Gymnasium `PortfolioEnv` → PPO (Stable-Baselines3) →
backtest vs equal-weight buy & hold → SHAP attribution → NL rationale.

**Evaluation plan for the chapter:**
* Quantitative: Sharpe, Sortino, CAGR, max drawdown, total return, turnover, on
  a held-out test period; vs buy-and-hold and best-single-asset references.
* Honest critique: RL on small portfolios / limited data, overfitting risk,
  transaction-cost sensitivity, single-seed variance.
* Improvements: walk-forward validation, multiple seeds, FinBERT sentiment in
  the state, distributional/risk-aware RL, the explainability→trust user study.

## 5. Build order
1. ✅ Confirm project + requirements (this doc).
2. ✅ Build + run the RL/XAI prototype, generate result artefacts.
3. ⏳ Write Chapter 4 (Feature Prototype) grounded in real results.
4. ⏳ Write Chapters 1–3 from the PPT.
5. ⏳ Assemble LaTeX report (deferred until user is ready) + `.bib`.
6. ⏳ Provide a video demo script.

## 5a. LOCKED prototype results (real S&P 500 data, held-out 2022-01→2024-12)
Trained PPO 150k steps; reward = log-return − 0.15·vol − 0.02·turnover; 40% per-asset
concentration cap; 0.1% transaction cost. Test = 752 unseen days. **No lookahead**
(decision at close t earns t→t+1 return; verified by fixing an initial leak that
gave a fake Sharpe 8.5).

| Strategy | Total Ret | CAGR | Vol | Sharpe | Sortino | Max DD |
|---|---|---|---|---|---|---|
| **QuantMind PPO** | 49.4% | 14.4% | 16.2% | **0.91** | 1.34 | −13.8% |
| Buy & Hold (1/N) | 42.0% | 12.5% | 16.0% | 0.82 | 1.20 | −12.8% |
| Best asset XOM (ex-post) | 85.1% | 22.9% | 27.2% | 0.89 | 1.35 | −20.5% |

Turnover ≈ 4.8%/day. Agent beats buy-and-hold on Sharpe/Sortino/CAGR/return at
comparable risk, and edges best-single-asset Sharpe with far lower drawdown.
Honest gaps for the evaluation chapter: below the aspirational Sharpe-1.2 PPT
target; single universe / single test period; single seed; concentration in
defensive/energy names. SHAP shows allocation driven by position inertia, then
MACD/RSI. Artefacts: `quantmind_prototype/results/` (equity_curve.png,
allocation.png, shap_importance.png, metrics.csv, run_summary.json, rationale.txt).

## 6. Environment notes
* Python 3.12 venv in `quantmind_prototype/.venv`.
* No LaTeX installed yet (deferred).
* Yahoo Finance rate-limits (HTTP 429) — hence caching + GBM fallback.
