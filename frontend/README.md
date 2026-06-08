# QuantMind — Web Dashboard (frontend)

A premium, fully self-contained dashboard for the QuantMind explainable-AI
portfolio advisor. It visualises the **real output of the trained RL prototype**
(not mock data) and demonstrates the product's three-tier vision: data → AI/ML
engine → web interface.

## Open it

Just double-click **`index.html`** (or `open index.html`). No server, no build,
no `npm install` — everything is vendored or pure HTML/CSS/JS. Works offline.

> Deep links: `index.html#recommendations`, `#explainability`, `#assistant`,
> `#performance`.

## Views
- **Dashboard** — KPI cards (value, Sharpe, CAGR, drawdown), equity curve vs
  buy-and-hold / best asset, current allocation donut, allocation-over-time.
- **Recommendations** — today's holdings with action badges and an expandable
  *"Why"* showing the top Shapley drivers per asset, plus the NL rationale.
- **Explainability** — global feature-importance (mean |Shapley value|) and a
  plain explanation of the XAI method.
- **AI Assistant** — natural-language Q&A grounded in the real results
  (rule-based answers; the full system would route these through a local LLM).
- **Performance** — metrics table vs baselines, risk/return scatter, Sharpe &
  Sortino bars.

## Design
- Bespoke dark fintech design system (`styles.css`): glassmorphism, dual-accent
  brand (indigo "Mind" → teal "Quant"), tabular numerics, soft depth.
- Charts via Chart.js (vendored locally in `vendor/`, ~205 KB, offline).
- System font stack (SF Pro on macOS) — no web-font dependency.

## Data
`data.js` is generated from the trained model by:

```bash
cd ../quantmind_prototype
.venv/bin/python export_frontend_data.py   # writes ../frontend/data.js
```

Re-run that after retraining to refresh every number, chart and explanation.

## Note
This is the **front end of the planned full product**. The current preliminary
submission's graded prototype is the RL + XAI engine in `../quantmind_prototype`;
this dashboard is the presentation layer that makes its output legible to a
non-technical user (and makes a strong demo video). Real-time data, FinBERT
sentiment and a production LLM chat are final-project work.
