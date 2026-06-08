"""Export the trained prototype's real results to a JS file the frontend reads.

Produces ../frontend/data.js as `window.QM_DATA = {...}` so the dashboard is
fully static (no server / no fetch / no CORS): just open index.html.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, backtest, explain, metrics, train  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(config.ROOT), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

ASSET_META = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    "JPM":  {"name": "JPMorgan Chase", "sector": "Financials"},
    "XOM":  {"name": "Exxon Mobil", "sector": "Energy"},
    "JNJ":  {"name": "Johnson & Johnson", "sector": "Healthcare"},
}


def downsample(index, *arrays, step):
    idx = list(range(0, len(index), step))
    if idx[-1] != len(index) - 1:
        idx.append(len(index) - 1)
    return [index[i] for i in idx], [[a[i] for i in idx] for a in arrays]


def main():
    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]
    tickers = list(te_rets.columns)

    model = train.load_agent()

    # --- Backtest series --------------------------------------------------
    agent_rets, weights = backtest.run_agent(model, te_feat, te_rets)
    bh = backtest.buy_and_hold(te_rets).reindex(agent_rets.index).fillna(0.0)
    best = backtest.best_single_asset(te_rets).reindex(agent_rets.index).fillna(0.0)
    best_name = best.name

    init = config.INITIAL_CASH
    eq_agent = metrics.equity_curve(agent_rets.values, init)
    eq_bh = metrics.equity_curve(bh.values, init)
    eq_best = metrics.equity_curve(best.values, init)
    dates = [d.strftime("%Y-%m-%d") for d in agent_rets.index]

    d_dates, (a, b, c) = downsample(dates, list(eq_agent), list(eq_bh), list(eq_best), step=3)

    # --- Allocation over time (downsampled stacked area) ------------------
    cols = list(weights.columns)  # assets + CASH
    w_dates_idx = [d.strftime("%Y-%m-%d") for d in weights.index]
    alloc_series = {col: weights[col].tolist() for col in cols}
    ad_dates, ad_vals = downsample(w_dates_idx, *[alloc_series[c] for c in cols], step=4)
    alloc_ds = {col: ad_vals[i] for i, col in enumerate(cols)}

    current_alloc = weights.iloc[-1].to_dict()

    # --- Metrics ----------------------------------------------------------
    table = metrics.summary_table(
        {agent_rets.name: agent_rets, bh.name: bh, best.name: best},
        rf_daily=config.RISK_FREE_DAILY,
    )
    metrics_dict = json.loads(table.to_json(orient="index"))

    # --- Explainability (recompute a small pass) --------------------------
    print("[export] computing Shapley attributions ...")
    phi_all, X_explain, sel, feat_names, _ = explain.explain(
        model, tr_feat, tr_rets, te_feat, te_rets,
        n_background=25, n_explain=30, n_perm=48,
    )
    importance = np.abs(phi_all).mean(axis=(0, 2))
    order = np.argsort(importance)[::-1][:12]
    feat_importance = [{"feature": feat_names[i], "value": float(importance[i])} for i in order]

    # Per-asset attribution for the LAST decision (for the recommendation cards).
    last_phi = phi_all[-1]
    last_obs = X_explain[-1]
    f = explain.make_policy_fn(model)
    last_weights = f(last_obs.reshape(1, -1))[0]

    def asset_drivers(asset_idx, asset_label):
        contribs = last_phi[:, asset_idx]
        own = [k for k, n in enumerate(feat_names) if n.startswith(f"{asset_label}:")]
        ranked = sorted(own, key=lambda k: abs(contribs[k]), reverse=True)[:3]
        return [{
            "feature": feat_names[k].split(":")[1],
            "phrase": explain.FEATURE_PHRASES.get(feat_names[k].split(":")[1], feat_names[k]),
            "direction": "up" if contribs[k] > 0 else "down",
            "magnitude": float(abs(contribs[k])),
        } for k in ranked]

    recommendations = []
    for i, tkr in enumerate(tickers):
        w = float(last_weights[i])
        action = ("Overweight" if w >= 0.30 else "Hold" if w >= 0.12 else "Underweight")
        recommendations.append({
            "ticker": tkr,
            "name": ASSET_META.get(tkr, {}).get("name", tkr),
            "sector": ASSET_META.get(tkr, {}).get("sector", ""),
            "weight": w,
            "action": action,
            "drivers": asset_drivers(i, tkr),
        })
    recommendations.sort(key=lambda r: r["weight"], reverse=True)

    # Natural-language rationale for the top pick.
    rationale = explain.narrate_decision(model, last_obs, tickers, last_phi, feat_names)

    avg_turnover = float(weights.diff().abs().sum(axis=1).mean())

    out = {
        "meta": {
            "data_source": source,
            "tickers": tickers,
            "asset_meta": ASSET_META,
            "test_start": dates[0],
            "test_end": dates[-1],
            "test_days": len(dates),
            "initial_cash": init,
            "max_weight": config.MAX_WEIGHT,
            "transaction_cost": config.TRANSACTION_COST,
            "timesteps": config.TOTAL_TIMESTEPS,
            "best_asset_name": best_name,
        },
        "kpis": {
            "portfolio_value": float(eq_agent[-1]),
            "total_return": metrics_dict[agent_rets.name]["Total Return"],
            "sharpe": metrics_dict[agent_rets.name]["Sharpe"],
            "max_drawdown": metrics_dict[agent_rets.name]["Max Drawdown"],
            "cagr": metrics_dict[agent_rets.name]["CAGR"],
            "volatility": metrics_dict[agent_rets.name]["Ann. Volatility"],
            "sortino": metrics_dict[agent_rets.name]["Sortino"],
            "avg_turnover": avg_turnover,
        },
        "equity": {"dates": d_dates, "agent": a, "buyhold": b, "best": c},
        "allocation_ts": {"dates": ad_dates, "series": alloc_ds, "columns": cols},
        "current_allocation": current_alloc,
        "metrics": metrics_dict,
        "feature_importance": feat_importance,
        "recommendations": recommendations,
        "rationale": rationale,
    }

    path = os.path.join(FRONTEND_DIR, "data.js")
    with open(path, "w") as fh:
        fh.write("window.QM_DATA = ")
        json.dump(out, fh, indent=2)
        fh.write(";\n")
    print(f"[export] wrote {path}")


if __name__ == "__main__":
    main()
