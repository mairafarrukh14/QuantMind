"""W4 - extended evaluation suite: populates most of the evaluation chapter.

Reads the locked W3 runs and the cached prices, then emits, all from disk with
no retraining:

  * results/eval_main_table.csv  -- every strategy row: PPO mean +- sd and best
    seed (per variant), 1/N, Markowitz, 12-1 momentum, random floor, best asset.
  * results/eval_stats.json      -- bootstrap 95% CI on the Sharpe difference vs
    1/N (per seed and pooled) and the Deflated Sharpe Ratio of the best seed.
  * results/evaluation.json      -- the combined machine-readable summary the
    report and dashboard read.

Run (after run_experiments_w3.py):  python run_evaluation.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src import backtest, baselines, config, data, metrics, stats

OUT = config.RESULTS_DIR
RUNS_DIR = os.path.join(OUT, "runs")
COLS = ["Total Return", "CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown"]


def _load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "metrics.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def _variant_rows(runs, variant):
    """mean +- sd and best-seed metric dicts for one variant."""
    rows = [r for r in runs if r["variant"] == variant]
    if not rows:
        return None
    df = pd.DataFrame([{**r["metrics"], "seed": r["seed"]} for r in rows]).set_index("seed")
    best_seed = df["Sharpe"].idxmax()
    mean = {c: (float(df[c].mean()), float(df[c].std(ddof=1))) for c in COLS + ["Turnover"]}
    best = {c: float(df.loc[best_seed, c]) for c in COLS + ["Turnover"]}
    return {"df": df, "best_seed": int(best_seed), "mean": mean, "best": best}


def _metric_dict(rets):
    m = metrics.summary(rets, rf_daily=config.RISK_FREE_DAILY)
    return {c: m[c] for c in COLS}


def main() -> int:
    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    _, tr_rets, _ = split["train"]
    te_feat, te_rets_full, te_prices_full = split["test"]

    # See backtest.align_to_tradeable: the agent only earns returns from day 1 of
    # the window onward, so every baseline is recomputed fresh on that same
    # window -- otherwise a baseline earning day 0's return (+2 to +4% across
    # most assets) would be compared against an agent that structurally cannot.
    te_rets = backtest.align_to_tradeable(te_rets_full)
    te_prices = te_prices_full.iloc[1:]
    test_simple = np.exp(te_rets.values) - 1.0
    train_simple = np.exp(tr_rets.values) - 1.0
    print(f"source={source}  test days={len(te_rets)} (aligned; dropped day 0, "
          f"which the agent cannot trade on)", flush=True)

    runs = _load_runs()
    variants = sorted({r["variant"] for r in runs})
    print(f"loaded {len(runs)} locked runs: variants={variants}", flush=True)

    # --- baselines (same 0.1% cost via the shared simulator / buy&hold), all on
    # the aligned 751-day window ------------------------------------------------
    bh = backtest.buy_and_hold(te_rets)
    best_asset = backtest.best_single_asset(te_rets)
    mk = baselines.markowitz_series(train_simple, test_simple)
    mo = baselines.momentum_series(te_prices, test_simple)
    rnd, rnd_info = baselines.random_series(test_simple, n_runs=100)

    table = {}
    for variant in variants:
        vr = _variant_rows(runs, variant)
        label = "PPO" if variant == "price" else "PPO+sentiment"
        table[f"{label} (mean)"] = {c: vr["mean"][c][0] for c in COLS}
        table[f"{label} (mean)"]["_sd"] = {c: vr["mean"][c][1] for c in COLS}
        table[f"{label} (best seed {vr['best_seed']})"] = vr["best"]
    table["Buy & Hold (1/N)"] = _metric_dict(bh.values)
    table["Markowitz (train-fit)"] = _metric_dict(mk)
    table["12-1 momentum"] = _metric_dict(mo)
    table["Random policy (mean of 100)"] = _metric_dict(rnd)
    table["Best asset (hindsight)"] = _metric_dict(best_asset.values)

    pd.DataFrame({k: {c: v.get(c) for c in COLS} for k, v in table.items()}).T.to_csv(
        os.path.join(OUT, "eval_main_table.csv"))

    # --- statistics: bootstrap vs 1/N + deflated Sharpe ----------------------
    # bh is already on the aligned 751-day window, matching every locked run's
    # returns.csv (which is indexed the same way by construction), so no further
    # reindexing is needed -- and none of these comparisons mixes day-0 in for
    # one side only.
    per_seed = {}
    price_runs = [r for r in runs if r["variant"] == "price"]
    all_sharpes = [r["metrics"]["Sharpe"] for r in runs]
    best_run = max(price_runs, key=lambda r: r["metrics"]["Sharpe"])
    for r in price_runs:
        ret_path = os.path.join(RUNS_DIR, f"{r['variant']}_seed{r['seed']}", "returns.csv")
        agent = pd.read_csv(ret_path, index_col=0)["port_ret"].values
        per_seed[r["seed"]] = stats.bootstrap_sharpe_diff_ci(agent, bh.values, n_boot=10_000)

    # pooled: stack all price-seed returns against 1/N repeated, day-for-day.
    pooled_agent = np.concatenate([
        pd.read_csv(os.path.join(RUNS_DIR, f"price_seed{r['seed']}", "returns.csv"),
                    index_col=0)["port_ret"].values for r in price_runs])
    pooled_bh = np.concatenate([bh.values for _ in price_runs])
    pooled = stats.bootstrap_sharpe_diff_ci(pooled_agent, pooled_bh, n_boot=10_000)

    # Same pooled comparison for the sentiment variant, since with real coverage
    # its mean now sits above 1/N and that claim needs the same statistical rigor
    # as the price-only one, not just a headline number.
    sentiment_runs = [r for r in runs if r["variant"] == "sentiment"]
    pooled_sentiment = None
    if sentiment_runs:
        pooled_sent_agent = np.concatenate([
            pd.read_csv(os.path.join(RUNS_DIR, f"sentiment_seed{r['seed']}", "returns.csv"),
                        index_col=0)["port_ret"].values for r in sentiment_runs])
        pooled_sent_bh = np.concatenate([bh.values for _ in sentiment_runs])
        pooled_sentiment = stats.bootstrap_sharpe_diff_ci(pooled_sent_agent, pooled_sent_bh,
                                                           n_boot=10_000)

    best_returns = pd.read_csv(
        os.path.join(RUNS_DIR, f"price_seed{best_run['seed']}", "returns.csv"),
        index_col=0)["port_ret"].values
    dsr = stats.deflated_sharpe_ratio(best_returns, n_trials=len(runs),
                                      all_trial_sharpes=np.array(all_sharpes))

    stats_out = {"per_seed_vs_1N": per_seed, "pooled_vs_1N": pooled,
                 "pooled_sentiment_vs_1N": pooled_sentiment,
                 "deflated_sharpe_best_seed": dsr, "random_floor": rnd_info,
                 "n_trials": len(runs)}
    with open(os.path.join(OUT, "eval_stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    with open(os.path.join(OUT, "evaluation.json"), "w") as fh:
        json.dump({"source": source, "test_days": len(te_rets), "variants": variants,
                   "main_table": table, "stats": stats_out}, fh, indent=2)

    # --- console summary -----------------------------------------------------
    print("\nMain results (Sharpe):")
    for name, m in table.items():
        print(f"  {name:32s} Sharpe={m.get('Sharpe', float('nan')):.3f}  "
              f"TotRet={m.get('Total Return', float('nan')):.3f}")
    print(f"\nPooled PPO vs 1/N Sharpe diff: {pooled['sharpe_diff']:.3f} "
          f"[{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}]  "
          f"P(>0)={pooled['prob_positive']:.2f}")
    if pooled_sentiment:
        print(f"Pooled PPO+sentiment vs 1/N Sharpe diff: {pooled_sentiment['sharpe_diff']:.3f} "
              f"[{pooled_sentiment['ci_low']:.3f}, {pooled_sentiment['ci_high']:.3f}]  "
              f"P(>0)={pooled_sentiment['prob_positive']:.2f}")
    print(f"Deflated Sharpe (best seed, {len(runs)} trials): "
          f"{dsr['deflated_sharpe']:.3f}")
    print("\n[done] wrote eval_main_table.csv, eval_stats.json, evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
