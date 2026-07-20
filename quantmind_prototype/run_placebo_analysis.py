"""CO-17 analysis: does the placebo confirm information or dynamics?

Reads the five locked placebo runs (results/runs/placebo_seed*) alongside the
price-only and real-sentiment runs already evaluated by run_evaluation.py, and
answers the question Section 5.3 left open: is the sentiment ablation's higher
mean Sharpe genuine information, or just the extra input dimension changing
training dynamics? If the placebo (same values, shuffled dates) performs like
real sentiment, it is dynamics; if it collapses toward price-only, it is
information.

Run (after run_experiments_placebo.py):  python run_placebo_analysis.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src import backtest, config, data, stats

OUT = config.RESULTS_DIR
RUNS_DIR = os.path.join(OUT, "runs")


def _load_returns(run_id: str) -> np.ndarray:
    path = os.path.join(RUNS_DIR, run_id, "returns.csv")
    return pd.read_csv(path, index_col=0)["port_ret"].values


def _load_sharpes(prefix: str) -> list[float]:
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, f"{prefix}_seed*", "metrics.json"))):
        with open(path) as fh:
            out.append(json.load(fh)["metrics"]["Sharpe"])
    return out


def main() -> int:
    placebo_sharpes = _load_sharpes("placebo")
    price_sharpes = _load_sharpes("price")
    sentiment_sharpes = _load_sharpes("sentiment")
    if not placebo_sharpes:
        raise SystemExit("no placebo runs found; run run_experiments_placebo.py first")

    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    _, te_rets_full, _ = split["test"]
    aligned = backtest.align_to_tradeable(te_rets_full)
    bh = backtest.buy_and_hold(aligned)

    placebo_seeds = sorted(int(os.path.basename(p).split("seed")[1])
                           for p in glob.glob(os.path.join(RUNS_DIR, "placebo_seed*")))
    pooled_placebo = np.concatenate([_load_returns(f"placebo_seed{s}") for s in placebo_seeds])
    pooled_bh = np.concatenate([bh.values for _ in placebo_seeds])
    vs_1N = stats.bootstrap_sharpe_diff_ci(pooled_placebo, pooled_bh, n_boot=10_000)

    sentiment_seeds = sorted(int(os.path.basename(p).split("seed")[1])
                             for p in glob.glob(os.path.join(RUNS_DIR, "sentiment_seed*")))
    pooled_sentiment = np.concatenate([_load_returns(f"sentiment_seed{s}") for s in sentiment_seeds])
    # placebo vs real sentiment: is the DROP from real to placebo distinguishable
    # from zero? A near-zero, straddling-zero result means "no detectable loss of
    # information when dates are shuffled" -- i.e. dynamics, not information.
    vs_real_sentiment = stats.bootstrap_sharpe_diff_ci(pooled_sentiment, pooled_placebo, n_boot=10_000)

    price_seeds = sorted(int(os.path.basename(p).split("seed")[1])
                         for p in glob.glob(os.path.join(RUNS_DIR, "price_seed*")))
    pooled_price = np.concatenate([_load_returns(f"price_seed{s}") for s in price_seeds])
    vs_price = stats.bootstrap_sharpe_diff_ci(pooled_placebo, pooled_price, n_boot=10_000)

    out = {
        "placebo_mean_sharpe": float(np.mean(placebo_sharpes)),
        "placebo_sd_sharpe": float(np.std(placebo_sharpes, ddof=1)),
        "placebo_sharpes": placebo_sharpes,
        "price_only_mean_sharpe": float(np.mean(price_sharpes)),
        "real_sentiment_mean_sharpe": float(np.mean(sentiment_sharpes)),
        "placebo_vs_1N": vs_1N,
        "placebo_vs_price_only": vs_price,
        "real_sentiment_vs_placebo": vs_real_sentiment,
        "verdict": (
            "dynamics: placebo (shuffled-date sentiment) performs close to real "
            "sentiment and well above price-only, so the earlier ablation's gain "
            "is not attributable to genuine information in the sentiment signal"
            if abs(np.mean(placebo_sharpes) - np.mean(sentiment_sharpes))
               < abs(np.mean(placebo_sharpes) - np.mean(price_sharpes))
            else
            "information: placebo collapses toward price-only while real sentiment "
            "holds higher, consistent with the sentiment signal carrying genuine "
            "information rather than just adding an input dimension"
        ),
    }
    with open(os.path.join(OUT, "placebo_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"placebo mean Sharpe: {out['placebo_mean_sharpe']:.3f} +/- {out['placebo_sd_sharpe']:.3f}")
    print(f"  vs price-only ({out['price_only_mean_sharpe']:.3f}): "
          f"diff {vs_price['sharpe_diff']:.3f} [{vs_price['ci_low']:.3f}, {vs_price['ci_high']:.3f}]")
    print(f"  vs real sentiment ({out['real_sentiment_mean_sharpe']:.3f}): "
          f"real-minus-placebo {vs_real_sentiment['sharpe_diff']:.3f} "
          f"[{vs_real_sentiment['ci_low']:.3f}, {vs_real_sentiment['ci_high']:.3f}]")
    print(f"  vs 1/N: {vs_1N['sharpe_diff']:.3f} [{vs_1N['ci_low']:.3f}, {vs_1N['ci_high']:.3f}] "
          f"P(>0)={vs_1N['prob_positive']:.2f}")
    print(f"\nverdict: {out['verdict']}")
    print("[done] wrote placebo_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
