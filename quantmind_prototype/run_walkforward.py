"""W4 walk-forward evaluation: does the agent hold up across market regimes?

A single train/test split can flatter a model. Walk-forward retrains the agent on
an expanding window and tests on the next unseen year, three times, under the same
fixed 150k-step protocol as the main runs. For each window it reports the five-seed
Sharpe distribution against the 1/N baseline, so the result is read as a
distribution per regime rather than a single number.

Windows (train -> test):
  2015-2019 -> 2020 ;  2015-2020 -> 2021 ;  2015-2021 -> 2022-2024

Writes results/walkforward.json and results/walkforward.png. Retrains 3 x 5 = 15
agents (~half an hour on CPU); price-only, since the news data is train-window only.

Run:  python run_walkforward.py
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest, config, data, metrics, train

OUT = config.RESULTS_DIR
SEEDS = [42, 1, 7, 13, 99]
WINDOWS = [
    ("2015-2019 -> 2020", "2020-01-01", "2021-01-01"),
    ("2015-2020 -> 2021", "2021-01-01", "2022-01-01"),
    ("2015-2021 -> 2022-24", "2022-01-01", "2025-01-01"),
]


def _slice(features, log_rets, prices, train_end, test_end):
    idx = log_rets.index
    tr = np.asarray(idx < pd.Timestamp(train_end))
    te = np.asarray((idx >= pd.Timestamp(train_end)) & (idx < pd.Timestamp(test_end)))
    return (features[tr], log_rets[tr]), (features[te], log_rets[te])


def main() -> int:
    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    print(f"source={source}", flush=True)

    results = []
    for label, train_end, test_end in WINDOWS:
        (tr_feat, tr_rets), (te_feat, te_rets) = _slice(
            features, log_rets, prices, train_end, test_end)
        # The agent only earns returns from day 1 of the window onward (day 0 is
        # consumed as its first observation), so 1/N is aligned to the same
        # window before comparison -- see run_evaluation.py for the full story.
        bh = backtest.buy_and_hold(te_rets.iloc[1:])
        bh_sharpe = metrics.sharpe(bh.values, config.RISK_FREE_DAILY)
        sharpes = []
        for seed in SEEDS:
            model = train.train_agent(tr_feat, tr_rets,
                                      total_timesteps=config.TOTAL_TIMESTEPS,
                                      seed=seed, verbose=0, save=False)
            agent_rets, _ = backtest.run_agent(model, te_feat, te_rets)
            sharpes.append(metrics.sharpe(agent_rets.values, config.RISK_FREE_DAILY))
        sharpes = np.array(sharpes)
        row = {"window": label, "test_days": int(te_rets.shape[0]),
               "agent_sharpe_mean": float(sharpes.mean()),
               "agent_sharpe_std": float(sharpes.std(ddof=1)),
               "agent_sharpes": sharpes.tolist(),
               "buy_hold_sharpe": float(bh_sharpe),
               "n_beating_bh": int((sharpes > bh_sharpe).sum())}
        results.append(row)
        print(f"[{label}] agent Sharpe {sharpes.mean():.2f} +/- {sharpes.std(ddof=1):.2f} "
              f"vs 1/N {bh_sharpe:.2f}; {row['n_beating_bh']}/5 beat it", flush=True)

    with open(os.path.join(OUT, "walkforward.json"), "w") as fh:
        json.dump({"windows": results, "seeds": SEEDS,
                   "timesteps": config.TOTAL_TIMESTEPS}, fh, indent=2)

    # Strip plot: per-window seed Sharpes with the 1/N marker.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(results):
        xs = np.full(len(r["agent_sharpes"]), i) + np.random.uniform(-0.06, 0.06, len(r["agent_sharpes"]))
        ax.scatter(xs, r["agent_sharpes"], color="#3b6ea5", s=40, zorder=3, label="PPO seeds" if i == 0 else None)
        ax.plot([i - 0.2, i + 0.2], [r["buy_hold_sharpe"]] * 2, color="#c0392b", lw=2,
                label="1/N baseline" if i == 0 else None)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r["window"] for r in results], fontsize=9)
    ax.set_ylabel("Sharpe ratio (test window)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "walkforward.png"), dpi=150)
    plt.close(fig)
    print("[done] wrote walkforward.json, walkforward.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
