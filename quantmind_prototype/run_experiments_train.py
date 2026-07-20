"""Training-based experiments for the revision:
  E1 — multi-seed robustness (train several agents, report mean +/- std).
  E3 — training stability (reward vs timesteps curve for each seed).

Writes results/multiseed.csv, results/multiseed.json and
results/training_curve.png. Does NOT overwrite the main saved model.
"""
from __future__ import annotations
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from src import data, backtest, train, metrics, config

OUT = config.RESULTS_DIR
SEEDS = [42, 1, 7, 13, 99]


class RewardLogger(BaseCallback):
    """Record mean episode reward against timesteps at each rollout end."""
    def __init__(self):
        super().__init__()
        self.t, self.r = [], []

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            self.t.append(self.num_timesteps)
            self.r.append(float(np.mean([e["r"] for e in buf])))

    def _on_step(self):
        return True


def main():
    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]

    # Fixed baseline (same for every seed).
    bh = backtest.buy_and_hold(te_rets)
    bh_m = metrics.summary(bh.values, rf_daily=config.RISK_FREE_DAILY)

    rows, curves = [], {}
    for s in SEEDS:
        print(f"[E1] training seed {s} ...", flush=True)
        cb = RewardLogger()
        model = train.train_agent(tr_feat, tr_rets,
                                  total_timesteps=config.TOTAL_TIMESTEPS,
                                  seed=s, verbose=0, callback=cb, save=False)
        agent_rets, weights = backtest.run_agent(model, te_feat, te_rets)
        m = metrics.summary(agent_rets.values, rf_daily=config.RISK_FREE_DAILY)
        m["Turnover"] = float(weights.diff().abs().sum(axis=1).mean())
        m["seed"] = s
        rows.append(m)
        curves[s] = (cb.t, cb.r)
        print(f"      seed {s}: Sharpe={m['Sharpe']:.3f} "
              f"TotRet={m['Total Return']:.3f} MaxDD={m['Max Drawdown']:.3f}", flush=True)

    df = pd.DataFrame(rows).set_index("seed")
    df.to_csv(os.path.join(OUT, "multiseed.csv"))

    cols = ["Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Turnover"]
    agg = {c: {"mean": float(df[c].mean()), "std": float(df[c].std()),
               "min": float(df[c].min()), "max": float(df[c].max())} for c in cols}
    out = {
        "seeds": SEEDS,
        "timesteps": config.TOTAL_TIMESTEPS,
        "agent": agg,
        "buy_and_hold": {k: float(v) for k, v in bh_m.items()},
        "n_beating_bh_sharpe": int((df["Sharpe"] > bh_m["Sharpe"]).sum()),
    }
    with open(os.path.join(OUT, "multiseed.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n[E1] across seeds:")
    print(f"   Sharpe  {agg['Sharpe']['mean']:.3f} +/- {agg['Sharpe']['std']:.3f} "
          f"(min {agg['Sharpe']['min']:.3f}, max {agg['Sharpe']['max']:.3f})")
    print(f"   TotRet  {agg['Total Return']['mean']:.3f} +/- {agg['Total Return']['std']:.3f}")
    print(f"   MaxDD   {agg['Max Drawdown']['mean']:.3f} +/- {agg['Max Drawdown']['std']:.3f}")
    print(f"   Buy&Hold Sharpe = {bh_m['Sharpe']:.3f}; "
          f"seeds beating it: {out['n_beating_bh_sharpe']}/{len(SEEDS)}")

    # E3 — training curves
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s in SEEDS:
        t, r = curves[s]
        if t:
            ax.plot(t, r, linewidth=1.3, label=f"seed {s}")
    ax.set_xlabel("Environment timesteps"); ax.set_ylabel("Mean episode reward")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=150)
    plt.close(fig)
    print("\n[done] wrote multiseed.csv, multiseed.json, training_curve.png")


if __name__ == "__main__":
    main()
