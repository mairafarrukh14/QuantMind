"""W3 - controlled retraining: the ten locked runs the evaluation reads.

Five seeds x two variants (sentiment OFF / ON), 150k PPO steps each, identical
hyperparameters, same 2015-2021 train window. For every run we save the model
zip, a metrics JSON and the per-day weight matrix, under results/runs/. These
artifacts are locked: W4 evaluation and the dashboard read them and nothing is
retrained afterwards.

The fixed 150k budget is a deliberate, recorded choice - it holds the training
signal constant so the seed spread and the sentiment ablation are a controlled
comparison, not a hyperparameter hunt. Known non-convergence is acknowledged as
further work rather than fixed here.

Run:  python run_experiments_w3.py
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from src import backtest, config, data, metrics, sentiment, train

SEEDS = [42, 1, 7, 13, 99]
VARIANTS = ["price", "sentiment"]  # price-only vs +FinBERT sentiment feature
RUNS_DIR = os.path.join(config.RESULTS_DIR, "runs")


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.t, self.r = [], []

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            self.t.append(self.num_timesteps)
            self.r.append(float(np.mean([e["r"] for e in buf])))

    def _on_step(self):
        return True


def _features_for(variant, base_features, log_rets):
    if variant == "sentiment":
        return sentiment.attach_sentiment(base_features, log_rets, list(log_rets.columns))
    return base_features


def _available_variants():
    """Run the sentiment variant only if its cache exists; otherwise price-only,
    so the primary results are never blocked on the news data."""
    if os.path.exists(sentiment.CACHE_CSV):
        return VARIANTS
    print("[note] no sentiment cache -> running price-only; "
          "re-run after build_sentiment.py to add the ablation", flush=True)
    return ["price"]


def main() -> int:
    os.makedirs(RUNS_DIR, exist_ok=True)
    prices, source = data.load_prices()
    base_features, log_rets = data.build_features(prices)
    print(f"data source={source}  days={len(log_rets)}", flush=True)

    variants = _available_variants()
    index = {"seeds": SEEDS, "variants": variants,
             "timesteps": config.TOTAL_TIMESTEPS, "runs": []}

    for variant in variants:
        features = _features_for(variant, base_features, log_rets)
        split = data.train_test_split(prices, features, log_rets)
        tr_feat, tr_rets, _ = split["train"]
        te_feat, te_rets, _ = split["test"]
        print(f"[{variant}] obs feature dims per asset = {features.shape[2]}", flush=True)

        for seed in SEEDS:
            run_id = f"{variant}_seed{seed}"
            run_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(run_dir, exist_ok=True)

            # Idempotent: skip a run whose locked artifacts already exist, so
            # adding the sentiment variant does not retrain the price runs.
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                with open(os.path.join(run_dir, "metrics.json")) as fh:
                    prev = json.load(fh)
                index["runs"].append({"run_id": run_id, "variant": variant,
                                      "seed": seed, "metrics": prev["metrics"]})
                print(f"[{variant}] seed {seed} already locked -> skip", flush=True)
                continue
            print(f"[{variant}] training seed {seed} ...", flush=True)

            cb = RewardLogger()
            model = train.train_agent(
                tr_feat, tr_rets, total_timesteps=config.TOTAL_TIMESTEPS,
                seed=seed, verbose=0, callback=cb,
                save_path=os.path.join(run_dir, "model"), save=True)

            agent_rets, weights = backtest.run_agent(model, te_feat, te_rets)
            m = metrics.summary(agent_rets.values, rf_daily=config.RISK_FREE_DAILY)
            m["Turnover"] = float(weights.diff().abs().sum(axis=1).mean())
            weights.to_csv(os.path.join(run_dir, "weights.csv"))
            agent_rets.to_frame("port_ret").to_csv(os.path.join(run_dir, "returns.csv"))
            with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
                json.dump({"variant": variant, "seed": seed,
                           "timesteps": config.TOTAL_TIMESTEPS, "metrics": m,
                           "curve": {"t": cb.t, "r": cb.r}}, fh, indent=2)

            index["runs"].append({"run_id": run_id, "variant": variant,
                                   "seed": seed, "metrics": m})
            print(f"      Sharpe={m['Sharpe']:.3f}  TotRet={m['Total Return']:.3f}  "
                  f"MaxDD={m['Max Drawdown']:.3f}", flush=True)

    with open(os.path.join(config.RESULTS_DIR, "w3_index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"\n[done] {len(index['runs'])} locked runs under {RUNS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
