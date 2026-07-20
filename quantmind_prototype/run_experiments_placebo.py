"""CO-17 - placebo-sentiment control: is the sentiment ablation information or dynamics?

Section 5.3 concedes that five seeds cannot rule out the extra input dimension
simply changing training dynamics rather than supplying information. This trains
five seeds under the identical W3 protocol, with the sentiment feature's time axis
shuffled per asset (attach_sentiment_placebo): same values, same coverage, same
scale -- but no longer aligned to the market days they are supposed to describe.

If the placebo's mean collapses toward the price-only result (0.55), the real
signal was informative. If the placebo also rises toward the real-sentiment
result (0.90), the ablation was picking up training dynamics, not information.

Run (needs the sentiment cache from build_sentiment.py):
    python run_experiments_placebo.py
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src import backtest, config, data, metrics, sentiment, train

SEEDS = [42, 1, 7, 13, 99]
RUNS_DIR = os.path.join(config.RESULTS_DIR, "runs")
VARIANT = "placebo"


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


def main() -> int:
    if not os.path.exists(sentiment.CACHE_CSV):
        raise SystemExit("placebo needs the sentiment cache; run build_sentiment.py first")

    os.makedirs(RUNS_DIR, exist_ok=True)
    prices, source = data.load_prices()
    base_features, log_rets = data.build_features(prices)
    print(f"data source={source}  days={len(log_rets)}", flush=True)

    features = sentiment.attach_sentiment_placebo(base_features, log_rets,
                                                   list(log_rets.columns), seed=0)
    print(f"[placebo] obs feature dims per asset = {features.shape[2]} "
          f"(sentiment column shuffled per asset, values/coverage unchanged)", flush=True)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]

    index = {"seeds": SEEDS, "variant": VARIANT, "timesteps": config.TOTAL_TIMESTEPS, "runs": []}

    for seed in SEEDS:
        run_id = f"{VARIANT}_seed{seed}"
        run_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            with open(os.path.join(run_dir, "metrics.json")) as fh:
                prev = json.load(fh)
            index["runs"].append({"run_id": run_id, "seed": seed, "metrics": prev["metrics"]})
            print(f"[placebo] seed {seed} already locked -> skip", flush=True)
            continue
        print(f"[placebo] training seed {seed} ...", flush=True)

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
            json.dump({"variant": VARIANT, "seed": seed,
                       "timesteps": config.TOTAL_TIMESTEPS, "metrics": m,
                       "curve": {"t": cb.t, "r": cb.r}}, fh, indent=2)

        index["runs"].append({"run_id": run_id, "seed": seed, "metrics": m})
        print(f"      Sharpe={m['Sharpe']:.3f}  TotRet={m['Total Return']:.3f}  "
              f"MaxDD={m['Max Drawdown']:.3f}", flush=True)

    with open(os.path.join(config.RESULTS_DIR, "placebo_index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    sharpes = [r["metrics"]["Sharpe"] for r in index["runs"]]
    mean, sd = float(np.mean(sharpes)), float(np.std(sharpes, ddof=1))
    print(f"\n[placebo] mean Sharpe {mean:.3f} +/- {sd:.3f} across {len(sharpes)} seeds", flush=True)
    print("  price-only mean was 0.553; real-sentiment mean was 0.903", flush=True)
    print("  placebo near price-only -> real signal is informative", flush=True)
    print("  placebo near real-sentiment -> ablation was training dynamics, not information", flush=True)
    print(f"[done] {len(index['runs'])} locked runs under {RUNS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
