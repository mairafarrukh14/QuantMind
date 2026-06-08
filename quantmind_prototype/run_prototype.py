"""QuantMind feature-prototype — end-to-end pipeline.

    python run_prototype.py [--timesteps N] [--refresh] [--quick]

Stages: load data -> engineer features -> train PPO -> backtest vs baselines
-> SHAP explanations -> natural-language rationale. All artefacts land in
results/ (figures, metrics.csv, rationale.txt) for the report and demo video.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow `python run_prototype.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, data, backtest, explain, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    ap.add_argument("--refresh", action="store_true", help="re-download price data")
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    ap.add_argument("--no-shap", action="store_true", help="skip the (slow) SHAP stage")
    args = ap.parse_args()
    if args.quick:
        args.timesteps = 5_000

    t0 = time.time()
    np.random.seed(config.SEED)

    # 1. Data ---------------------------------------------------------------
    prices, source = data.load_prices(force_refresh=args.refresh)
    print(f"[data] source={source}  prices={prices.shape}  "
          f"{prices.index.min().date()}..{prices.index.max().date()}")
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]
    print(f"[data] train={tr_feat.shape[0]} days  test={te_feat.shape[0]} days  "
          f"assets={tr_feat.shape[1]}  features/asset={tr_feat.shape[2]}")

    # 2. Train --------------------------------------------------------------
    print(f"[train] PPO for {args.timesteps:,} timesteps ...")
    model = train.train_agent(tr_feat, tr_rets, total_timesteps=args.timesteps, verbose=0)

    # 3. Backtest -----------------------------------------------------------
    table, weights, agent_rets, extra = backtest.evaluate(model, te_feat, te_rets, source=source)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n[backtest] held-out test-period performance:\n")
    print(table.to_string())
    print(f"\n[backtest] avg daily turnover: {extra['Avg daily turnover']:.4f}")

    # 4. Explainability -----------------------------------------------------
    rationale_text = ""
    if not args.no_shap:
        print("\n[xai] computing Shapley-value attributions on the policy ...")
        phi_all, X_explain, sel, feat_names, tickers = explain.explain(
            model, tr_feat, tr_rets, te_feat, te_rets,
            n_background=15 if args.quick else 30,
            n_explain=8 if args.quick else 40,
            n_perm=24 if args.quick else 64,
        )
        # Narrate the most recent explained decision.
        j = len(X_explain) - 1
        rationale_text = explain.narrate_decision(
            model, X_explain[j], tickers, phi_all[j], feat_names,
        )
        print("\n[xai] example per-decision rationale:\n")
        print(rationale_text)
        with open(os.path.join(config.RESULTS_DIR, "rationale.txt"), "w") as fh:
            fh.write(rationale_text + "\n")

    # 5. Persist a run summary ---------------------------------------------
    summary = {
        "data_source": source,
        "train_days": int(tr_feat.shape[0]),
        "test_days": int(te_feat.shape[0]),
        "timesteps": int(args.timesteps),
        "metrics": json.loads(table.to_json(orient="index")),
        "avg_daily_turnover": float(extra["Avg daily turnover"]),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(config.RESULTS_DIR, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[done] {summary['runtime_sec']}s — artefacts in results/")


if __name__ == "__main__":
    main()
