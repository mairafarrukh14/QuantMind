"""Build the FinBERT per-(asset, date) sentiment cache and report coverage.

Reads the filtered headlines produced by ``fetch_news.py``, scores each with
local FinBERT under the hard pre-close leakage filter, aggregates to one value
per asset per trading day, and writes:

  * data/news/sentiment_cache.csv      -- the cache the env/dashboard read
  * results/sentiment_coverage.json    -- coverage %, the honest-signal metric
                                           Chapter 5 needs

Run once, offline-reproducible thereafter:  python build_sentiment.py
"""
from __future__ import annotations

import json
import os

from src import config, data, sentiment


def main() -> int:
    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    trading_days = log_rets.index
    print(f"prices source={source}  trading days={len(trading_days)} "
          f"({trading_days.min().date()} -> {trading_days.max().date()})", flush=True)

    print("scoring headlines with FinBERT (leakage-filtered)...", flush=True)
    agg = sentiment.build_sentiment_cache(trading_days)
    print(f"cached {len(agg):,} asset-day scores -> {sentiment.CACHE_CSV}", flush=True)

    # Full-span, train-window and test-window coverage (CO-4): the ablation in
    # Section 5.3 reads the test-window, per-asset breakdown specifically, so
    # that split -- not just the full-span average -- is what must be locked.
    split = data.train_test_split(prices, features, log_rets)
    _, train_rets, _ = split["train"]
    _, test_rets, _ = split["test"]

    full_span = sentiment.coverage_report(trading_days=trading_days)
    train_window = sentiment.coverage_report(trading_days=train_rets.index)
    test_window = sentiment.coverage_report(trading_days=test_rets.index)

    out = {
        "source": "FNSPID (Dong et al. 2024, CC BY-NC-4.0)",
        "full_span": full_span,
        "train_window": train_window,
        "test_window": test_window,
        # Kept for any code still reading the old flat shape.
        "overall": full_span["overall"], "per_asset": full_span["per_asset"],
        "n_trading_days": full_span["n_trading_days"],
    }
    with open(os.path.join(config.RESULTS_DIR, "sentiment_coverage.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("coverage (fraction of days with >=1 qualifying headline):")
    print(f"  full-span overall: {full_span['overall']:.1%} "
          f"(n={full_span['n_trading_days']} feature-aligned days)")
    print(f"  train-window overall: {train_window['overall']:.1%}")
    print(f"  test-window overall: {test_window['overall']:.1%}")
    for tkr, frac in test_window["per_asset"].items():
        print(f"    test-window {tkr}: {frac:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
