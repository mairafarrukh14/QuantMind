"""Live recommendation path: advice for a user-supplied amount and as-of date.

The dashboard replays a locked backtest. This module answers the template's other
requirement, that a non-technical user can obtain advice for their own situation.
It never trains anything. At request time it:

  1. loads the cached price feed and computes the feature vector for the as-of
     date (every indicator is causal, so the vector at day t uses only data up to
     and including day t);
  2. builds the policy's observation for a new portfolio, which starts in cash;
  3. runs the locked best-seed policy to get target weights, then applies the
     concentration cap (and any tighter user cap or exclusions);
  4. attributes each holding to its top Shapley drivers and, if a language model
     is available, phrases the top holding's reason through the rationale audit.

Dates on or after the train/test split are held out from training; earlier dates
are in-sample for the policy and are flagged as such.
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, data, explain, rationale
from .env import apply_position_cap

ASSET_META = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
}
MIN_AMOUNT, MAX_AMOUNT = 100.0, 100_000_000.0
MIN_CAP = 1.0 / len(config.TICKERS)   # below equal weight the cap could not sum to 1


@lru_cache(maxsize=1)
def _state():
    """Load everything that does not depend on the request, once."""
    from stable_baselines3 import PPO

    prices, source = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]

    best_path, best_sharpe = None, -1e9
    for mp in glob.glob(os.path.join(config.RESULTS_DIR, "runs", "price_seed*", "metrics.json")):
        with open(mp) as fh:
            sh = json.load(fh)["metrics"]["Sharpe"]
        if sh > best_sharpe:
            best_sharpe, best_path = sh, os.path.join(os.path.dirname(mp), "model")
    model = PPO.load(best_path) if best_path else PPO.load(
        os.path.join(config.MODELS_DIR, "ppo_quantmind"))

    rng = np.random.default_rng(config.SEED)
    bg_all = explain._observations(model, tr_feat, tr_rets)
    background = bg_all[rng.choice(len(bg_all), size=min(30, len(bg_all)), replace=False)]
    return {"features": features, "index": log_rets.index, "tickers": list(log_rets.columns),
            "model": model, "background": background, "source": source,
            "model_name": os.path.basename(os.path.dirname(best_path)) if best_path else "ppo_quantmind"}


def date_range() -> dict:
    """First and last date a recommendation can be made for."""
    s = _state()
    return {"first": s["index"][0].strftime("%Y-%m-%d"),
            "last": s["index"][-1].strftime("%Y-%m-%d"),
            "held_out_from": config.TRAIN_TEST_SPLIT, "source": s["source"]}


def resolve_date(index: pd.DatetimeIndex, as_of: str | None) -> tuple[int, str | None]:
    """Position of the latest trading day on or before ``as_of`` and a note if it moved."""
    if as_of is None:
        return len(index) - 1, None
    try:
        ts = pd.Timestamp(as_of)
    except (ValueError, TypeError):
        raise ValueError("as_of must be a date such as 2024-06-28") from None
    if ts < index[0]:
        raise ValueError(f"as_of must be on or after {index[0].date()}")
    pos = int(index.searchsorted(ts, side="right")) - 1
    if ts > index[-1]:
        return pos, f"The price feed ends on {index[-1].date()}, so that date was used."
    if index[pos] != ts:
        return pos, f"{ts.date()} was not a trading day; {index[pos].date()} was used."
    return pos, None


def target_weights(model, obs: np.ndarray, cap: float, exclude: tuple[str, ...],
                   tickers: list[str]) -> np.ndarray:
    """Policy weights for one observation, with the user's constraints applied.

    The policy output already respects the environment's cap. A tighter cap is
    re-applied with cash absorbing the excess. Excluded assets are set to zero and
    their weight also moves to cash. Both are constraints on the output, not a
    different policy, and the interface says so.
    """
    n = len(tickers)
    w = explain.make_policy_fn(model)(obs.reshape(1, -1))[0].astype(float)
    if cap < config.MAX_WEIGHT - 1e-12:
        w = apply_position_cap(w, cap, n)
    for tk in exclude:
        if tk in tickers:
            i = tickers.index(tk)
            w[-1] += w[i]
            w[i] = 0.0
    return w


def recommend(amount: float, as_of: str | None = None, max_weight: float | None = None,
              exclude: list[str] | None = None, explanations: bool = True,
              n_perm: int = 64) -> dict:
    amount = float(amount)
    if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
        raise ValueError(f"amount must be between {MIN_AMOUNT:,.0f} and {MAX_AMOUNT:,.0f}")
    cap = config.MAX_WEIGHT if max_weight is None else float(max_weight)
    if not (MIN_CAP <= cap <= config.MAX_WEIGHT):
        raise ValueError(f"max_weight must be between {MIN_CAP:.2f} and {config.MAX_WEIGHT:.2f}")
    s = _state()
    tickers = s["tickers"]
    exclude = tuple(exclude or ())
    unknown = [t for t in exclude if t not in tickers]
    if unknown:
        raise ValueError(f"unknown assets: {unknown}")
    if len(exclude) >= len(tickers):
        raise ValueError("at least one asset must remain")

    pos, note = resolve_date(s["index"], as_of)
    # A new portfolio starts fully in cash, exactly as the environment resets.
    start_w = np.zeros(len(tickers) + 1, dtype=np.float32)
    start_w[-1] = 1.0
    obs = np.concatenate([s["features"][pos].reshape(-1), start_w]).astype(np.float32)

    w = target_weights(s["model"], obs, cap, exclude, tickers)
    date = s["index"][pos]
    out = {
        "as_of": date.strftime("%Y-%m-%d"), "date_note": note,
        "in_sample": bool(date < pd.Timestamp(config.TRAIN_TEST_SPLIT)),
        "amount": amount, "cap_used": cap, "excluded": list(exclude),
        "starting_portfolio": "all cash",
        "model": s["model_name"], "data_source": s["source"],
        "cash": {"weight": float(w[-1]), "amount": round(float(w[-1]) * amount, 2)},
        "notice": ("Backtested research prototype, not financial advice. The policy was "
                   "trained on 2015-2021 prices and has no live market feed."),
    }
    holdings = []
    for i, tk in enumerate(tickers):
        wi = float(w[i])
        holdings.append({"ticker": tk, "name": ASSET_META.get(tk, {}).get("name", tk),
                         "sector": ASSET_META.get(tk, {}).get("sector", ""),
                         "weight": wi, "amount": round(wi * amount, 2),
                         "action": "Overweight" if wi >= 0.30 else "Hold" if wi >= 0.12 else "Underweight"})

    if explanations:
        f = explain.make_policy_fn(s["model"])
        rng = np.random.default_rng(config.SEED)
        phi = explain.shapley_sampling(f, obs, s["background"], n_perm=n_perm, rng=rng)
        names = explain.build_feature_names(tickers)
        for i, h in enumerate(holdings):
            own = [k for k, n in enumerate(names) if n.startswith(f"{h['ticker']}:")]
            top = sorted(own, key=lambda k: abs(phi[k, i]), reverse=True)[:3]
            h["drivers"] = [{"feature": names[k].split(":")[1],
                             "phrase": explain.FEATURE_PHRASES.get(names[k].split(":")[1], names[k]),
                             "direction": "up" if phi[k, i] > 0 else "down",
                             "magnitude": float(abs(phi[k, i]))} for k in top]
        lead = max(holdings, key=lambda h: h["weight"])
        if lead["weight"] > 0 and lead["drivers"]:
            out["rationale"] = {"holding": lead["ticker"], **rationale.generate_rationale(
                lead["ticker"], lead["weight"],
                [{"feature": d["feature"], "phrase": d["phrase"],
                  "sign": 1 if d["direction"] == "up" else -1} for d in lead["drivers"]])}
    holdings.sort(key=lambda h: h["weight"], reverse=True)
    out["holdings"] = holdings
    return out
