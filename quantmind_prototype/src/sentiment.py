"""FinBERT news-sentiment pipeline (Work Order W2).

Turns time-stamped headlines into one sentiment score per asset per day, in
[-1, 1], that the RL agent can observe as an 8th feature. Two rules make it
safe to use in a backtest:

* **Hard leakage filter, in code.** A headline can only enter the score the
  agent observes at the close of trading day *t* if it was published strictly
  before that close. Headlines published after the close roll forward to the
  next session. The cut is enforced at 20:00 UTC (16:00 US Eastern in summer),
  the conservative choice: a late-afternoon headline is deferred rather than
  risk leaking into a decision that could not have seen it.
* **Honest coverage.** No qualifying headline for an asset-day yields a score of
  0.0 together with a no-coverage flag, so a mostly-missing signal is visible as
  missing rather than silently treated as neutral conviction.

Scores are FinBERT's ``mean(P(pos) - P(neg))`` over the qualifying headlines
(Araci, 2019; ProsusAI/finbert), run locally via Transformers — no paid APIs.
The per-(asset, date) result is cached to disk so the ablation is reproducible
offline.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

MODEL_NAME = "ProsusAI/finbert"
CLOSE_UTC_HOUR = 20  # conservative pre-close cut (16:00 US Eastern, summer)
NEWS_DIR = os.path.join(config.DATA_DIR, "news")
RAW_CSV = os.path.join(NEWS_DIR, "headlines_raw.csv")
CACHE_CSV = os.path.join(NEWS_DIR, "sentiment_cache.csv")


# --------------------------------------------------------------------------- #
#  FinBERT scorer (loaded lazily, once)                                       #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _finbert():
    import torch  # noqa: F401  (imported for side-effect / availability)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, lab["positive"], lab["negative"]


def finbert_scores(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return P(pos) - P(neg) in [-1, 1] for each headline, via local FinBERT."""
    import torch

    if not texts:
        return np.zeros(0, dtype=np.float64)
    tok, model, i_pos, i_neg = _finbert()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=64)
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            out.append(probs[:, i_pos] - probs[:, i_neg])
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
#  Leakage-safe per-asset-day scoring primitive                               #
# --------------------------------------------------------------------------- #
def score_asset_day(asset, day, headlines, close_hour: int = 16, scorer=None) -> float:
    """Sentiment for one asset on one day, over headlines strictly pre-close.

    ``headlines`` is an iterable of ``(asset, timestamp, text)``. Only same-asset
    headlines time-stamped before ``close_hour`` on ``day`` qualify; a headline
    after the close is excluded and cannot move the score. ``scorer`` maps a list
    of texts to per-headline scores (defaults to FinBERT); tests inject a stub so
    the leakage property is checked without loading the model.
    """
    scorer = scorer or finbert_scores
    day = pd.Timestamp(day).date() if not hasattr(day, "year") else day
    qualifying = [
        text for (a, ts, text) in headlines
        if a == asset and pd.Timestamp(ts).date() == day
        and pd.Timestamp(ts).hour < close_hour
    ]
    if not qualifying:
        return 0.0
    return float(np.mean(scorer(qualifying)))


# --------------------------------------------------------------------------- #
#  Cache builder: raw headlines -> per-(asset, date) score                    #
# --------------------------------------------------------------------------- #
def _effective_trading_date(ts_utc: pd.Series, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Map each UTC headline timestamp to the first trading day whose close is
    strictly after it -- the earliest observation that may legally include it."""
    closes = (trading_days.normalize() + pd.Timedelta(hours=CLOSE_UTC_HOUR))
    # first close strictly greater than ts  ->  searchsorted 'right'
    pos = np.searchsorted(closes.values, ts_utc.values, side="right")
    eff = np.full(len(ts_utc), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = pos < len(trading_days)
    eff[valid] = trading_days.values[pos[valid]]
    return eff


def build_sentiment_cache(trading_days: pd.DatetimeIndex,
                          raw_csv: str = RAW_CSV,
                          out_csv: str = CACHE_CSV) -> pd.DataFrame:
    """Score every raw headline with FinBERT and aggregate to one value per
    (asset, trading-date). Writes ``out_csv`` and returns the tidy frame."""
    raw = pd.read_csv(raw_csv)
    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"].str.replace(" UTC", "", regex=False), utc=False,
        errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc", "headline", "symbol"])

    raw["eff_date"] = _effective_trading_date(raw["timestamp_utc"],
                                              trading_days.tz_localize(None))
    raw = raw.dropna(subset=["eff_date"])

    raw["score"] = finbert_scores(raw["headline"].astype(str).tolist())

    agg = (raw.groupby(["symbol", "eff_date"])
              .agg(score=("score", "mean"), n_headlines=("score", "size"))
              .reset_index()
              .rename(columns={"eff_date": "date"}))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return agg


# --------------------------------------------------------------------------- #
#  Loading + feature attachment                                               #
# --------------------------------------------------------------------------- #
def load_sentiment_matrix(dates: pd.DatetimeIndex, tickers: list[str],
                          cache_csv: str = CACHE_CSV):
    """Return (scores, covered) aligned to ``dates`` x ``tickers``.

    ``scores`` is float in [-1, 1] with 0.0 where there is no coverage;
    ``covered`` is a bool mask of asset-days that had >=1 qualifying headline.
    """
    scores = pd.DataFrame(0.0, index=dates, columns=tickers)
    covered = pd.DataFrame(False, index=dates, columns=tickers)
    if not os.path.exists(cache_csv):
        return scores, covered
    cache = pd.read_csv(cache_csv, parse_dates=["date"])
    for tkr in tickers:
        sub = cache[cache["symbol"] == tkr].set_index("date")
        idx = sub.index.intersection(dates)
        scores.loc[idx, tkr] = sub.loc[idx, "score"].values
        covered.loc[idx, tkr] = True
    return scores, covered


def attach_sentiment(features: np.ndarray, log_rets, tickers: list[str],
                     cache_csv: str = CACHE_CSV) -> np.ndarray:
    """Append the sentiment score as an 8th per-asset feature (41 -> 46 dims).

    This is the single switch behind ``use_sentiment``: same env, one extra
    feature column per asset.
    """
    scores, _ = load_sentiment_matrix(log_rets.index, tickers, cache_csv)
    T, n_assets, _ = features.shape
    sent = scores.values.reshape(T, n_assets, 1).astype(np.float32)
    return np.concatenate([features, sent], axis=2)


def attach_sentiment_placebo(features: np.ndarray, log_rets, tickers: list[str],
                             cache_csv: str = CACHE_CSV, seed: int = 0) -> np.ndarray:
    """Placebo control for the sentiment ablation (Work Order CO-17).

    Attaches the same real sentiment scores as ``attach_sentiment``, but with
    each asset's time axis independently shuffled, so the *values* (and their
    distribution, coverage rate, scale) are identical to the real feature while
    the *date alignment* to the market it is meant to describe is destroyed.
    If a model trained on this placebo performs like the real-sentiment model,
    the earlier ablation was picking up the extra input dimension changing
    training dynamics, not information; if it collapses toward the price-only
    result, the real feature was carrying genuine information.
    """
    scores, _ = load_sentiment_matrix(log_rets.index, tickers, cache_csv)
    T, n_assets, _ = features.shape
    rng = np.random.default_rng(seed)
    shuffled = scores.values.copy()
    for j in range(n_assets):
        perm = rng.permutation(T)
        shuffled[:, j] = shuffled[perm, j]
    sent = shuffled.reshape(T, n_assets, 1).astype(np.float32)
    return np.concatenate([features, sent], axis=2)


def coverage_report(cache_csv: str = CACHE_CSV,
                    trading_days: pd.DatetimeIndex | None = None) -> dict:
    """Fraction of asset-days with >=1 qualifying headline, overall and per asset.

    When ``trading_days`` is given, both the denominator (days in that window) and
    the numerator (cached dates restricted to that window) use it -- otherwise a
    window such as the test period would be scored against coverage counted over
    the whole cache, which can nonsensically exceed 100%.
    """
    if not os.path.exists(cache_csv):
        return {"overall": 0.0, "per_asset": {}}
    cache = pd.read_csv(cache_csv, parse_dates=["date"])
    if trading_days is not None:
        window = pd.DatetimeIndex(trading_days).normalize()
        cache = cache[cache["date"].isin(window)]
        n_days = len(window)
    else:
        n_days = cache["date"].nunique()
    per_asset = {}
    for tkr in config.TICKERS:
        covered_days = cache[cache["symbol"] == tkr]["date"].nunique()
        per_asset[tkr] = round(covered_days / n_days, 4) if n_days else 0.0
    overall = round(float(np.mean(list(per_asset.values()))), 4) if per_asset else 0.0
    return {"overall": overall, "per_asset": per_asset, "n_trading_days": n_days}
