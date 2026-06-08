"""Data ingestion + feature engineering for QuantMind.

Responsibilities:
  1. Download daily OHLCV for the asset universe (yfinance), with on-disk caching
     so the pipeline is reproducible and works offline after the first run.
  2. If the network / Yahoo is unavailable (it rate-limits aggressively), fall
     back to a correlated geometric-Brownian-motion simulator so the prototype
     and the demo video always run. The fallback is clearly flagged.
  3. Engineer the per-asset technical features the RL agent observes.

All indicators are implemented from first principles (no TA-Lib dependency) so
the report can describe exactly what each feature is.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
#  Price loading                                                              #
# --------------------------------------------------------------------------- #
def _cache_path() -> str:
    return os.path.join(config.DATA_DIR, "prices.csv")


def _download_yfinance(tickers, start, end) -> pd.DataFrame | None:
    """Return a (dates x tickers) DataFrame of adjusted close, or None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True,
            progress=False, threads=False,
        )
        if raw is None or len(raw) == 0:
            return None
        # yfinance returns a column MultiIndex (field, ticker) for >1 ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:  # single ticker
            close = raw[["Close"]].copy()
            close.columns = tickers
        close = close.dropna(how="all").ffill().dropna()
        # Require a reasonable amount of history per ticker.
        if close.shape[0] < 200 or close.shape[1] < len(tickers):
            return None
        return close
    except Exception:
        return None


def _simulate_prices(tickers, start, end, seed=config.SEED) -> pd.DataFrame:
    """Correlated GBM fallback so the prototype always has data to run on."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days, n_assets = len(dates), len(tickers)

    # Plausible annual drifts / vols for the five sectors in TICKERS.
    mu = np.array([0.18, 0.16, 0.10, 0.08, 0.07])[:n_assets]
    sigma = np.array([0.28, 0.26, 0.24, 0.30, 0.18])[:n_assets]
    mu, sigma = mu / 252.0, sigma / np.sqrt(252.0)

    # Shared market factor -> realistic cross-asset correlation.
    corr = 0.4 * np.ones((n_assets, n_assets)) + 0.6 * np.eye(n_assets)
    chol = np.linalg.cholesky(corr)

    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    rets = mu + sigma * shocks
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def load_prices(force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Load adjusted-close prices for the universe.

    Returns (prices, source) where source is 'yfinance', 'cache' or 'simulated'.
    """
    path = _cache_path()
    if os.path.exists(path) and not force_refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        return prices, "cache"

    prices = _download_yfinance(config.TICKERS, config.START_DATE, config.END_DATE)
    if prices is not None:
        prices.to_csv(path)
        return prices, "yfinance"

    prices = _simulate_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    prices.to_csv(path)
    return prices, "simulated"


# --------------------------------------------------------------------------- #
#  Technical indicators                                                       #
# --------------------------------------------------------------------------- #
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _macd_hist(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def build_features(prices: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Turn prices into the per-asset feature tensor the agent observes.

    Returns:
        features: ndarray (T, n_assets, n_features)  -- normalised, ready for RL
        log_rets: DataFrame (T, n_assets)            -- realised next-step returns
    """
    log_px = np.log(prices)
    log_ret_1d = log_px.diff()
    log_ret_5d = log_px.diff(5)

    per_asset = []
    for tkr in prices.columns:
        px = prices[tkr]
        rsi = (_rsi(px) - 50.0) / 50.0                       # -> ~[-1, 1]
        macd = _macd_hist(px)
        macd = macd / (px.rolling(20).std() + 1e-8)          # scale-free
        vol20 = log_ret_1d[tkr].rolling(20).std() * np.sqrt(252)
        sma5 = px.rolling(5).mean()
        sma20 = px.rolling(20).mean()
        feats = pd.DataFrame({
            "ret_1d": log_ret_1d[tkr],
            "ret_5d": log_ret_5d[tkr],
            "rsi_14": rsi,
            "macd_hist": macd.clip(-5, 5),
            "vol_20": (vol20 - 0.2).clip(-1, 2),             # centre near 0
            "px_sma20": (px / sma20 - 1.0).clip(-0.5, 0.5),
            "sma5_sma20": (sma5 / sma20 - 1.0).clip(-0.5, 0.5),
        })
        per_asset.append(feats[config.FEATURE_NAMES])

    # Align, drop the warm-up rows where indicators are undefined.
    combined = pd.concat(per_asset, axis=1, keys=prices.columns).dropna()
    log_rets = log_ret_1d.loc[combined.index]

    T = len(combined)
    n_assets = len(prices.columns)
    n_features = len(config.FEATURE_NAMES)
    features = combined.values.reshape(T, n_assets, n_features).astype(np.float32)
    return features, log_rets


def train_test_split(prices, features, log_rets, split=config.TRAIN_TEST_SPLIT):
    """Split aligned arrays/frames at a date boundary (no leakage)."""
    idx = log_rets.index
    mask = np.asarray(idx < pd.Timestamp(split))
    prices_aligned = prices.loc[log_rets.index]
    return {
        "train": (features[mask], log_rets[mask], prices_aligned[mask]),
        "test": (features[~mask], log_rets[~mask], prices_aligned[~mask]),
    }


if __name__ == "__main__":
    prices, source = load_prices()
    print(f"Loaded prices from: {source}  shape={prices.shape}")
    feats, rets = build_features(prices)
    print(f"Features: {feats.shape}  (T, assets, features)")
    print(f"Date range: {rets.index.min().date()} -> {rets.index.max().date()}")
