"""Tests for the live recommendation path (src/recommend.py).

The look-ahead test is the important one: a recommendation for date t must not
depend on any price after t.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import config, data, recommend  # noqa: E402


class _StubModel:
    """Deterministic stand-in policy: fixed logits favouring the first asset."""
    def predict(self, X, deterministic=True):
        X = np.atleast_2d(X)
        a = np.zeros((len(X), len(config.TICKERS) + 1), dtype=np.float32)
        a[:, 0] = 5.0
        return a, None


def test_features_at_t_ignore_later_prices():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (400, len(config.TICKERS))), 0)),
                      index=idx, columns=config.TICKERS)
    full, full_r = data.build_features(px)
    cut = full_r.index[250]
    part, part_r = data.build_features(px.loc[:cut])
    assert part_r.index[-1] == cut
    np.testing.assert_allclose(part[-1], full[list(full_r.index).index(cut)], atol=1e-6)


def test_weights_form_valid_allocation_and_respect_cap():
    n = len(config.TICKERS)
    obs = np.zeros(n * len(config.FEATURE_NAMES) + n + 1, dtype=np.float32)
    w = recommend.target_weights(_StubModel(), obs, cap=0.25, exclude=(), tickers=config.TICKERS)
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-9).all()
    assert w[:n].max() <= 0.25 + 1e-6


def test_excluded_asset_gets_zero_and_weight_moves_to_cash():
    n = len(config.TICKERS)
    obs = np.zeros(n * len(config.FEATURE_NAMES) + n + 1, dtype=np.float32)
    base = recommend.target_weights(_StubModel(), obs, config.MAX_WEIGHT, (), config.TICKERS)
    ex = recommend.target_weights(_StubModel(), obs, config.MAX_WEIGHT, (config.TICKERS[0],),
                                  config.TICKERS)
    assert ex[0] == 0.0 and abs(ex.sum() - 1.0) < 1e-6
    assert ex[-1] >= base[-1] + base[0] - 1e-6


def test_resolve_date_uses_last_trading_day_and_reports_it():
    idx = pd.bdate_range("2020-01-01", periods=30)
    pos, note = recommend.resolve_date(idx, "2020-01-04")          # a Saturday
    assert idx[pos] == pd.Timestamp("2020-01-03") and note
    pos, note = recommend.resolve_date(idx, None)
    assert pos == len(idx) - 1 and note is None
    with pytest.raises(ValueError):
        recommend.resolve_date(idx, "2019-01-01")


@pytest.mark.parametrize("kwargs", [dict(amount=5), dict(amount=1e12),
                                    dict(amount=1000, max_weight=0.01),
                                    dict(amount=1000, exclude=["ZZZZ"]),
                                    dict(amount=1000, exclude=list(config.TICKERS))])
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        recommend.recommend(**kwargs)


def test_end_to_end_with_locked_policy():
    r = recommend.recommend(10_000, "2023-06-15", explanations=False)
    total = sum(h["weight"] for h in r["holdings"]) + r["cash"]["weight"]
    assert abs(total - 1.0) < 1e-6
    assert all(h["weight"] <= config.MAX_WEIGHT + 1e-6 for h in r["holdings"])
    assert abs(sum(h["amount"] for h in r["holdings"]) + r["cash"]["amount"] - 10_000) < 0.05
    assert r["in_sample"] is False and "drivers" not in r["holdings"][0]
    assert recommend.recommend(10_000, "2018-03-01", explanations=False)["in_sample"] is True
