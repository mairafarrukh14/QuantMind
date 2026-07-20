"""QuantMind property test suite (Work Order W1).

Eight properties that turn the report's correctness claims into evidence:

  1. No look-ahead  - the reward at day t depends on t+1 returns, never t.
  2. Valid simplex  - allocations are non-negative and sum to 1 every step.
  3. Cap enforced   - no asset ever exceeds the 40% concentration limit.
  4. Cost accounting - the deducted cost equals rate x turnover exactly.
  5. Shapley closed form - the estimator matches analytic values on a linear f.
  6. Shapley efficiency  - contributions sum to f(x) - f(baseline).
  7. Sentiment no-leakage - a post-close headline never moves that day's score.
  8. Audit catches violations - corrupted rationales are rejected.

Properties 7 and 8 are red-green against the sentiment pipeline (W2) and the
rationale audit (W6): they auto-activate via ``importorskip`` the moment those
modules exist, and until then report as skipped in the exported pass table.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.env import PortfolioEnv, apply_position_cap
from src.explain import shapley_sampling


# --------------------------------------------------------------------------- #
#  1. No look-ahead                                                           #
# --------------------------------------------------------------------------- #
def test_no_lookahead(synth_market):
    """Reward at t reacts to returns at t+1, and is blind to returns at t.

    This pins the exact bug that once produced the impossible Sharpe ~ 8.5: if
    the reward ever read the same-day return the agent is choosing weights for,
    the backtest would be trading on information it could not have had.
    """
    features, log_rets = synth_market
    action = np.array([2.0, -1.0, 0.5, 0.0], dtype=np.float64)  # fixed, 3 assets + cash

    def first_reward(perturb_index: int | None):
        lr = log_rets.copy()
        if perturb_index is not None:
            lr.iloc[perturb_index] += 0.05  # shock every asset on that day
        env = PortfolioEnv(features, lr)
        env.reset()
        _, reward, *_ = env.step(action)  # this is the decision taken at t=0
        return reward

    base = first_reward(None)
    # Perturbing t+1 (index 1, the return actually earned by the t=0 decision).
    assert abs(first_reward(1) - base) > 1e-6, "reward ignored the t+1 return"
    # Perturbing t (index 0) must not move the t=0 reward: that return is never
    # earned by the agent, so leaking it would be look-ahead.
    assert first_reward(0) == pytest.approx(base, abs=1e-12), "reward leaked the t return"


# --------------------------------------------------------------------------- #
#  2. Valid simplex                                                           #
# --------------------------------------------------------------------------- #
def test_valid_simplex(synth_market, rng):
    """Every allocation the env emits is a valid probability simplex."""
    features, log_rets = synth_market
    env = PortfolioEnv(features, log_rets)
    env.reset()
    done = False
    while not done:
        action = rng.uniform(-8, 8, size=env.n_assets + 1)
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        for w in (info["target_weights"], env.weights):
            assert np.all(w >= -1e-9), "negative weight"
            assert w.sum() == pytest.approx(1.0, abs=1e-5), "weights do not sum to 1"


# --------------------------------------------------------------------------- #
#  3. Cap enforced                                                            #
# --------------------------------------------------------------------------- #
def test_cap_enforced(rng):
    """apply_position_cap never leaves an asset above the cap, even after
    redistribution and in the degenerate all-in-one-asset case."""
    cap, n_assets = config.MAX_WEIGHT, 5
    for _ in range(500):
        raw = rng.uniform(0, 1, size=n_assets + 1)
        raw = raw / raw.sum()
        capped = apply_position_cap(raw, cap, n_assets)
        assert np.all(capped[:n_assets] <= cap + 1e-6), "asset exceeds cap"
        assert capped.sum() == pytest.approx(1.0, abs=1e-5)
    # Degenerate: everything in one asset must be pulled back to the cap.
    spike = np.zeros(n_assets + 1)
    spike[0] = 1.0
    capped = apply_position_cap(spike, cap, n_assets)
    assert capped[0] <= cap + 1e-6


# --------------------------------------------------------------------------- #
#  4. Cost accounting                                                         #
# --------------------------------------------------------------------------- #
def test_cost_accounting(synth_market, rng):
    """The cost deducted from the return equals cost_rate x turnover exactly.

    We reconstruct the gross portfolio return from the reported target weights
    and the realised t+1 returns, and check net = gross - rate*turnover.
    """
    features, log_rets = synth_market
    env = PortfolioEnv(features, log_rets)
    env.reset()
    done = False
    while not done:
        t = env._t
        action = rng.uniform(-8, 8, size=env.n_assets + 1)
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        target = info["target_weights"]
        asset_rets = env.simple_rets[t + 1]
        gross = float(np.dot(target[:-1], asset_rets)) + target[-1] * config.RISK_FREE_DAILY
        expected_net = gross - config.TRANSACTION_COST * info["turnover"]
        assert info["port_ret"] == pytest.approx(expected_net, abs=1e-9)


# --------------------------------------------------------------------------- #
#  Shapley helpers                                                            #
# --------------------------------------------------------------------------- #
def _linear_fn(w):
    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return (X @ w).reshape(-1, 1)
    return f


# --------------------------------------------------------------------------- #
#  5. Shapley closed form                                                     #
# --------------------------------------------------------------------------- #
def test_shapley_closed_form(rng):
    """For a linear function the Shapley value of feature i is
    w_i * (x_i - E_background[x_i]); the sampler must match it to < 0.01."""
    M = 8
    w = rng.uniform(-1.0, 1.0, size=M)
    x = rng.uniform(-1.0, 1.0, size=M)
    background = rng.uniform(-1.0, 1.0, size=(6, M))
    f = _linear_fn(w)

    phi = shapley_sampling(f, x, background, n_perm=6000,
                           rng=np.random.default_rng(0))[:, 0]
    closed = w * (x - background.mean(axis=0))
    assert np.max(np.abs(phi - closed)) < 0.01


# --------------------------------------------------------------------------- #
#  6. Shapley efficiency                                                      #
# --------------------------------------------------------------------------- #
def test_shapley_efficiency(rng):
    """Contributions sum to f(x) - f(baseline). With a single-row background
    the telescoping identity holds exactly, independent of the function."""
    M = 8
    x = rng.uniform(-1.0, 1.0, size=M)
    background = rng.uniform(-1.0, 1.0, size=(1, M))

    def f(X):  # a genuinely non-linear function
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.tanh(X).sum(axis=1, keepdims=True)

    phi = shapley_sampling(f, x, background, n_perm=200,
                           rng=np.random.default_rng(0))[:, 0]
    lhs = phi.sum()
    rhs = float(f(x.reshape(1, -1))[0, 0] - f(background)[0, 0])
    # Exact in real arithmetic; the estimator reveals features through a float32
    # chain, so the telescoping sum matches to float32 precision (~1e-8).
    assert lhs == pytest.approx(rhs, abs=1e-5)


# --------------------------------------------------------------------------- #
#  7. Sentiment no-leakage  (red-green against W2)                            #
# --------------------------------------------------------------------------- #
def test_sentiment_no_leakage():
    """A headline time-stamped after day t's close must never change day t's
    sentiment score. Auto-activates once the W2 pipeline exists."""
    sentiment = pytest.importorskip(
        "src.sentiment", reason="W2 sentiment pipeline not built yet")
    import datetime as dt

    # A deterministic stub scorer keeps the test hermetic: it checks the leakage
    # filter, not FinBERT itself (that is exercised by the cache builder).
    stub = lambda texts: np.array([len(t) for t in texts], dtype=float)

    day = dt.date(2021, 6, 15)
    before = [("A0", dt.datetime(2021, 6, 15, 9, 30), "shares surge on strong earnings")]
    after = before + [("A0", dt.datetime(2021, 6, 15, 18, 0), "shares crash after hours")]

    score_before = sentiment.score_asset_day("A0", day, before, close_hour=16, scorer=stub)
    score_after = sentiment.score_asset_day("A0", day, after, close_hour=16, scorer=stub)
    assert score_before == pytest.approx(score_after), "post-close headline leaked"
    # And the pre-close headline must actually register (guards against a filter
    # that rejects everything and looks leakage-safe by accident).
    assert score_before != 0.0


# --------------------------------------------------------------------------- #
#  8. Audit catches violations  (red-green against W6)                        #
# --------------------------------------------------------------------------- #
def test_audit_catches_violations():
    """The rationale audit rejects a sentence with a wrong sign, an invented
    number, or an unlisted driver. Auto-activates once the W6 audit exists."""
    audit = pytest.importorskip(
        "src.audit", reason="W6 rationale audit not built yet")

    drivers = [
        {"feature": "ret_5d", "sign": +1, "weight": 0.30},
        {"feature": "rsi_14", "sign": -1, "weight": 0.30},
    ]
    good = "Momentum (5-day return) increased the weight while RSI reduced it."
    assert audit.audit_sentence(good, drivers).passed

    wrong_sign = "Momentum (5-day return) reduced the weight while RSI reduced it."
    invented_number = "Momentum drove the weight to 87% of the portfolio."
    unlisted_driver = "MACD trend strength increased the weight."
    for bad in (wrong_sign, invented_number, unlisted_driver):
        assert not audit.audit_sentence(bad, drivers).passed, f"audit passed: {bad!r}"
