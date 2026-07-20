"""Explainability layer for QuantMind (the XAI in 'Explainable AI advisor').

We treat the trained PPO policy as a black-box function
    f : observation -> target portfolio weights (softmax of the policy logits)
and attribute each allocation decision back to the input features using
**Shapley-value sampling** (Strumbelj & Kononenko, 2014) -- the model-agnostic
estimator that underlies SHAP (Lundberg & Lee, 2017) and generalises the local
explanation idea of LIME (Ribeiro et al., 2016).

It is implemented here directly in NumPy (no `shap`/`numba` dependency), which
keeps the prototype lightweight and makes the algorithm fully transparent for
the report. For each feature i the Shapley value is the average marginal change
in the output as i is added to a random coalition of features, with absent
features sampled from a background distribution:

    phi_i = E_perm,bg [ f(x_{S U {i}}) - f(x_{S}) ]

By construction sum_i phi_i = f(x) - f(background)  (the efficiency axiom).

Two products:
  1. A *global* feature-importance chart (mean |phi| over the test set).
  2. A *local*, per-decision natural-language rationale for the asset the agent
     most favoured on a chosen day -- the plain-English explanation a
     non-technical QuantMind user would see.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

# Human-readable phrasing for each feature, for the NL rationale.
FEATURE_PHRASES = {
    "ret_1d": "recent 1-day return",
    "ret_5d": "5-day momentum",
    "rsi_14": "RSI (overbought/oversold)",
    "macd_hist": "MACD trend strength",
    "vol_20": "20-day volatility",
    "px_sma20": "price vs its 20-day average",
    "sma5_sma20": "short- vs medium-term trend",
}


def build_feature_names(tickers) -> list[str]:
    names = [f"{t}:{f}" for t in tickers for f in config.FEATURE_NAMES]
    names += [f"w_{t}" for t in tickers] + ["w_CASH"]
    return names


def make_policy_fn(model):
    """Return f(X)->weights matrix (n, n_assets+1), batched, for attribution.

    Applies the same softmax + concentration cap the environment uses, so the
    explained weights match the agent's actual allocations.
    """
    from .env import apply_position_cap
    n_assets = len(config.TICKERS)

    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        actions, _ = model.predict(X, deterministic=True)
        z = actions - actions.max(axis=1, keepdims=True)
        e = np.exp(z)
        w = e / (e.sum(axis=1, keepdims=True) + 1e-12)
        return np.array([apply_position_cap(row, config.MAX_WEIGHT, n_assets) for row in w])
    return f


# --------------------------------------------------------------------------- #
#  Shapley-value sampling (the core XAI algorithm)                            #
# --------------------------------------------------------------------------- #
def shapley_sampling(f, x, background, n_perm=64, rng=None):
    """Estimate Shapley values of every input feature for every output.

    Args:
        f: batched function  X (n, M) -> Y (n, K)
        x: single instance (M,)
        background: reference set (B, M) for absent features
        n_perm: number of random permutations to average over
    Returns:
        phi: (M, K) Shapley values; sum_M phi == f(x) - mean_B f(background).
    """
    rng = rng or np.random.default_rng(config.SEED)
    x = np.asarray(x, dtype=np.float32)
    M = x.shape[0]
    K = f(x.reshape(1, -1)).shape[1]
    phi = np.zeros((M, K), dtype=np.float64)

    for _ in range(n_perm):
        order = rng.permutation(M)
        bg = background[rng.integers(len(background))]      # one reference draw

        # Build the chain of M+1 synthetic inputs: start from background, then
        # reveal features one-by-one in `order` until we reach x. Batch them.
        chain = np.tile(bg.astype(np.float32), (M + 1, 1))
        revealed = np.zeros(M, dtype=bool)
        for step, feat in enumerate(order):
            revealed[feat] = True
            chain[step + 1, revealed] = x[revealed]
        preds = f(chain)                                    # (M+1, K)
        marginal = preds[1:] - preds[:-1]                   # (M, K) in reveal order
        # Scatter each marginal back to its feature index.
        phi[order] += marginal

    return phi / n_perm


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def _observations(model, features, log_rets):
    """Replay the policy through the env to collect the obs it actually sees."""
    from .env import PortfolioEnv
    env = PortfolioEnv(features, log_rets)
    obs, _ = env.reset()
    out = [obs]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
        if not done:
            out.append(obs)
    return np.array(out, dtype=np.float32)


def explain(model, train_features, train_rets, test_features, test_rets,
            n_background=30, n_explain=40, n_perm=64, seed=config.SEED):
    """Compute Shapley attributions over a sample of test decisions.

    Returns (phi_all, X_explain, sel, feature_names, tickers) where
    phi_all is (n_explain, M, K).
    """
    rng = np.random.default_rng(seed)
    tickers = list(test_rets.columns)
    feat_names = build_feature_names(tickers)
    f = make_policy_fn(model)

    bg_all = _observations(model, train_features, train_rets)
    background = bg_all[rng.choice(len(bg_all), size=min(n_background, len(bg_all)), replace=False)]

    test_obs = _observations(model, test_features, test_rets)
    sel = np.sort(rng.choice(len(test_obs), size=min(n_explain, len(test_obs)), replace=False))
    X_explain = test_obs[sel]

    phi_all = np.array([shapley_sampling(f, x, background, n_perm=n_perm, rng=rng)
                        for x in X_explain])               # (n_explain, M, K)

    _plot_global_importance(phi_all, feat_names)
    return phi_all, X_explain, sel, feat_names, tickers


def _plot_global_importance(phi_all, feat_names, top_k=15):
    # Mean |phi| across explained instances and output weights.
    importance = np.abs(phi_all).mean(axis=(0, 2))         # (M,)
    order = np.argsort(importance)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feat_names[i] for i in order][::-1], importance[order][::-1], color="#3b6ea5")
    ax.set_xlabel("Mean |Shapley value| (impact on allocation)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "shap_importance.png"), dpi=150)
    plt.close(fig)


def narrate_decision(model, day_obs, tickers, phi_for_day, feat_names):
    """Plain-English rationale for the agent's top pick on one day.

    This is the 'LLM narration' slot in the QuantMind design; here it is a
    deterministic template grounded in the Shapley attributions (a local
    LLaMA-3 model would phrase the same facts in the full system).
    """
    f = make_policy_fn(model)
    weights = f(day_obs.reshape(1, -1))[0]
    labels = tickers + ["CASH"]
    top_idx = int(np.argmax(weights))
    top_label = labels[top_idx]

    lines = [
        f"QuantMind allocates most to {top_label} "
        f"({weights[top_idx] * 100:.0f}% of the portfolio).",
    ]
    if top_idx < len(tickers):
        contribs = phi_for_day[:, top_idx]                 # (M,) drivers of this asset
        own = [i for i, n in enumerate(feat_names) if n.startswith(f"{top_label}:")]
        ranked = sorted(own, key=lambda i: abs(contribs[i]), reverse=True)[:3]
        reasons = []
        for i in ranked:
            feat = feat_names[i].split(":")[1]
            direction = "increased" if contribs[i] > 0 else "reduced"
            reasons.append(f"its {FEATURE_PHRASES.get(feat, feat)} {direction} the weight")
        lines.append("Main drivers: " + "; ".join(reasons) + ".")
    else:
        lines.append("The agent is risk-off and is holding cash.")
    return "\n".join(lines)
