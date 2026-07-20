"""W5 - explanation faithfulness by deletion (contribution C2).

Turns "the Shapley code is validated" into "the explanations are faithful". For
a sample of (day, holding) decisions we rank the input features by their Shapley
attribution for that holding, then delete them (set to a background baseline) in
that order and watch how fast the policy's weight on the holding falls -- against
deleting in random order. If the attribution is faithful, its top features should
matter most, so attribution-ordered deletion degrades the weight faster than
random. The gap between the two curves (area between curves) is the statistic;
the figure and statistic are exported for the evaluation chapter.

Answers the plausibility-versus-faithfulness doubt (Jacovi and Goldberg 2020)
with the deletion procedure of Samek et al. (2017).

Run (after run_experiments_w3.py):  python run_faithfulness.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src import config, data, explain

OUT = config.RESULTS_DIR
N_PAIRS = 120
N_PERM = 64
SEED = 0


def _best_price_model():
    """Load the best price-only seed's locked model."""
    import glob
    best_path, best_sharpe = None, -np.inf
    for mp in glob.glob(os.path.join(OUT, "runs", "price_seed*", "metrics.json")):
        with open(mp) as fh:
            s = json.load(fh)["metrics"]["Sharpe"]
        if s > best_sharpe:
            best_sharpe, best_path = s, os.path.join(os.path.dirname(mp), "model")
    return PPO.load(best_path), best_path


def main() -> int:
    model, path = _best_price_model()
    print(f"loaded model {path}", flush=True)
    prices, _ = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]

    f = explain.make_policy_fn(model)
    n_assets = len(config.TICKERS)

    # Background baseline = mean training observation (the "absent" reference).
    bg_obs = explain._observations(model, tr_feat, tr_rets)
    baseline = bg_obs.mean(axis=0)
    background = bg_obs[np.random.default_rng(SEED).choice(
        len(bg_obs), size=min(30, len(bg_obs)), replace=False)]

    test_obs = explain._observations(model, te_feat, te_rets)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(test_obs), size=min(N_PAIRS, len(test_obs)), replace=False)

    M = test_obs.shape[1]
    attr_curves, rand_curves = [], []
    for i in sel:
        x = test_obs[i]
        w0 = f(x.reshape(1, -1))[0]
        holding = int(np.argmax(w0[:n_assets]))   # the explained holding
        if w0[holding] < 1e-3:
            continue
        phi = explain.shapley_sampling(f, x, background, n_perm=N_PERM,
                                       rng=np.random.default_rng(int(i)))[:, holding]
        order_attr = np.argsort(np.abs(phi))[::-1]      # most important first
        order_rand = rng.permutation(M)

        def _curve(order):
            weights = [w0[holding]]
            xx = x.copy()
            for k in order:
                xx[k] = baseline[k]
                weights.append(f(xx.reshape(1, -1))[0][holding])
            return np.array(weights)

        attr_curves.append(_curve(order_attr))
        rand_curves.append(_curve(order_rand))

    attr = np.mean(attr_curves, axis=0)
    rand = np.mean(rand_curves, axis=0)
    frac = np.arange(M + 1) / M
    # Area between curves: how much faster attribution-ordered deletion drops the
    # weight than random. Positive => attribution is faithful. Trapezoid rule,
    # written out so it works across NumPy versions.
    diff = rand - attr
    abc = float(np.sum((diff[:-1] + diff[1:]) / 2 * np.diff(frac)))

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(frac, attr, label="attribution-ordered deletion", linewidth=2)
    plt.plot(frac, rand, label="random-order deletion", linewidth=1.6, linestyle="--")
    plt.xlabel("fraction of features deleted")
    plt.ylabel("policy weight on the explained holding")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT, "deletion_faithfulness.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    out = {"n_decisions": len(attr_curves), "n_perm": N_PERM,
           "area_between_curves": abc,
           "attr_curve": attr.tolist(), "random_curve": rand.tolist(),
           "faithful": bool(abc > 0)}
    with open(os.path.join(OUT, "faithfulness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"area between curves = {abc:.4f}  ({'faithful' if abc > 0 else 'NOT faithful'}); "
          f"n={len(attr_curves)} decisions", flush=True)
    print(f"[done] wrote deletion_faithfulness.png, faithfulness.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
