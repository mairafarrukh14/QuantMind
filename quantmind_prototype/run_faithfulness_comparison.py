"""Faithfulness with uncertainty, and against an established alternative.

Re-derives the same 120 sampled decisions as run_faithfulness.py (same seeds, so the
Shapley deletion area must come out identical, which is checked) and adds:

  * a bootstrap 95% interval on the deletion area, over decisions;
  * LIME (Ribeiro et al., 2016) written from scratch as the comparator: a
    kernel-weighted ridge regression on binary keep/replace masks around the
    decision, given the same model-evaluation budget as the Shapley estimate;
  * the same deletion test applied to LIME's ranking, a paired bootstrap of the
    difference, and how far the two rankings agree (top-3 overlap, Spearman).

Reads the locked best price-only model; nothing is retrained. Writes
results/faithfulness_comparison.json.

Run:  python run_faithfulness_comparison.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3 import PPO

from run_faithfulness import N_PAIRS, N_PERM, SEED, _best_price_model
from src import config, data, explain

OUT = config.RESULTS_DIR
N_BOOT = 10_000


def lime_attribution(f, x, baseline, holding, n_samples, rng, alpha=1.0):
    """Weighted ridge regression of f(x_masked)[holding] on the keep-mask.

    A mask entry of 1 keeps the feature at x, 0 replaces it with the baseline.
    Kernel width follows LIME's tabular default, 0.75 * sqrt(number of features).
    """
    M = x.shape[0]
    Z = rng.integers(0, 2, size=(n_samples, M)).astype(np.float64)
    Z[0] = 1.0                                             # the instance itself
    X = np.where(Z > 0, x[None, :], baseline[None, :]).astype(np.float32)
    y = f(X)[:, holding].astype(np.float64)
    d = np.sqrt((1.0 - Z).sum(axis=1))
    w = np.exp(-(d ** 2) / (0.75 * np.sqrt(M)) ** 2)
    Zc = np.hstack([Z, np.ones((n_samples, 1))])
    A = Zc.T @ (Zc * w[:, None]) + alpha * np.eye(M + 1)
    coef = np.linalg.solve(A, Zc.T @ (w * y))
    return coef[:M]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ci(v, rng):
    v = np.asarray(v)
    m = np.array([rng.choice(v, len(v)).mean() for _ in range(N_BOOT)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    model, path = _best_price_model()
    prices, _ = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]
    f = explain.make_policy_fn(model)
    n_assets = len(config.TICKERS)

    bg_obs = explain._observations(model, tr_feat, tr_rets)
    baseline = bg_obs.mean(axis=0)
    background = bg_obs[np.random.default_rng(SEED).choice(
        len(bg_obs), size=min(30, len(bg_obs)), replace=False)]
    test_obs = explain._observations(model, te_feat, te_rets)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(test_obs), size=min(N_PAIRS, len(test_obs)), replace=False)

    M = test_obs.shape[1]
    lime_budget = N_PERM * (M + 1)          # same number of model evaluations as Shapley
    frac = np.arange(M + 1) / M
    area_shap, area_lime, top3, rho = [], [], [], []
    for i in sel:
        x = test_obs[i]
        w0 = f(x.reshape(1, -1))[0]
        holding = int(np.argmax(w0[:n_assets]))
        if w0[holding] < 1e-3:
            continue
        phi = explain.shapley_sampling(f, x, background, n_perm=N_PERM,
                                       rng=np.random.default_rng(int(i)))[:, holding]
        order_shap = np.argsort(np.abs(phi))[::-1]
        order_rand = rng.permutation(M)                 # same consumption order as the locked run
        lime = lime_attribution(f, x, baseline, holding, lime_budget,
                                np.random.default_rng(int(i) + 1))
        order_lime = np.argsort(np.abs(lime))[::-1]

        def curve(order):
            ws, xx = [w0[holding]], x.copy()
            for k in order:
                xx[k] = baseline[k]
                ws.append(f(xx.reshape(1, -1))[0][holding])
            return np.array(ws)

        def area(c_rand, c_attr):
            d = c_rand - c_attr
            return float(np.sum((d[:-1] + d[1:]) / 2 * np.diff(frac)))

        c_rand = curve(order_rand)
        area_shap.append(area(c_rand, curve(order_shap)))
        area_lime.append(area(c_rand, curve(order_lime)))
        top3.append(len(set(order_shap[:3]) & set(order_lime[:3])))
        rho.append(spearman(np.abs(phi), np.abs(lime)))

    rb = np.random.default_rng(1)
    a_s, a_l = np.array(area_shap), np.array(area_lime)
    out = {
        "model": os.path.basename(os.path.dirname(path)), "n_decisions": len(a_s),
        "shapley_budget_evals": lime_budget, "lime_budget_evals": lime_budget,
        "shapley_area_mean": float(a_s.mean()), "shapley_area_ci95": boot_ci(a_s, rb),
        "lime_area_mean": float(a_l.mean()), "lime_area_ci95": boot_ci(a_l, rb),
        "shapley_minus_lime_mean": float((a_s - a_l).mean()),
        "shapley_minus_lime_ci95": boot_ci(a_s - a_l, rb),
        "top3_overlap_mean": float(np.mean(top3)),
        "top3_overlap_if_random": 3 * 3 / M,
        "spearman_abs_attribution_mean": float(np.mean(rho)),
        "reproduces_locked_area": bool(abs(a_s.mean() - json.load(open(
            os.path.join(OUT, "faithfulness.json")))["area_between_curves"]) < 1e-6),
    }
    with open(os.path.join(OUT, "faithfulness_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
