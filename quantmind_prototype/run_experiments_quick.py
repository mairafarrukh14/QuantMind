"""Quick (no-training) experiments for the revision: cap evidence, Shapley
validation against a known reference, and the energy-reliance analysis.

Writes figures + a JSON summary to results/.
"""
from __future__ import annotations
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data, backtest, train, config, explain

OUT = config.RESULTS_DIR
summary = {}

# --------------------------------------------------------------------------- #
# Load data + model once
# --------------------------------------------------------------------------- #
prices, source = data.load_prices()
features, log_rets = data.build_features(prices)
split = data.train_test_split(prices, features, log_rets)
tr_feat, tr_rets, _ = split["train"]
te_feat, te_rets, _ = split["test"]
model = train.load_agent()
agent_rets, weights = backtest.run_agent(model, te_feat, te_rets)
asset_cols = [c for c in weights.columns if c != "CASH"]

# --------------------------------------------------------------------------- #
# A3 — concentration-cap evidence: per-asset weight lines + 40% cap line
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 4.5))
for c in asset_cols:
    ax.plot(weights.index, weights[c], linewidth=1.1, label=c)
ax.axhline(config.MAX_WEIGHT, color="black", linestyle="--", linewidth=1.3,
           label=f"{int(config.MAX_WEIGHT*100)}% cap")
ax.set_ylabel("Weight"); ax.set_xlabel("Date"); ax.set_ylim(0, 0.5)
ax.legend(ncol=6, fontsize=8, loc="upper center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cap_check.png"), dpi=150); plt.close(fig)

summary["cap_check"] = {
    "max_weight_per_asset": {c: float(weights[c].max()) for c in asset_cols},
    "cap": config.MAX_WEIGHT,
    "violations": int((weights[asset_cols] > config.MAX_WEIGHT + 1e-6).sum().sum()),
}
print("[A3] max weight per asset:", {c: round(weights[c].max(), 4) for c in asset_cols},
      "violations:", summary["cap_check"]["violations"])

# --------------------------------------------------------------------------- #
# E2 — validate from-scratch Shapley on a KNOWN linear function
#   For f(x) = w.x, the exact Shapley value of feature i w.r.t. a background
#   distribution is phi_i = w_i * (x_i - E_bg[x_i]).  We compare our sampler.
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
M = 8
w_true = rng.normal(size=M)
bg = rng.normal(size=(200, M))
x = rng.normal(size=M)

def f_lin(X):
    X = np.atleast_2d(X)
    return (X @ w_true).reshape(-1, 1)   # (n, 1)

phi_exact = w_true * (x - bg.mean(axis=0))                 # closed form
phi_hat = explain.shapley_sampling(f_lin, x, bg, n_perm=2000, rng=rng)[:, 0]
max_err = float(np.max(np.abs(phi_hat - phi_exact)))
# efficiency: sum of phi should equal f(x) - mean_bg f
eff_lhs = float(phi_hat.sum())
eff_rhs = float(f_lin(x.reshape(1, -1))[0, 0] - f_lin(bg).mean())
summary["shapley_validation"] = {
    "n_features": M, "n_perm": 2000,
    "max_abs_error_vs_closed_form": max_err,
    "efficiency_lhs_sum_phi": eff_lhs,
    "efficiency_rhs_fx_minus_base": eff_rhs,
    "efficiency_gap": abs(eff_lhs - eff_rhs),
}
print(f"[E2] Shapley vs closed-form: max abs error = {max_err:.4f}; "
      f"efficiency gap = {abs(eff_lhs-eff_rhs):.4e}")

# --------------------------------------------------------------------------- #
# E4 — energy-reliance analysis: is the XOM tilt specific to the 2022 spike?
# --------------------------------------------------------------------------- #
import pandas as pd
xom = weights["XOM"]
spike = xom[(xom.index >= "2022-01-01") & (xom.index < "2022-11-01")]   # energy/inflation
later = xom[xom.index >= "2023-01-01"]                                   # normalised
summary["energy_analysis"] = {
    "xom_mean_weight_2022_spike": float(spike.mean()),
    "xom_mean_weight_2023_onward": float(later.mean()),
    "xom_mean_weight_full_test": float(xom.mean()),
    "ratio_spike_to_later": float(spike.mean() / (later.mean() + 1e-9)),
}
print(f"[E4] XOM mean weight: 2022 spike={spike.mean():.3f}, "
      f"2023+={later.mean():.3f}, full={xom.mean():.3f}")

with open(os.path.join(OUT, "revision_quick.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print("\n[done] wrote results/cap_check.png and results/revision_quick.json")
