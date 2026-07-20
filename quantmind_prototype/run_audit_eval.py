"""W6 audit evaluation: the language-model rationale pass-rate and failure taxonomy.

Over >=100 sampled test decisions, we build the Shapley-grounded drivers for the
agent's top holding, ask the local model to rephrase them, and audit the result.
We report how often the model's sentence passed the audit unaided (the pass-rate),
how often it took a retry, how often it fell back to the template, and a taxonomy
of why sentences failed -- a headline evaluation metric for contribution C3.

Needs Ollama running with the model pulled. Run:  python run_audit_eval.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from src import config, data, explain, rationale
from src.explain import FEATURE_PHRASES

OUT = config.RESULTS_DIR
N_DECISIONS = 100
N_PERM = 48
TOP_K = 3


def _best_price_model():
    best_path, best = None, -1e9
    for mp in glob.glob(os.path.join(OUT, "runs", "price_seed*", "metrics.json")):
        with open(mp) as fh:
            s = json.load(fh)["metrics"]["Sharpe"]
        if s > best:
            best, best_path = s, os.path.join(os.path.dirname(mp), "model")
    return PPO.load(best_path)


def main() -> int:
    model = _best_price_model()
    prices, _ = data.load_prices()
    features, log_rets = data.build_features(prices)
    split = data.train_test_split(prices, features, log_rets)
    tr_feat, tr_rets, _ = split["train"]
    te_feat, te_rets, _ = split["test"]
    tickers = list(te_rets.columns)
    feat_names = explain.build_feature_names(tickers)
    f = explain.make_policy_fn(model)
    n_assets = len(tickers)

    bg = explain._observations(model, tr_feat, tr_rets)
    background = bg[np.random.default_rng(0).choice(len(bg), size=min(25, len(bg)),
                                                    replace=False)]
    test_obs = explain._observations(model, te_feat, te_rets)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(test_obs), size=min(N_DECISIONS, len(test_obs)), replace=False)

    records = []
    for n, i in enumerate(sel):
        x = test_obs[i]
        w = f(x.reshape(1, -1))[0]
        holding = int(np.argmax(w[:n_assets]))
        label = tickers[holding]
        phi = explain.shapley_sampling(f, x, background, n_perm=N_PERM,
                                       rng=np.random.default_rng(int(i)))[:, holding]
        own = [k for k, nm in enumerate(feat_names) if nm.startswith(f"{label}:")]
        ranked = sorted(own, key=lambda k: abs(phi[k]), reverse=True)[:TOP_K]
        drivers = [{"feature": feat_names[k].split(":")[1],
                    "sign": 1 if phi[k] > 0 else -1,
                    "phrase": FEATURE_PHRASES.get(feat_names[k].split(":")[1],
                                                  feat_names[k].split(":")[1])}
                   for k in ranked]
        res = rationale.generate_rationale(label, float(w[holding]), drivers)
        records.append(res)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(sel)} decisions...", flush=True)

    n_total = len(records)
    n_llm = sum(r["source"] == "llm" for r in records)
    n_pass1 = sum(r["source"] == "llm" and r["attempts"] == 1 for r in records)
    n_template = sum(r["source"] == "template" for r in records)
    taxonomy = Counter()
    for r in records:
        for reason in r["fail_reasons"]:
            key = reason.split(":")[0].split("(")[0].strip()
            taxonomy[key] += 1

    out = {
        "n_decisions": n_total,
        "model": rationale.MODEL,
        "pass_rate": round(n_llm / n_total, 3),
        "pass_rate_first_try": round(n_pass1 / n_total, 3),
        "template_fallback_rate": round(n_template / n_total, 3),
        "failure_taxonomy": dict(taxonomy),
    }
    with open(os.path.join(OUT, "audit_eval.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\npass-rate (LLM sentence audited OK): {out['pass_rate']:.1%}  "
          f"(first try {out['pass_rate_first_try']:.1%}); "
          f"template fallback {out['template_fallback_rate']:.1%}")
    print(f"failure taxonomy: {dict(taxonomy)}")
    print("[done] wrote audit_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
