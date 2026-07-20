"""W8 - artifact reconciliation: one source of truth, three surfaces.

Grounds every headline number in the locked best-seed run, then checks that the
dashboard payload and the evaluation summary agree with it. Resolves the named
defect (Max Drawdown shown as -13.7% in one place and -13.8% in another) by taking
the single artifact value and reporting it. Writes results/reconciliation.json --
the canonical headline set the report, the figures and the dashboard must all match.

Run:  python run_reconcile.py
"""
from __future__ import annotations

import json
import os

from src import config

OUT = config.RESULTS_DIR
BEST_SEED_RUN = "price_seed42"   # the chosen best-seed (highest price-only Sharpe)
HEADLINE = ["Total Return", "CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown"]


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    truth = _load(os.path.join(OUT, "runs", BEST_SEED_RUN, "metrics.json"))["metrics"]
    canonical = {k: round(truth[k], 4) for k in HEADLINE}
    canonical["Turnover"] = round(truth.get("Turnover", 0.0), 4)
    canonical["Portfolio value"] = round(config.INITIAL_CASH * (1 + truth["Total Return"]), 2)

    mismatches = []

    # Surface 1: dashboard payload.
    payload = _load(os.path.join(OUT, "frontend_payload.json"))
    k = payload["kpis"]
    checks = {
        "dashboard.sharpe": (round(k["sharpe"], 4), canonical["Sharpe"]),
        "dashboard.total_return": (round(k["total_return"], 4), canonical["Total Return"]),
        "dashboard.max_drawdown": (round(k["max_drawdown"], 4), canonical["Max Drawdown"]),
        "dashboard.cagr": (round(k["cagr"], 4), canonical["CAGR"]),
    }
    # Surface 2: evaluation summary (best-seed row).
    ev = _load(os.path.join(OUT, "evaluation.json"))["main_table"]
    best_row = next((v for name, v in ev.items() if "best seed" in name and "sentiment" not in name), {})
    if best_row:
        checks["eval.sharpe"] = (round(best_row["Sharpe"], 4), canonical["Sharpe"])
        checks["eval.max_drawdown"] = (round(best_row["Max Drawdown"], 4), canonical["Max Drawdown"])

    for name, (got, want) in checks.items():
        if abs(got - want) > 5e-4:
            mismatches.append({"surface": name, "value": got, "canonical": want})

    out = {
        "best_seed_run": BEST_SEED_RUN,
        "canonical": canonical,
        # Format from the RAW artifact values (not the 4-dp-rounded copies) to
        # avoid double-rounding flipping a boundary value like -13.7497%.
        "canonical_pct": {
            "Total Return": f"{truth['Total Return']*100:.1f}%",
            "CAGR": f"{truth['CAGR']*100:.1f}%",
            "Ann. Volatility": f"{truth['Ann. Volatility']*100:.1f}%",
            "Sharpe": f"{truth['Sharpe']:.2f}",
            "Sortino": f"{truth['Sortino']:.2f}",
            "Max Drawdown": f"{truth['Max Drawdown']*100:.1f}%",
        },
        "surfaces_checked": list(checks.keys()),
        "mismatches": mismatches,
        "clean": not mismatches,
    }
    with open(os.path.join(OUT, "reconciliation.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("Canonical headline set (best-seed run", BEST_SEED_RUN + "):")
    for kk, vv in out["canonical_pct"].items():
        print(f"  {kk:16s} {vv}")
    print(f"  Portfolio value  £{canonical['Portfolio value']:,.0f}")
    print(f"\nsurfaces checked: {len(checks)}; mismatches: {len(mismatches)}")
    for m in mismatches:
        print(f"  ! {m['surface']}: {m['value']} vs canonical {m['canonical']}")
    print("[done] wrote reconciliation.json" + ("  (ZERO mismatches)" if not mismatches else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
