"""Shared fixtures and a pass-table exporter for the QuantMind property suite.

The suite is deliberately hermetic: every test builds its own small synthetic
environment or synthetic function, so `pytest` runs in a few seconds on a fresh
clone with no trained model, no network and no downloaded weights. The eight
properties are the correctness claims the report makes, converted from assertion
into evidence (Work Order W1).

On session end this file writes ``results/property_tests.json`` and
``results/property_tests.csv`` — the pass table Chapter 5 reads.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# The project is imported as ``from src import ...`` with the prototype root on
# the path (the same convention the run_experiments_* scripts use).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import config  # noqa: E402


# --------------------------------------------------------------------------- #
#  Synthetic data fixtures (no trained model needed)                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def rng():
    return np.random.default_rng(20240607)


@pytest.fixture
def synth_market(rng):
    """A small, well-formed (features, log_rets) pair for the environment.

    T days, n_assets assets, the project's 7 technical features. Returns are
    modest daily log returns so the simplex/cap/cost invariants are exercised
    without numerical extremes.
    """
    import pandas as pd

    T, n_assets, n_features = 60, 3, len(config.FEATURE_NAMES)
    features = rng.standard_normal((T, n_assets, n_features)).astype(np.float32)
    log_rets = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(T, n_assets)),
        index=pd.bdate_range("2020-01-01", periods=T),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    return features, log_rets


# --------------------------------------------------------------------------- #
#  Pass-table exporter — the artifact Chapter 5 reads                         #
# --------------------------------------------------------------------------- #
# Human-readable property name per test function, keyed by the Work Order's
# numbering so the exported table matches the report's eight-property table.
PROPERTY_LABELS = {
    "test_no_lookahead": "1. No look-ahead in reward",
    "test_valid_simplex": "2. Valid simplex (weights >=0, sum to 1)",
    "test_cap_enforced": "3. Position cap enforced (<=40%)",
    "test_cost_accounting": "4. Cost = rate x turnover exactly",
    "test_shapley_closed_form": "5. Shapley matches closed form (<0.01)",
    "test_shapley_efficiency": "6. Shapley efficiency axiom",
    "test_sentiment_no_leakage": "7. Sentiment pre-close no-leakage",
    "test_audit_catches_violations": "8. Audit rejects corrupted rationales",
}

_OUTCOMES: dict[str, str] = {}


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    func = report.nodeid.split("::")[-1].split("[")[0]
    if func not in PROPERTY_LABELS:
        return
    if report.skipped:
        _OUTCOMES[func] = "skipped"
    elif report.passed:
        _OUTCOMES[func] = "passed"
    elif report.failed:
        _OUTCOMES[func] = "failed"


def pytest_sessionfinish(session, exitstatus):
    if not _OUTCOMES:
        return
    import csv
    import json

    rows = []
    for func, label in PROPERTY_LABELS.items():
        rows.append({
            "property": label,
            "result": _OUTCOMES.get(func, "not run"),
        })
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "property_tests.json"), "w") as fh:
        json.dump({"tests": rows,
                   "n_passed": sum(r["result"] == "passed" for r in rows),
                   "n_total": len(rows)}, fh, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "property_tests.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["property", "result"])
        w.writeheader()
        w.writerows(rows)
