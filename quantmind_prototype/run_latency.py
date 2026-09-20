"""Serve-time latency of the live recommendation path, and test-suite cold start.

Times 20 recommendations with explanations (different as-of dates, so no request
repeats another) after the policy has been loaded once, in process on the machine
that runs it, and times a full run of the test suite. Writes results/latency.json.

Run:  python run_latency.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from src import config, recommend

OUT = os.path.join(config.ROOT, "results", "latency.json")


def main() -> int:
    t0 = time.perf_counter()
    recommend.date_range()                                   # loads the policy once
    load_s = time.perf_counter() - t0
    dates = ["2022-03-01", "2022-06-15", "2022-09-30", "2022-12-15", "2023-02-28",
             "2023-05-10", "2023-08-01", "2023-10-20", "2023-12-29", "2024-02-15",
             "2024-03-28", "2024-05-16", "2024-06-27", "2024-07-31", "2024-08-30",
             "2024-09-26", "2024-10-24", "2024-11-14", "2024-12-05", "2024-12-30"]
    secs = []
    for d in dates:
        t = time.perf_counter()
        recommend.recommend(5000, d, explanations=True)
        secs.append(time.perf_counter() - t)
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       cwd=config.ROOT, capture_output=True, text=True)
    suite_s = time.perf_counter() - t
    out = {"machine": f"{platform.machine()} {platform.system()}", "python": platform.python_version(),
           "requests": len(secs), "policy_load_s": round(load_s, 2),
           "median_s": round(float(np.median(secs)), 2),
           "p95_s": round(float(np.percentile(secs, 95)), 2), "max_s": round(float(max(secs)), 2),
           "threshold_median_s": 5.0, "latency_pass": bool(np.median(secs) < 5.0),
           "test_suite_s": round(suite_s, 1),
           "test_suite_n": int(__import__("re").search(r"(\d+) passed", r.stdout).group(1)) if "passed" in r.stdout else 0, "test_suite_passed": r.returncode == 0,
           "test_summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
