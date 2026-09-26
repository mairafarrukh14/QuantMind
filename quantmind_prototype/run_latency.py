"""Serve-time latency of the live recommendation path, and test-suite cold start.

The recommendation path calls the local language model whenever Ollama is running
(the template is only a fallback), so a single latency figure conflates two very
different cases. This script detects which case it is running under from the
actual model call --Ollama up or down-- and writes that half of results/latency.json,
so running it twice (once with Ollama stopped, once with it serving) fills in both
`template_path` and `with_llm` without either run overwriting the other's numbers.

Run:  python run_latency.py           # once with `ollama stop`/no server running
      python run_latency.py           # once with Ollama serving llama3.2:1b
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
from src.rationale import _ollama_generate

OUT = os.path.join(config.ROOT, "results", "latency.json")


def _ollama_up() -> bool:
    return _ollama_generate("ping", timeout=15) is not None


def _time_recommend(dates: list[str]) -> dict:
    secs = []
    for d in dates:
        t = time.perf_counter()
        recommend.recommend(5000, d, explanations=True)
        secs.append(time.perf_counter() - t)
    return {"requests": len(secs), "median_s": round(float(np.median(secs)), 2),
            "p95_s": round(float(np.percentile(secs, 95)), 2), "max_s": round(float(max(secs)), 2)}


def main() -> int:
    t0 = time.perf_counter()
    recommend.date_range()                                   # loads the policy once
    load_s = time.perf_counter() - t0
    dates = ["2022-03-01", "2022-06-15", "2022-09-30", "2022-12-15", "2023-02-28",
             "2023-05-10", "2023-08-01", "2023-10-20", "2023-12-29", "2024-02-15",
             "2024-03-28", "2024-05-16", "2024-06-27", "2024-07-31", "2024-08-30",
             "2024-09-26", "2024-10-24", "2024-11-14", "2024-12-05", "2024-12-30"]

    up = _ollama_up()
    print(f"Ollama reachable: {up} -> timing the {'with-LLM' if up else 'template'} path", flush=True)
    timing = _time_recommend(dates)

    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       cwd=config.ROOT, capture_output=True, text=True)
    suite_s = time.perf_counter() - t

    out = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            out = json.load(fh)
    out.update({"machine": f"{platform.machine()} {platform.system()}", "python": platform.python_version(),
               "policy_load_s": round(load_s, 2), "threshold_median_s": 5.0,
               "test_suite_s": round(suite_s, 1),
               "test_suite_n": int(__import__("re").search(r"(\d+) passed", r.stdout).group(1)) if "passed" in r.stdout else 0,
               "test_suite_passed": r.returncode == 0,
               "test_summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""})
    key = "with_llm" if up else "template_path"
    out[key] = timing
    out["latency_pass"] = all(v["median_s"] < 5.0 for k, v in out.items()
                              if k in ("template_path", "with_llm"))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
