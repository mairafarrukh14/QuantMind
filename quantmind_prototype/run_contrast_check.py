"""WCAG contrast check of the dashboard's text colours against its backgrounds.

Parses the colour variables in frontend/styles.css and reports the contrast ratio of
each text colour on each solid background, against the AA threshold of 4.5:1 for body
text. Writes results/contrast.json.

Run:  python run_contrast_check.py
"""
from __future__ import annotations

import json
import os
import re

from src import config

CSS = os.path.join(os.path.dirname(config.ROOT), "frontend", "styles.css")
OUT = os.path.join(config.ROOT, "results", "contrast.json")
TEXT = ["--text", "--text-2", "--text-3", "--pos", "--neg", "--warn"]
BACK = ["--bg-0", "--bg-1"]
AA = 4.5


def _lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_: str) -> float:
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def ratio(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def main() -> int:
    css = open(CSS, encoding="utf-8").read()
    var = {m.group(1): m.group(2) for m in re.finditer(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", css)}
    rows = [{"text": t, "colour": var[t], "background": b, "ratio": round(ratio(var[t], var[b]), 2),
             "passes_AA": ratio(var[t], var[b]) >= AA} for t in TEXT for b in BACK]
    out = {"threshold": AA, "rows": rows, "all_pass": all(r["passes_AA"] for r in rows),
           "min_ratio": min(r["ratio"] for r in rows)}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(f'{r["text"]:9s} {r["colour"]} on {r["background"]}: {r["ratio"]:5.2f} '
              f'{"ok" if r["passes_AA"] else "FAIL"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
