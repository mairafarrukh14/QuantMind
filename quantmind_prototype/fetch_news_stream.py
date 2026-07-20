"""Authenticated stream-filter of the FNSPID Nasdaq news file (stores nothing big).

The 23 GB Nasdaq CSV will not fit on disk, and hf_hub_download refuses to write
unless the full file size is free. With an authenticated connection the Hub does
not rate-limit, so we can stream the file through and keep only the rows for our
five tickers -- never storing the multi-GB source. The article column contains
embedded newlines, so we parse with ``csv.reader`` over the continuous stream
(which handles quoted newlines correctly) rather than splitting on lines.

Needs an HF token in the environment:  HF_TOKEN=hf_... python fetch_news_stream.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

import requests

from src import config

URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
       "Stock_news/nasdaq_exteral_data.csv")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
TICKERS = set(config.TICKERS)
START_YEAR, END_YEAR = int(config.START_DATE[:4]), int(config.END_DATE[:4])
OUT_DIR = os.path.join(config.DATA_DIR, "news")
OUT_PATH = os.path.join(OUT_DIR, "headlines_raw.csv")
BACKUP_ALL_EXTERNAL = os.path.join(OUT_DIR, "headlines_all_external.csv")


def _load_backup(rows: dict) -> None:
    if not os.path.exists(BACKUP_ALL_EXTERNAL):
        return
    with open(BACKUP_ALL_EXTERNAL, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["symbol"], r["timestamp_utc"], r["headline"])] = None
    print(f"seeded {len(rows):,} rows from All_external backup", flush=True)


def main() -> int:
    if not TOKEN:
        print("ERROR: set HF_TOKEN in the environment", file=sys.stderr)
        return 2
    os.makedirs(OUT_DIR, exist_ok=True)
    csv.field_size_limit(50_000_000)
    rows: dict = {}
    _load_backup(rows)

    print("streaming Nasdaq file (authenticated, no full download)...", flush=True)
    with requests.get(URL, headers={"Authorization": f"Bearer {TOKEN}"},
                      stream=True, timeout=(15, 300)) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        text = io.TextIOWrapper(resp.raw, encoding="utf-8", newline="")
        reader = csv.reader(text)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_date, i_title, i_sym = col["Date"], col["Article_title"], col["Stock_symbol"]

        seen = 0
        for row in reader:
            seen += 1
            if seen % 1_000_000 == 0:
                print(f"  {seen:,} rows scanned, kept {len(rows):,}", flush=True)
            if len(row) <= max(i_date, i_title, i_sym):
                continue
            sym = row[i_sym].strip().upper()
            if sym not in TICKERS:
                continue
            date = row[i_date].strip()
            if len(date) < 4 or not (str(START_YEAR) <= date[:4] <= str(END_YEAR)):
                continue
            title = row[i_title].strip()
            if title:
                rows[(sym, date, title)] = None

    per_ticker: dict[str, int] = {t: 0 for t in TICKERS}
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "timestamp_utc", "headline"])
        for (sym, date, title) in rows:
            writer.writerow([sym, date, title])
            per_ticker[sym] += 1

    print(f"DONE kept {len(rows):,} unique headlines", flush=True)
    for t in sorted(per_ticker):
        print(f"  {t}: {per_ticker[t]:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
