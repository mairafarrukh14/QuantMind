"""Download + filter the FNSPID financial-news dataset to the QuantMind universe.

FNSPID (Dong et al., 2024; https://huggingface.co/datasets/Zihan1004/FNSPID,
CC BY-NC-4.0) ships a 5.7 GB CSV of time-stamped headlines for S&P 500 names.
We fetch it once with the resumable Hub downloader (robust to the Hub's
unauthenticated rate-limiting, which stalls a plain HTTP stream), then keep only
the rows for our five tickers within the backtest window, writing a compact
``data/news/headlines_raw.csv``. The multi-GB source is a transient cache.

Schema (source): Date, Article_title, Stock_symbol, Url, Publisher, Author, ...
Output schema:   symbol, timestamp_utc, headline

Run:  python fetch_news.py
"""
from __future__ import annotations

import csv
import os
import sys

import shutil

from huggingface_hub import hf_hub_download

from src import config

REPO_ID = "Zihan1004/FNSPID"
# The large Nasdaq file adds AAPL/MSFT and extends coverage to 2023. The smaller
# All_external file (2015-2020, NYSE names) was filtered in an earlier pass and
# its rows are kept in the backup below, so we only fetch the Nasdaq file now --
# on a tight disk we cannot hold both 23 GB + 5.7 GB at once.
NASDAQ_FILE = "Stock_news/nasdaq_exteral_data.csv"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--Zihan1004--FNSPID")
TICKERS = set(config.TICKERS)
START_YEAR = int(config.START_DATE[:4])
END_YEAR = int(config.END_DATE[:4])
OUT_DIR = os.path.join(config.DATA_DIR, "news")
OUT_PATH = os.path.join(OUT_DIR, "headlines_raw.csv")
BACKUP_ALL_EXTERNAL = os.path.join(OUT_DIR, "headlines_all_external.csv")


def _filter_file(local: str, rows: dict) -> None:
    """Add qualifying (symbol, timestamp, headline) rows from ``local`` to ``rows``
    (keyed for dedup). Column lookup is by name, so the two files' differing
    schemas -- one has a leading unnamed index column -- are both handled."""
    seen = 0
    with open(local, newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_date, i_title, i_sym = col["Date"], col["Article_title"], col["Stock_symbol"]
        for row in reader:
            seen += 1
            if seen % 2_000_000 == 0:
                print(f"    scanned {seen:,} rows, kept {len(rows):,}", flush=True)
            if len(row) <= max(i_date, i_title, i_sym):
                continue
            sym = row[i_sym].strip().upper()
            if sym not in TICKERS:
                continue
            date = row[i_date].strip()
            if len(date) < 4 or not (str(START_YEAR) <= date[:4] <= str(END_YEAR)):
                continue
            title = row[i_title].strip()
            if not title:
                continue
            rows[(sym, date, title)] = None  # dedup on identical (sym, ts, headline)
    print(f"    file done: scanned {seen:,}", flush=True)


def _load_backup(rows: dict) -> None:
    """Seed rows from the previously-filtered All_external headlines."""
    if not os.path.exists(BACKUP_ALL_EXTERNAL):
        return
    with open(BACKUP_ALL_EXTERNAL, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows[(r["symbol"], r["timestamp_utc"], r["headline"])] = None
    print(f"seeded {len(rows):,} rows from All_external backup", flush=True)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    csv.field_size_limit(10_000_000)
    rows: dict = {}
    _load_backup(rows)

    print(f"downloading {REPO_ID}/{NASDAQ_FILE} (resumable, ~23 GB)...", flush=True)
    local = hf_hub_download(repo_id=REPO_ID, filename=NASDAQ_FILE, repo_type="dataset")
    print(f"  local copy: {local}\n  filtering...", flush=True)
    _filter_file(local, rows)

    per_ticker: dict[str, int] = {t: 0 for t in TICKERS}
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "timestamp_utc", "headline"])
        for (sym, date, title) in rows:
            writer.writerow([sym, date, title])
            per_ticker[sym] += 1

    # Reclaim the 23 GB source immediately -- we only need the filtered output.
    print("deleting the 23 GB source to reclaim disk...", flush=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

    print(f"DONE kept={len(rows):,} unique headlines", flush=True)
    for t in sorted(per_ticker):
        print(f"  {t}: {per_ticker[t]:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
