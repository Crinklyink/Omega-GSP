"""Download ~2 years of hourly bars for the universe (yfinance 60m, 730d limit).

Network-bound: safe to run while a CPU-bound optimize/evaluate is going.

Run:  python scripts/fetch_hourly.py [--limit N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsp import universe as U
from gsp.intraday import update_universe_hourly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tickers = U.load_from_file() or U.FALLBACK
    if a.limit:
        tickers = tickers[: a.limit]
    print(f"[intraday] fetching hourly bars for {len(tickers)} tickers")
    have = update_universe_hourly(tickers)
    print(f"[intraday] done: {len(have)} tickers have hourly data")


if __name__ == "__main__":
    main()
