"""Defines the set of tickers we scan.

IMPORTANT (read this): +8% intraday moves are RARE in large-cap S&P 500 names and
much more common in smaller, more volatile stocks. If you train only on mega-caps
your positive class will be tiny and the model learns little. So the real universe
should lean toward liquid-but-volatile small/mid caps. This module gives you:

  * load_sp500()      -> S&P 500 constituents (stable, liquid, but few 8% pops)
  * load_from_file()  -> read tickers from data/universe.txt (one per line)
  * default_universe()-> file if present, else S&P 500, else a small fallback list

To scan "the whole market", drop a big list of tickers (NASDAQ + NYSE common
stock) into data/universe.txt. A helper to fetch one is in scripts/fetch_universe.py.
"""
from __future__ import annotations
import io
from pathlib import Path
from .config import DATA_DIR

UNIVERSE_FILE = DATA_DIR / "universe.txt"

# Small fallback so the pipeline runs even with no network list. Deliberately
# mixes liquid large caps with more volatile names so positives aren't zero.
FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL", "NFLX",
    "INTC", "MU", "QCOM", "AVGO", "MRVL", "SMCI", "PLTR", "SOFI", "RIVN",
    "LCID", "COIN", "MARA", "RIOT", "AFRM", "ROKU", "SNAP", "PINS", "UPST",
    "DKNG", "CVNA", "GME", "AMC", "BBBYQ", "F", "BAC", "T", "PFE", "XOM",
]


def load_from_file(path: Path = UNIVERSE_FILE) -> list[str] | None:
    if not path.exists():
        return None
    tickers = []
    for line in path.read_text().splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            tickers.append(t)
    return sorted(set(tickers)) or None


def load_sp500() -> list[str] | None:
    """Scrape current S&P 500 members from Wikipedia. Returns None on failure."""
    try:
        import pandas as pd
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
        return sorted(set(syms.str.upper().tolist()))
    except Exception as e:  # noqa: BLE001
        print(f"[universe] S&P500 fetch failed ({e}); using fallback.")
        return None


def default_universe() -> list[str]:
    return load_from_file() or load_sp500() or FALLBACK
