"""Probe the API keys in .env to discover what each unlocks. Prints provider +
HTTP status + a short response snippet. Never prints the keys themselves.

The "Massive" key is of unknown provenance, so we try it against several common
stock-data providers whose keys look similar (Polygon, FMP, Twelve Data, Tiingo,
Marketstack) and report which one accepts it.

Run:  python scripts/probe_keys.py
"""
from __future__ import annotations
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsp.secrets import KEYS, masked


def get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "gsp/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:200]


def snippet(body: str, n: int = 160) -> str:
    return " ".join(body.split())[:n]


def looks_ok(status, body: str) -> bool:
    if status != 200:
        return False
    low = body.lower()
    bad = ("error", "invalid api key", "apikey", "not authorized", "unauthorized",
           "limit reached", "premium", "you don't have access", "rate limit")
    # AlphaVantage returns 200 with a "Note"/"Information" on throttle/invalid.
    if '"information"' in low or '"note"' in low or '"error message"' in low:
        return False
    if any(b in low for b in ("invalid api key", "not authorized", "unauthorized",
                              "you don't have access")):
        return False
    return True


def probe_named(name: str, tests: list[tuple[str, str]]):
    key = KEYS.get(name, "")
    print(f"\n=== {name}  ({masked(name)}) ===")
    if not key:
        print("  (no key set)")
        return
    for label, url in tests:
        status, body = get(url.replace("KEY", key))
        ok = looks_ok(status, body)
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {label:28s} http={status}  {snippet(body)}")


# --- Known providers ---
probe_named("ALPHAVANTAGE_API_KEY", [
    ("GLOBAL_QUOTE AAPL", "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=KEY"),
    ("NEWS_SENTIMENT", "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&limit=1&apikey=KEY"),
])

probe_named("FINNHUB_API_KEY", [
    ("quote AAPL", "https://finnhub.io/api/v1/quote?symbol=AAPL&token=KEY"),
    ("earnings calendar", "https://finnhub.io/api/v1/calendar/earnings?from=2026-06-28&to=2026-07-05&token=KEY"),
    ("company news", "https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2026-06-20&to=2026-06-28&token=KEY"),
    ("recommendation", "https://finnhub.io/api/v1/stock/recommendation?symbol=AAPL&token=KEY"),
    ("daily candle", "https://finnhub.io/api/v1/stock/candle?symbol=AAPL&resolution=D&count=5&token=KEY"),
])

# --- Unknown "Massive" key: try several providers ---
print("\n########  Identifying the 'Massive' key  ########")
mk = KEYS.get("MASSIVE_API_KEY", "")
massive_tests = [
    ("Polygon ref tickers", "https://api.polygon.io/v3/reference/tickers?limit=1&apiKey=KEY"),
    ("Polygon prev-close", "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=KEY"),
    ("Polygon news", "https://api.polygon.io/v2/reference/news?limit=1&apiKey=KEY"),
    ("FMP quote", "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=KEY"),
    ("TwelveData quote", "https://api.twelvedata.com/quote?symbol=AAPL&apikey=KEY"),
    ("Tiingo daily", "https://api.tiingo.com/tiingo/daily/AAPL/prices?token=KEY"),
    ("Marketstack eod", "http://api.marketstack.com/v1/eod?access_key=KEY&symbols=AAPL&limit=1"),
    ("Polygon snapshot gainers", "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey=KEY"),
]
if mk:
    for label, url in massive_tests:
        status, body = get(url.replace("KEY", mk))
        ok = looks_ok(status, body)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:26s} http={status}  {snippet(body)}")
else:
    print("  (no key set)")

print("\nDone. Look for OK rows to see what each key unlocks.")
