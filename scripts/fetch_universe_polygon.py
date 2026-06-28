"""Build a broad, survivorship-AWARE universe from Polygon's reference data, then
write it to data/universe.txt for use with `--universe file`.

Why Polygon here: it lists not just *currently active* common stocks but also
*delisted* ones (active=false) — the names that went to zero or got acquired. Most
free lists only show survivors, which is exactly what inflates backtests. Including
delisted tickers (whatever history yfinance still has for them) pushes back on that.

We only hit Polygon's reference endpoint (cheap, paginated) — the actual price
history is still pulled by the existing yfinance layer (long history, no rate cap).
Polygon free is ~5 calls/min, so we pace the pagination.

Run:
  python scripts/fetch_universe_polygon.py                 # active common stock
  python scripts/fetch_universe_polygon.py --delisted      # + delisted (survivorship)
  python scripts/fetch_universe_polygon.py --max 4000
"""
from __future__ import annotations
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsp.secrets import KEYS
from gsp.config import DATA_DIR

BASE = "https://api.polygon.io/v3/reference/tickers"


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "gsp/0.1"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited -> wait out the minute
                print("  [rate-limit] sleeping 13s…")
                time.sleep(13)
                continue
            print(f"  HTTP {e.code}: {e.read().decode('utf-8','replace')[:120]}")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  err: {e}")
            time.sleep(3)
    return None


def collect(active: bool, key: str, max_n: int | None) -> list[str]:
    syms: list[str] = []
    url = (f"{BASE}?type=CS&market=stocks&active={'true' if active else 'false'}"
           f"&limit=1000&sort=ticker&apiKey={key}")
    page = 0
    while url:
        data = _get(url)
        if not data:
            break
        for r in data.get("results", []):
            t = (r.get("ticker") or "").upper()
            if t and t.isalpha() and len(t) <= 5:
                syms.append(t)
        page += 1
        print(f"  page {page}: {len(syms)} tickers so far "
              f"({'active' if active else 'delisted'})")
        if max_n and len(syms) >= max_n:
            break
        nxt = data.get("next_url")
        url = f"{nxt}&apiKey={key}" if nxt else None
        if url:
            time.sleep(13)  # ~5 calls/min budget
    return syms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delisted", action="store_true",
                    help="also include delisted common stock (survivorship-aware)")
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()

    key = KEYS.get("MASSIVE_API_KEY")
    if not key:
        print("No Polygon (MASSIVE_API_KEY) key in .env."); return

    print("Fetching active common stock…")
    syms = set(collect(True, key, a.max))
    if a.delisted:
        print("Fetching delisted common stock (survivorship-aware)…")
        syms |= set(collect(False, key, a.max))

    tickers = sorted(syms)
    if a.max:
        tickers = tickers[:a.max]
    out = DATA_DIR / "universe.txt"
    out.write_text("\n".join(tickers) + "\n")
    print(f"\nWrote {len(tickers):,} tickers -> {out}")
    print("Next:")
    print("  python cli.py download --universe file   (yfinance pulls the history)")
    print("  python cli.py build    --universe file")
    print("  python cli.py evaluate --universe file")


if __name__ == "__main__":
    main()
