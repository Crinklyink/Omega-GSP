"""Filter data/universe.txt down to names with market cap >= a floor (default $50M),
using Finnhub company profiles. Results are cached so you can stop/resume; re-runs
only fetch tickers not already cached.

Finnhub free is ~60 calls/min, so this paces ~1 req/sec. For a ~6k-name universe
expect ~1.5–2 hours; it's resumable, so just run it again if interrupted. Run it in
the background and let it finish.

Run:
  python scripts/filter_universe_mcap.py                 # >= $50M, strict
  python scripts/filter_universe_mcap.py --min-mcap 100  # >= $100M
  python scripts/filter_universe_mcap.py --keep-unknown  # keep names with no data
"""
from __future__ import annotations
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsp.secrets import KEYS
from gsp.config import DATA_DIR

UNIVERSE = DATA_DIR / "universe.txt"
CACHE = DATA_DIR / "mcap_cache.json"          # {ticker: market_cap_millions or null}
FULL_BACKUP = DATA_DIR / "universe_full.txt"  # the pre-filter list


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _mcap(ticker: str, token: str) -> float | None:
    """Finnhub marketCapitalization is in millions USD."""
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={token}"
    req = urllib.request.Request(url, headers={"User-Agent": "gsp/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        v = d.get("marketCapitalization")
        return float(v) if v else None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(20)
            return _mcap(ticker, token)
        return None
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mcap", type=float, default=50.0, help="floor in $millions")
    ap.add_argument("--keep-unknown", action="store_true")
    a = ap.parse_args()

    token = KEYS.get("FINNHUB_API_KEY")
    if not token:
        print("No FINNHUB_API_KEY in .env."); return
    if not UNIVERSE.exists():
        print("No data/universe.txt. Run scripts/fetch_universe_polygon.py first."); return

    tickers = [t.strip().upper() for t in UNIVERSE.read_text().splitlines() if t.strip()]
    if not FULL_BACKUP.exists():
        FULL_BACKUP.write_text("\n".join(tickers) + "\n")
    cache = _load_cache()

    todo = [t for t in tickers if t not in cache]
    print(f"{len(tickers)} tickers; {len(cache)} cached; {len(todo)} to fetch "
          f"(~{len(todo)/60:.0f} min at 60/min)")

    for i, t in enumerate(todo, 1):
        cache[t] = _mcap(t, token)
        if i % 50 == 0:
            CACHE.write_text(json.dumps(cache))
            kept = sum(1 for v in cache.values() if v and v >= a.min_mcap)
            print(f"  {i}/{len(todo)} fetched; {kept} >= ${a.min_mcap:.0f}M so far")
        time.sleep(1.05)  # ~57/min, safely under the cap
    CACHE.write_text(json.dumps(cache))

    keep = []
    for t in tickers:
        v = cache.get(t)
        if v is None:
            if a.keep_unknown:
                keep.append(t)
        elif v >= a.min_mcap:
            keep.append(t)

    UNIVERSE.write_text("\n".join(sorted(keep)) + "\n")
    print(f"\nKept {len(keep):,}/{len(tickers):,} with mcap >= ${a.min_mcap:.0f}M "
          f"-> {UNIVERSE}")
    print(f"(full pre-filter list preserved at {FULL_BACKUP})")


if __name__ == "__main__":
    main()
