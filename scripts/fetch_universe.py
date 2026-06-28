"""Fetch a broad US common-stock universe (NASDAQ + NYSE/other listed) and write
it to data/universe.txt, one symbol per line. Use it with `--universe file`.

Source: NASDAQ Trader symbol directory (public, pipe-delimited).
We keep common stock, drop ETFs / test issues / warrants / units / preferreds,
and apply a light symbol sanity filter. This is still a *current-membership* list
(survivorship-biased) — it just covers far more names than the S&P 500, which is
what you want for actually finding 8% movers.

Run:  python scripts/fetch_universe.py [--max N]
"""
from __future__ import annotations
import sys
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gsp.config import DATA_DIR

NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

BAD_TOKENS = ("WARRANT", "UNIT", "PREFERRED", "DEPOSITARY", "RIGHT", "NOTES",
              "%", "ETF", "ETN", "FUND", "TRUST")


def _fetch(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("latin-1").splitlines()


def _parse(lines: list[str], sym_col: str, name_col: str,
           etf_col: str | None, test_col: str | None) -> list[str]:
    header = lines[0].split("|")
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            continue
        parts = ln.split("|")
        if len(parts) <= max(idx.get(sym_col, 0), idx.get(name_col, 0)):
            continue
        sym = parts[idx[sym_col]].strip().upper()
        name = parts[idx[name_col]].upper() if name_col in idx else ""
        if etf_col and etf_col in idx and parts[idx[etf_col]].strip() == "Y":
            continue
        if test_col and test_col in idx and parts[idx[test_col]].strip() == "Y":
            continue
        if not sym or not sym.isalpha() or len(sym) > 5:
            continue  # drop weird symbols, warrants (W), units (U) suffixes
        if any(tok in name for tok in BAD_TOKENS):
            continue
        out.append(sym)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="cap number of tickers")
    a = ap.parse_args()

    syms: set[str] = set()
    try:
        syms.update(_parse(_fetch(NASDAQ), "Symbol", "Security Name",
                           "ETF", "Test Issue"))
    except Exception as e:  # noqa: BLE001
        print(f"[universe] NASDAQ list failed: {e}")
    try:
        syms.update(_parse(_fetch(OTHER), "ACT Symbol", "Security Name",
                           "ETF", "Test Issue"))
    except Exception as e:  # noqa: BLE001
        print(f"[universe] other-listed failed: {e}")

    tickers = sorted(syms)
    if a.max:
        tickers = tickers[:a.max]
    if not tickers:
        print("[universe] got 0 tickers (network blocked?). Nothing written.")
        return
    out = DATA_DIR / "universe.txt"
    out.write_text("\n".join(tickers) + "\n")
    print(f"[universe] wrote {len(tickers):,} tickers -> {out}")
    print("Next: python cli.py download --universe file   (this can take a while)")


if __name__ == "__main__":
    main()
