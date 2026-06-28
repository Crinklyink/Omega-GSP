"""Download and cache daily OHLCV bars. One parquet file per ticker under data/raw.

Uses auto-adjusted prices so that stock splits and dividends do NOT create fake
8% moves in the history. Downloads in batches and caches incrementally, so a second
run only fetches new bars.
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, HISTORY_START, MARKET_INDEX

warnings.filterwarnings("ignore", category=FutureWarning)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return RAW_DIR / f"{safe}.parquet"


def load_cached(ticker: str) -> pd.DataFrame | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df[COLUMNS]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Volume"].fillna(0) >= 0]
    return df


def _download_batch(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    raw = yf.download(
        tickers, start=start, auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    if raw is None or raw.empty:
        return out
    # Normalise single vs multi ticker layout.
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t].dropna(how="all")
                if not sub.empty:
                    out[t] = _clean(sub)
    else:
        t = tickers[0]
        out[t] = _clean(raw)
    return out


def update_universe(tickers: list[str], batch_size: int = 50,
                    sleep: float = 1.0, incremental: bool = True) -> list[str]:
    """Download/refresh all tickers. Returns list of tickers that have data."""
    have = []
    todo = []
    for t in tickers:
        cached = load_cached(t) if incremental else None
        if cached is not None and not cached.empty:
            last = cached.index.max()
            # If we already have data within the last ~4 days, skip the refetch.
            if (pd.Timestamp.utcnow().tz_localize(None) - last).days <= 4:
                have.append(t)
                continue
        todo.append(t)

    print(f"[data] {len(have)} fresh, {len(todo)} to (re)download")
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        try:
            got = _download_batch(chunk, HISTORY_START)
        except Exception as e:  # noqa: BLE001
            print(f"[data] batch {i//batch_size} failed: {e}")
            got = {}
        for t, df in got.items():
            if df is None or df.empty:
                continue
            cached = load_cached(t)
            if cached is not None:
                df = pd.concat([cached, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(_path(t))
            have.append(t)
        done = min(i + batch_size, len(todo))
        print(f"[data] {done}/{len(todo)} downloaded ({len(have)} total with data)")
        time.sleep(sleep)
    return sorted(set(have))


def load_market() -> pd.DataFrame | None:
    """The market index series (SPY) used for regime/relative features."""
    df = load_cached(MARKET_INDEX)
    if df is None:
        got = _download_batch([MARKET_INDEX], HISTORY_START)
        df = got.get(MARKET_INDEX)
        if df is not None and not df.empty:
            df.to_parquet(_path(MARKET_INDEX))
    return df
