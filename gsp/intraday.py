"""Hourly (60m) bar cache — the intraday tier.

Free yfinance intraday data only reaches back ~730 calendar days, so this layer
CANNOT feed features to the 14-year model. What it CAN do — and why it exists —
is replay the strategy's trades bar-by-bar over the last two years:

  * did the +8% limit fill BEFORE the stop-loss, or after? (daily bars can't say;
    the daily-bar sim conservatively assumes the stop always fills first)
  * what hour of the session do the pops actually print?
  * which intraday exit rules make the strategy's expectancy positive?

One parquet per ticker under data/raw_60m/. Timestamps are exchange-local
(America/New_York) as returned by yfinance.
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_60M_DIR = DATA_DIR / "raw_60m"
RAW_60M_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
LOOKBACK_DAYS = 729  # yfinance hard limit for 60m bars is 730 days


def _path(ticker: str) -> Path:
    return RAW_60M_DIR / f"{ticker.replace('/', '_')}.parquet"


def load_cached_hourly(ticker: str) -> pd.DataFrame | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[COLUMNS]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def _download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    raw = yf.download(
        tickers, period=f"{LOOKBACK_DAYS}d", interval="60m",
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t].dropna(how="all")
                if not sub.empty:
                    out[t] = _clean(sub)
    else:
        out[tickers[0]] = _clean(raw)
    return out


def update_universe_hourly(tickers: list[str], batch_size: int = 25,
                           sleep: float = 1.5, incremental: bool = True) -> list[str]:
    """Download/refresh hourly bars. Returns tickers that have data."""
    have, todo = [], []
    now = pd.Timestamp.utcnow()
    for t in tickers:
        cached = load_cached_hourly(t) if incremental else None
        if cached is not None and not cached.empty:
            last = cached.index.max()
            last_utc = last.tz_convert("UTC") if last.tzinfo else last.tz_localize("UTC")
            if (now - last_utc).days <= 3:
                have.append(t)
                continue
        todo.append(t)

    print(f"[intraday] {len(have)} fresh, {len(todo)} to (re)download", flush=True)
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        try:
            got = _download_batch(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"[intraday] batch {i // batch_size} failed: {e}", flush=True)
            got = {}
        for t, df in got.items():
            if df is None or df.empty:
                continue
            cached = load_cached_hourly(t)
            if cached is not None:
                df = pd.concat([cached, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(_path(t))
            have.append(t)
        done = min(i + batch_size, len(todo))
        if done % 250 < batch_size or done == len(todo):
            print(f"[intraday] {done}/{len(todo)} downloaded", flush=True)
        time.sleep(sleep)
    return sorted(set(have))


def day_bars(ticker: str, day) -> pd.DataFrame | None:
    """The hourly bars of one session (exchange-local date), in time order."""
    df = load_cached_hourly(ticker)
    if df is None or df.empty:
        return None
    day = pd.Timestamp(day).date()
    idx = df.index
    if idx.tz is None:
        dates = idx.normalize().date
    else:
        dates = idx.tz_convert("America/New_York").date
    sub = df[dates == day]
    return sub if not sub.empty else None
