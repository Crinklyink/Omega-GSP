"""Assemble the full training table: stack every ticker's (features + label) rows
into one long DataFrame indexed by (date, ticker), after applying liquidity filters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (DATASET_DIR, MIN_PRICE, MIN_DOLLAR_VOLUME, MAX_ATR_PCT)
from .data import load_cached, load_market
from .features import make_features, market_features, add_cross_sectional
from .labels import make_labels

DATASET_PATH = DATASET_DIR / "dataset.parquet"

# Forward-looking fields used only by the backtest, never as features.
FWD_COLS = ["fwd_high_ret", "fwd_open_ret", "fwd_close_ret"]


def build_ticker_frame(ticker: str, mkt_feats: pd.DataFrame | None) -> pd.DataFrame:
    raw = load_cached(ticker)
    if raw is None or len(raw) < 260:
        return pd.DataFrame()

    feats = make_features(raw, mkt_feats)
    if feats.empty:
        return pd.DataFrame()
    labels = make_labels(raw)

    df = feats.join(labels, how="inner")
    # Liquidity / tradeability filter, computed point-in-time.
    dollar_vol = (raw["Close"] * raw["Volume"]).rolling(20, min_periods=20).mean()
    keep = (raw["Close"] >= MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOLUME)
    df = df[keep.reindex(df.index).fillna(False)]
    # Volatility ceiling: hyper-volatile names are excluded from training so the
    # model never learns to love them. scan.py applies the identical cap live.
    if "atr14_pct" in df.columns:
        df = df[df["atr14_pct"] <= MAX_ATR_PCT]

    df["ticker"] = ticker
    df.index.name = "date"
    return df


def build_dataset(tickers: list[str], save: bool = True) -> pd.DataFrame:
    mkt = load_market()
    mkt_feats = market_features(mkt) if mkt is not None else None

    frames = []
    for t in tqdm(tickers, desc="features"):
        try:
            fr = build_ticker_frame(t, mkt_feats)
            if not fr.empty:
                frames.append(fr)
        except Exception as e:  # noqa: BLE001
            print(f"[dataset] {t} failed: {e}")
    if not frames:
        raise RuntimeError("No data assembled — did you run `download` first?")

    full = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
    # Cross-sectional (per-day, universe-relative) features. Point-in-time safe:
    # each day's ranks use only that day's values. scan.py applies the same
    # transform to its live one-day panel.
    print("[dataset] adding cross-sectional features ...")
    full = add_cross_sectional(full)
    # Drop rows whose label is undefined (the very last bar per ticker) for training,
    # but keep them out here — scan.py recomputes live rows separately.
    full = full.replace([np.inf, -np.inf], np.nan)
    # float32 is plenty for returns/ratios and halves the panel's memory footprint.
    f64 = full.select_dtypes(include="float64").columns
    full[f64] = full[f64].astype("float32")

    if save:
        full.reset_index().to_parquet(DATASET_PATH)
        print(f"[dataset] saved {len(full):,} rows -> {DATASET_PATH}")
    return full


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    return df.set_index(["date", "ticker"]).sort_index()
