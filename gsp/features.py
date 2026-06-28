"""Point-in-time feature engineering.

LEAKAGE RULE (the whole ballgame): every feature for the row dated `t` may use
ONLY information known at the close of day `t` — i.e. OHLCV up to and including
day t. We never use a future bar in a feature. Rolling/ewm windows in pandas are
backward-looking by construction, and we never call shift(-k) here. The label
(see labels.py) is the only thing allowed to look at day t+1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def market_features(market: pd.DataFrame) -> pd.DataFrame:
    """Features derived from the market index (SPY). Indexed by date."""
    c = market["Close"]
    out = pd.DataFrame(index=market.index)
    out["mkt_ret_1"] = c.pct_change(1)
    out["mkt_ret_5"] = c.pct_change(5)
    out["mkt_ret_20"] = c.pct_change(20)
    sma50 = c.rolling(50, min_periods=50).mean()
    out["mkt_above_sma50"] = (c > sma50).astype("float32")
    out["mkt_vol_20"] = c.pct_change().rolling(20, min_periods=20).std()
    return out


def make_features(df: pd.DataFrame, market_feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature matrix for one ticker. Returns a DataFrame indexed by date.

    Input df: OHLCV indexed by date (ascending), auto-adjusted prices.
    """
    if df is None or len(df) < 220:
        return pd.DataFrame()

    df = df.sort_index()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    # --- Momentum / trailing returns -------------------------------------
    for k in (1, 2, 3, 5, 10, 20, 60, 120):
        f[f"ret_{k}"] = c.pct_change(k)

    # --- Distance from moving averages -----------------------------------
    for k in (10, 20, 50, 200):
        sma = c.rolling(k, min_periods=k).mean()
        f[f"px_to_sma{k}"] = c / sma - 1.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # --- Volatility / range ----------------------------------------------
    ret1 = c.pct_change()
    for k in (5, 10, 20, 60):
        f[f"vol_{k}"] = ret1.rolling(k, min_periods=k).std()
    f["atr14_pct"] = _atr(df, 14) / c
    daily_range = (h - l) / c
    f["range_today"] = daily_range
    f["range_avg_10"] = daily_range.rolling(10, min_periods=10).mean()
    # Today's close position within today's range (0 = closed on low, 1 = on high)
    f["close_in_range"] = (c - l) / (h - l).replace(0.0, np.nan)

    # --- Gaps -------------------------------------------------------------
    prev_c = c.shift(1)
    f["gap_today"] = (o - prev_c) / prev_c
    f["gap_avg_5"] = ((o - prev_c) / prev_c).rolling(5, min_periods=5).mean()

    # --- RSI / stochastic position ---------------------------------------
    f["rsi_14"] = _rsi(c, 14) / 100.0
    lo20 = l.rolling(20, min_periods=20).min()
    hi20 = h.rolling(20, min_periods=20).max()
    f["stoch_20"] = (c - lo20) / (hi20 - lo20).replace(0.0, np.nan)

    # --- 52-week context --------------------------------------------------
    hi252 = c.rolling(252, min_periods=120).max()
    lo252 = c.rolling(252, min_periods=120).min()
    f["dist_52w_high"] = c / hi252 - 1.0
    f["dist_52w_low"] = c / lo252 - 1.0

    # --- Volume -----------------------------------------------------------
    vol_sma20 = v.rolling(20, min_periods=20).mean()
    f["vol_ratio_20"] = v / vol_sma20.replace(0.0, np.nan)
    vol_std20 = v.rolling(20, min_periods=20).std()
    f["vol_z_20"] = (v - vol_sma20) / vol_std20.replace(0.0, np.nan)
    dollar_vol = c * v
    f["log_dollar_vol_20"] = np.log1p(dollar_vol.rolling(20, min_periods=20).mean())

    # --- "Does this name pop?" history (counts of past big up days) -------
    big_up = (ret1 >= 0.08).astype("float32")
    f["pops_20"] = big_up.rolling(20, min_periods=20).sum()
    f["pops_60"] = big_up.rolling(60, min_periods=60).sum()
    f["max_up_10"] = ret1.rolling(10, min_periods=10).max()
    f["max_up_20"] = ret1.rolling(20, min_periods=20).max()

    # --- Price level (cheaper stocks pop more) ---------------------------
    f["log_price"] = np.log(c)

    # --- Seasonality (weak, but free) ------------------------------------
    f["dow"] = df.index.dayofweek.astype("float32")
    f["month"] = df.index.month.astype("float32")

    # --- Market-relative --------------------------------------------------
    if market_feats is not None and not market_feats.empty:
        mf = market_feats.reindex(f.index).ffill()
        for col in mf.columns:
            f[col] = mf[col]
        f["rs_5"] = f["ret_5"] - mf["mkt_ret_5"]
        f["rs_20"] = f["ret_20"] - mf["mkt_ret_20"]

    return f


FEATURE_COLUMNS_CACHE: list[str] | None = None


def feature_columns(sample: pd.DataFrame) -> list[str]:
    """All feature columns (everything that isn't a label/meta/forward column).

    CRITICAL: anything starting with 'fwd_' is a FUTURE return used only by the
    backtest/labels — it must NEVER be a model feature. We exclude by prefix so a
    new forward column can't silently leak in again (the bug that taught us this).
    """
    meta = {"y", "ticker", "date", "asof", "close",
            "Open", "High", "Low", "Close", "Volume"}
    cols = [c for c in sample.columns if c not in meta and not c.startswith("fwd_")]
    leaked = [c for c in cols if c.startswith("fwd_") or c in ("y",)]
    assert not leaked, f"forward/label columns leaked into features: {leaked}"
    return cols
