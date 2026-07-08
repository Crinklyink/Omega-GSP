"""Point-in-time feature engineering.

LEAKAGE RULE (the whole ballgame): every feature for the row dated `t` may use
ONLY information known at the close of day `t` — i.e. OHLCV up to and including
day t. We never use a future bar in a feature. Rolling/ewm windows in pandas are
backward-looking by construction, and we never call shift(-k) here. The label
(see labels.py) is the only thing allowed to look at day t+1.

Cross-sectional features (add_cross_sectional) rank each name against the rest
of the universe ON THE SAME DAY — also point-in-time safe, because at the close
of day t you can observe every other stock's day-t bar too.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import TARGET_MOVE, TARGET_MODE, MAX_DIP


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


def _streak(cond: pd.Series) -> pd.Series:
    """Length of the current run of consecutive True values, backward-looking."""
    b = cond.astype(int)
    grp = (b != b.shift()).cumsum()
    return b * (b.groupby(grp).cumcount() + 1)


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
    out["mkt_vol_60"] = c.pct_change().rolling(60, min_periods=60).std()
    # distance from 20d high: how stretched/dipped is the market itself
    hi20 = c.rolling(20, min_periods=20).max()
    out["mkt_dist_hi20"] = c / hi20 - 1.0
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
    atr14 = _atr(df, 14)
    f["atr14_pct"] = atr14 / c
    daily_range = (h - l) / c
    f["range_today"] = daily_range
    f["range_avg_10"] = daily_range.rolling(10, min_periods=10).mean()
    range_avg_5 = daily_range.rolling(5, min_periods=5).mean()
    range_avg_20 = daily_range.rolling(20, min_periods=20).mean()
    # range compression/expansion: recent range vs the month's norm. Squeezes
    # (values < 1) often precede expansion moves.
    f["range_trend"] = range_avg_5 / range_avg_20.replace(0.0, np.nan)
    # NR7: today's range is the narrowest of the last 7 sessions
    f["nr7"] = (daily_range <= daily_range.rolling(7, min_periods=7).min() * 1.0001).astype("float32")
    # Today's close position within today's range (0 = closed on low, 1 = on high)
    f["close_in_range"] = (c - l) / (h - l).replace(0.0, np.nan)

    # --- Vol-scaled momentum (a 5% move means more in a quiet name) ------
    vol20 = f["vol_20"]
    f["ret_1_volscaled"] = ret1 / vol20.replace(0.0, np.nan)
    f["ret_5_volscaled"] = f["ret_5"] / (vol20 * np.sqrt(5)).replace(0.0, np.nan)
    f["ret_20_volscaled"] = f["ret_20"] / (vol20 * np.sqrt(20)).replace(0.0, np.nan)

    # --- Bollinger context -------------------------------------------------
    mean20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    f["bb_z_20"] = (c - mean20) / std20.replace(0.0, np.nan)
    bb_width = (4.0 * std20) / mean20.replace(0.0, np.nan)
    f["bb_width_20"] = bb_width
    # where today's bandwidth sits vs its own last ~6 months (0 = tightest squeeze)
    f["bb_squeeze_120"] = bb_width.rolling(120, min_periods=60).rank(pct=True)

    # --- Gaps / overnight vs intraday decomposition -----------------------
    prev_c = c.shift(1)
    gap = (o - prev_c) / prev_c
    f["gap_today"] = gap
    f["gap_avg_5"] = gap.rolling(5, min_periods=5).mean()
    f["gap_vol_20"] = gap.rolling(20, min_periods=20).std()
    intraday = c / o.replace(0.0, np.nan) - 1.0
    f["intraday_ret"] = intraday
    f["intraday_ret_avg_5"] = intraday.rolling(5, min_periods=5).mean()
    f["intraday_ret_avg_20"] = intraday.rolling(20, min_periods=20).mean()

    # --- Candle anatomy (wicks tell you about intraday reach) -------------
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    upper_shadow = (h - body_top) / c
    lower_shadow = (body_bot - l) / c
    f["body_pct"] = (c - o).abs() / c
    f["upper_shadow"] = upper_shadow
    f["lower_shadow"] = lower_shadow
    f["upper_shadow_avg_10"] = upper_shadow.rolling(10, min_periods=10).mean()
    f["lower_shadow_avg_10"] = lower_shadow.rolling(10, min_periods=10).mean()

    # --- Label-aligned intraday pop history --------------------------------
    # The best single predictor of the label is how often THIS name has done
    # exactly what the label asks, in the past. hi_vs_open/lo_vs_open on day t
    # use only day-t's own bar -> point-in-time. Under "clean_pop" the event is
    # dip-conditioned to match the label: reached +TARGET_MOVE from the open
    # WITHOUT trading more than MAX_DIP below it.
    hi_vs_open = h / o.replace(0.0, np.nan) - 1.0
    lo_vs_open = l / o.replace(0.0, np.nan) - 1.0
    pop_event = (hi_vs_open >= TARGET_MOVE)
    if TARGET_MODE == "clean_pop":
        pop_event = pop_event & (lo_vs_open >= -MAX_DIP)
    pop_event = pop_event.astype("float32")
    f["hi_open_today"] = hi_vs_open
    f["pop_ho_20"] = pop_event.rolling(20, min_periods=20).mean()
    f["pop_ho_60"] = pop_event.rolling(60, min_periods=60).mean()
    f["pop_ho_120"] = pop_event.rolling(120, min_periods=120).mean()
    f["hi_open_avg_10"] = hi_vs_open.rolling(10, min_periods=10).mean()
    f["hi_open_avg_20"] = hi_vs_open.rolling(20, min_periods=20).mean()
    f["hi_open_max_20"] = hi_vs_open.rolling(20, min_periods=20).max()

    # --- RSI / stochastic position ---------------------------------------
    f["rsi_14"] = _rsi(c, 14) / 100.0
    f["rsi_2"] = _rsi(c, 2) / 100.0
    lo20 = l.rolling(20, min_periods=20).min()
    hi20 = h.rolling(20, min_periods=20).max()
    f["stoch_20"] = (c - lo20) / (hi20 - lo20).replace(0.0, np.nan)
    # distance to the 20d high measured in ATRs (breakout proximity)
    f["atr_dist_hi20"] = (hi20 - c) / atr14.replace(0.0, np.nan)
    # how many of the last 10 sessions printed a fresh 20d high
    f["new_hi20_count_10"] = (h >= hi20).astype("float32").rolling(10, min_periods=10).sum()

    # --- Streaks -----------------------------------------------------------
    f["up_streak"] = _streak(ret1 > 0).astype("float32")
    f["down_streak"] = _streak(ret1 < 0).astype("float32")

    # --- 52-week context --------------------------------------------------
    hi252 = c.rolling(252, min_periods=120).max()
    lo252 = c.rolling(252, min_periods=120).min()
    f["dist_52w_high"] = c / hi252 - 1.0
    f["dist_52w_low"] = c / lo252 - 1.0

    # --- Volume -----------------------------------------------------------
    vol_sma20 = v.rolling(20, min_periods=20).mean()
    vol_sma5 = v.rolling(5, min_periods=5).mean()
    f["vol_ratio_20"] = v / vol_sma20.replace(0.0, np.nan)
    f["vol_mom"] = vol_sma5 / vol_sma20.replace(0.0, np.nan) - 1.0
    vol_std20 = v.rolling(20, min_periods=20).std()
    f["vol_z_20"] = (v - vol_sma20) / vol_std20.replace(0.0, np.nan)
    dollar_vol = c * v
    f["log_dollar_vol_20"] = np.log1p(dollar_vol.rolling(20, min_periods=20).mean())
    # what fraction of the month's volume traded on up days (accumulation?)
    up_vol = v.where(ret1 > 0, 0.0)
    f["upvol_ratio_20"] = (up_vol.rolling(20, min_periods=20).sum()
                           / v.rolling(20, min_periods=20).sum().replace(0.0, np.nan))
    # Amihud illiquidity: price impact per dollar traded (log scale)
    amihud = (ret1.abs() / dollar_vol.replace(0.0, np.nan)).rolling(20, min_periods=20).mean()
    f["amihud_20"] = np.log(amihud.replace(0.0, np.nan) + 1e-14)

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
    f["dom"] = df.index.day.astype("float32")

    # --- Market-relative --------------------------------------------------
    if market_feats is not None and not market_feats.empty:
        mf = market_feats.reindex(f.index).ffill()
        for col in mf.columns:
            f[col] = mf[col]
        f["rs_5"] = f["ret_5"] - mf["mkt_ret_5"]
        f["rs_20"] = f["ret_20"] - mf["mkt_ret_20"]
        mret = mf["mkt_ret_1"]
        f["corr_mkt_60"] = ret1.rolling(60, min_periods=60).corr(mret)
        mvar = mret.rolling(60, min_periods=60).var()
        f["beta_60"] = ret1.rolling(60, min_periods=60).cov(mret) / mvar.replace(0.0, np.nan)

    return f


# Columns ranked cross-sectionally (per day, against the whole universe).
CS_RANK_COLS = [
    "vol_20", "atr14_pct", "ret_1", "ret_5", "ret_20", "vol_ratio_20",
    "log_dollar_vol_20", "pop_ho_60", "hi_open_avg_20", "bb_width_20",
    "gap_today", "rs_5",
]


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-day cross-sectional features to a (date, ticker)-indexed panel.

    Point-in-time safe: every value for day t uses only day-t values of features
    that are themselves point-in-time. Ranks answer "how volatile / hot / liquid
    is this name TODAY relative to everything else tradeable TODAY?" — which is
    exactly the comparison the daily top-K selection makes.
    """
    g = panel.groupby(level="date", sort=False)
    for col in CS_RANK_COLS:
        if col in panel.columns:
            panel[f"cs_rank_{col}"] = g[col].rank(pct=True).astype("float32")
    # Same-day market breadth, broadcast to every name (regime context beyond SPY).
    if "ret_1" in panel.columns:
        panel["cs_breadth_up"] = (panel["ret_1"] > 0).astype("float32") \
            .groupby(panel.index.get_level_values("date"), sort=False).transform("mean")
    if "vol_20" in panel.columns:
        panel["cs_median_vol20"] = g["vol_20"].transform("median").astype("float32")
    if "pop_ho_60" in panel.columns:
        # how "poppy" the whole tape is lately — small-cap froth indicator
        panel["cs_mean_pop60"] = g["pop_ho_60"].transform("mean").astype("float32")
    return panel


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
