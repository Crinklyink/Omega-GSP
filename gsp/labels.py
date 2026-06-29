"""Label construction.

Decision time = close of day t. We act at the OPEN of day t+1. The default target
(TARGET_MODE="high_vs_open") asks the TRADEABLE question:

    y = 1  iff  High_{t+1} >= Open_{t+1} * (1 + TARGET_MOVE)

i.e. after you buy at tomorrow's open, does the stock climb >= 8% intraday that day
(so a +8% limit fills)? This deliberately excludes the overnight gap — a name that
opened already up doesn't count unless it keeps climbing.

We also stash forward returns used only by the backtest (NEVER features):

    fwd_high_ret = High_{t+1}  / Close_t - 1
    fwd_open_ret = Open_{t+1}  / Close_t - 1
    fwd_close_ret= Close_{t+1} / Close_t - 1

The last row of every ticker has no t+1, so its label is NaN and gets dropped from
training but is exactly the row we score live in scan.py.
"""
from __future__ import annotations
import pandas as pd

from .config import TARGET_MOVE, TARGET_MODE


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    close_t = df["Close"]
    high_next = df["High"].shift(-1)
    open_next = df["Open"].shift(-1)
    close_next = df["Close"].shift(-1)
    low_next = df["Low"].shift(-1)

    out = pd.DataFrame(index=df.index)
    out["fwd_high_ret"] = high_next / close_t - 1.0
    out["fwd_open_ret"] = open_next / close_t - 1.0
    out["fwd_close_ret"] = close_next / close_t - 1.0
    out["fwd_low_ret"] = low_next / close_t - 1.0  # next-day low (for stop-loss sim)

    if TARGET_MODE == "high_vs_open":
        # +8% from the OPEN, intraday (tradeable, excludes the gap).
        out["y"] = (high_next >= open_next * (1.0 + TARGET_MOVE)).astype("float32")
        out.loc[high_next.isna() | open_next.isna(), "y"] = pd.NA
    else:
        # +8% from the prior CLOSE (includes overnight gap).
        out["y"] = (high_next >= close_t * (1.0 + TARGET_MOVE)).astype("float32")
        out.loc[high_next.isna(), "y"] = pd.NA
    return out
