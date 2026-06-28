"""Label construction.

Decision time = close of day t. The label asks about day t+1 ONLY:

    y = 1  iff  High_{t+1} >= Close_t * (1 + TARGET_MOVE)

This is the one place we are allowed to peek at the future. We also stash two
forward returns used later by the realistic backtest (these are NOT features and
must never be fed to the model):

    fwd_high_ret = High_{t+1} / Close_t - 1     (best-case if you held from close)
    fwd_open_ret = Open_{t+1} / Close_t - 1     (the overnight gap you can't trade)

The last row of every ticker has no t+1, so its label is NaN and gets dropped from
training but is exactly the row we score live in scan.py.
"""
from __future__ import annotations
import pandas as pd

from .config import TARGET_MOVE


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    close_t = df["Close"]
    high_next = df["High"].shift(-1)
    open_next = df["Open"].shift(-1)
    close_next = df["Close"].shift(-1)

    out = pd.DataFrame(index=df.index)
    out["fwd_high_ret"] = high_next / close_t - 1.0
    out["fwd_open_ret"] = open_next / close_t - 1.0
    out["fwd_close_ret"] = close_next / close_t - 1.0
    out["y"] = (high_next >= close_t * (1.0 + TARGET_MOVE)).astype("float32")
    # Where there is no next day, the label is undefined.
    out.loc[high_next.isna(), "y"] = pd.NA
    return out
