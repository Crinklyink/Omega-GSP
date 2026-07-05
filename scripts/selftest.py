"""Fast invariant self-test — run after ANY edit to features/labels/data code.

The centerpiece is the truncation-invariance test: a feature for day t may use
only data up to day t, therefore computing features on data truncated at day t
must give EXACTLY the same row t as computing on the full history. Any feature
that peeks forward fails this immediately — it's the strongest cheap lookahead
check that exists (stronger than label-shuffling, and it needs no model).

Run:  python scripts/selftest.py     (exits non-zero on failure)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def synth_ohlcv(n: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.03, n)
    close = 20 * np.cumprod(1 + ret)
    open_ = close * (1 + rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.02, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.02, n)))
    vol = rng.integers(200_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


print("== truncation invariance (the lookahead test) ==")
from gsp.features import make_features, market_features, add_cross_sectional, feature_columns

df = synth_ohlcv()
mkt = synth_ohlcv(seed=11)
mfe = market_features(mkt)
full_feats = make_features(df, mfe)

# pick several checkpoints; features at t must not change when the future is cut off
ok_all = True
bad_cols = set()
for t in (260, 300, 380, 419):
    trunc = make_features(df.iloc[: t + 1], mfe)
    a = full_feats.iloc[t]
    b = trunc.iloc[-1]
    assert full_feats.index[t] == trunc.index[-1]
    for col in full_feats.columns:
        va, vb = a[col], b[col]
        if (pd.isna(va) and pd.isna(vb)):
            continue
        if not np.isclose(va, vb, rtol=1e-9, atol=1e-12, equal_nan=True):
            ok_all = False
            bad_cols.add(col)
check("features are truncation-invariant (no future dependence)", ok_all,
      f"leaking columns: {sorted(bad_cols)}")

print("== labels ==")
from gsp.labels import make_labels
lb = make_labels(df)
t = 100
o1, h1, c0 = df["Open"].iloc[t + 1], df["High"].iloc[t + 1], df["Close"].iloc[t]
from gsp.config import TARGET_MOVE
check("y(t) is defined by day t+1's bar",
      bool(lb["y"].iloc[t]) == bool(h1 >= o1 * (1 + TARGET_MOVE)))
check("fwd_high_ret(t) = High(t+1)/Close(t)-1",
      np.isclose(lb["fwd_high_ret"].iloc[t], h1 / c0 - 1.0))
check("last row's label is NaN (no t+1)", pd.isna(lb["y"].iloc[-1]))

print("== feature_columns guard ==")
sample = full_feats.copy()
sample["y"] = 0.0
sample["fwd_high_ret"] = 0.0
sample["ticker"] = "X"
cols = feature_columns(sample)
check("y and fwd_* excluded from features",
      "y" not in cols and not any(c.startswith("fwd_") for c in cols))

print("== cross-sectional ==")
p1 = full_feats.iloc[-3:].copy(); p1["ticker"] = "AAA"
p2 = (full_feats.iloc[-3:] * 1.5).copy(); p2["ticker"] = "BBB"
panel = pd.concat([p1, p2]).reset_index().rename(columns={"index": "date"}) \
          .set_index(["date", "ticker"]).sort_index()
panel = add_cross_sectional(panel)
r = panel["cs_rank_vol_20"].groupby(level="date").max()
check("per-day cross-sectional rank maxes at 1.0", bool((r == 1.0).all()))

print("== data sanitizer ==")
from gsp.data import _sanitize_bars
sdf = pd.DataFrame({
    "Open": [10.0, 10.0], "High": [10.5, 500.0], "Low": [9.8, 9.9],
    "Close": [10.1, 10.2], "Volume": [1e6, 1e6],
}, index=pd.date_range("2024-01-01", periods=2))
s = _sanitize_bars(sdf)
check("bogus High clipped to body top", s["High"].iloc[1] == 10.2)

print("== trade_return ==")
from gsp.model import trade_return
tt = pd.DataFrame({"fwd_open_ret": [0.0, 0.02], "fwd_high_ret": [0.10, 0.05],
                   "fwd_close_ret": [0.01, -0.03]})
tr = trade_return(tt)
check("hit trade returns TARGET_MOVE", np.isclose(tr[0], TARGET_MOVE))
check("miss trade returns close/open-1", np.isclose(tr[1], 0.97 / 1.02 - 1.0))

print()
if FAILURES:
    sys.exit(f"SELFTEST FAILED: {FAILURES}")
print("SELFTEST PASSED — no lookahead, labels/guards/sanitizer all correct.")
