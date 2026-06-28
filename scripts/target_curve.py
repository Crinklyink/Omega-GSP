"""Show how the #1 pick's hit rate changes as we raise the target from the open.
Aiming higher does NOT make stocks climb more — it makes the target rarer. This
quantifies that tradeoff on real out-of-sample data so the target is chosen with
eyes open. Uses a ticker subsample for speed.

Run:  python scripts/target_curve.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from gsp.dataset import load_dataset
from gsp.model import walk_forward

df = load_dataset()
tk = df.index.get_level_values("ticker").unique().to_numpy()
rng = np.random.default_rng(1)
keep = set(rng.choice(tk, size=int(len(tk) * 0.25), replace=False))
sub = df[df.index.get_level_values("ticker").isin(keep)]
print(f"subsample: {len(keep)} tickers, {len(sub):,} rows")

preds = walk_forward(sub, verbose=False).reset_index()
hi = 1.0 + preds["fwd_high_ret"].to_numpy()
op = 1.0 + preds["fwd_open_ret"].to_numpy()
intraday_from_open = hi / op  # High_{t+1} / Open_{t+1}

op_ret = preds["fwd_open_ret"].to_numpy()
cl_ret = preds["fwd_close_ret"].to_numpy()
COST = 0.0025  # 25 bps round trip

print("\ntgt | base | #1 hit | lift | avg $/trade | ann.(add) | profit?")
print("----|------|--------|------|-------------|-----------|--------")
best = None
for T in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
    hit = (intraday_from_open >= (1.0 + T))
    base = hit.mean()
    tmp = preds.assign(_hit=hit.astype(float))
    idx = tmp.groupby("date")["score"].idxmax()           # the #1 pick each day
    top = tmp.loc[idx]
    th = top["_hit"].to_numpy().astype(bool)
    entry = 1.0 + top["fwd_open_ret"].to_numpy()
    close_mult = 1.0 + top["fwd_close_ret"].to_numpy()
    # buy at open; if intraday high reaches +T from open -> +T, else exit at close
    realized = np.where(th, T, close_mult / entry - 1.0) - COST
    p1 = th.mean(); avg = realized.mean(); ann = avg * 252
    flag = "YES" if avg > 0 else "no"
    print(f"+{T*100:3.0f}%| {base:5.2%}| {p1:5.1%} |{p1/base:4.1f}x| {avg:+8.3%}  | {ann:+7.1%}  | {flag}")
    if best is None or avg > best[1]:
        best = (T, avg)

print(f"\nProfit peak: +{best[0]*100:.0f}% target  (avg {best[1]:+.3%}/trade, untuned).")
print("Higher target = bigger wins but rarer; the peak balances them. The 12-hour")
print("run then tunes the model to push that peak higher.")
