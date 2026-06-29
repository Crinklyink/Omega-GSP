"""Simulate the real trading plan on out-of-sample data:
  buy the #1 pick at the next OPEN, set a +8% sell-limit AND a stop-loss.

Finds which stop level actually makes money. Uses the tuned best_params and a
ticker subsample for speed (subsampling lowers the daily candidate pool, so the
#1 hit-rate here is a touch CONSERVATIVE vs the full-universe ~60%).

Run:  python scripts/stoploss_curve.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from gsp.config import MODEL_DIR, TARGET_MOVE
from gsp.dataset import load_dataset
from gsp.model import walk_forward
from gsp.backtest import simulate_strategy, precision_curve

bp = MODEL_DIR / "best_params.json"
params = json.loads(bp.read_text())["best_params"] if bp.exists() else None
print("using", "tuned best_params" if params else "default params")

df = load_dataset()
tk = df.index.get_level_values("ticker").unique().to_numpy()
rng = np.random.default_rng(7)
keep = set(rng.choice(tk, size=int(len(tk) * 0.5), replace=False))
sub = df[df.index.get_level_values("ticker").isin(keep)]
print(f"subsample: {len(keep)} tickers, {len(sub):,} rows")

preds = walk_forward(sub, params=params, verbose=False)
p1 = next(c for c in precision_curve(preds, ks=(1,)))
print(f"\n#1-pick hit rate (+{int(TARGET_MOVE*100)}% from open): {p1['precision']:.1%}  ({p1['lift']:.1f}x base)\n")

K = 1  # the single best pick — what you actually trade
print(f"Strategy: buy №1 at open, +{int(TARGET_MOVE*100)}% sell-limit, 25bps cost, equal 1/day")
print("stop  | stop-rate | win% | avg $/trade | ann.(add) | Sharpe | maxDD | profit?")
print("------|-----------|------|-------------|-----------|--------|-------|-------")
best = None
for stop in (None, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
    s = simulate_strategy(preds, k=K, cost_bps=25, stop=stop)
    lab = "none " if stop is None else f"-{stop*100:.0f}%  "
    ann = s["additive_return_per_yr"]; avg = s["avg_trade_return"]
    flag = "YES" if avg > 0 else "no"
    print(f"{lab}| {s['stop_rate']:8.1%} |{s['win_rate']:5.0%} | {avg:+8.3%}  | {ann:+7.1%} "
          f"| {s['sharpe_annualized']:5.2f}  |{s['max_drawdown']:5.0%} | {flag}")
    if best is None or avg > best[1]:
        best = (stop, avg, s)

bs, ba, bd = best
lab = "no stop" if bs is None else f"-{bs*100:.0f}% stop"
print(f"\nBest: {lab}  ->  {ba:+.3%}/trade, Sharpe {bd['sharpe_annualized']:.2f}, "
      f"maxDD {bd['max_drawdown']:.0%}")
print("Note: 'stop first if both touched' => this is a conservative LOWER bound.")
print("Real fills, slippage and intraday order will differ — paper-trade to confirm.")
