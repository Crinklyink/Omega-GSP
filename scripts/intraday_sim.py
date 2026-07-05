"""Replay the model's out-of-sample top-K trades bar-by-bar against hourly data.

The daily-bar strategy sim cannot know whether the +8% target or the stop-loss
filled first, so it conservatively assumes the stop always wins ties. This script
answers that question with real intraday paths (last ~2 years, the yfinance 60m
window):

  * fill order: target-first vs stop-first vs ambiguous-within-one-bar
  * timing: which bar of the session the +8% print actually happens in
  * an exit-rule grid: stop levels x (hold to close | bail after N bars)

Needs:  models/oos_preds.parquet   (written by `cli.py evaluate`)
        data/raw_60m/*.parquet     (written by scripts/fetch_hourly.py)

Run:  python scripts/intraday_sim.py [--k 5] [--cost-bps 25]
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsp.config import MODEL_DIR, TARGET_MOVE
from gsp.data import load_cached
from gsp.intraday import day_bars, load_cached_hourly

STOPS = [None, 0.03, 0.05, 0.07]
EXIT_BARS = [None, 3]  # None = hold to close; 3 = flatten at end of 3rd hourly bar


def _next_session(daily_index_map: dict, ticker: str, t: pd.Timestamp):
    m = daily_index_map.get(ticker)
    if m is None:
        raw = load_cached(ticker)
        if raw is None or raw.empty:
            daily_index_map[ticker] = {}
            return None
        idx = raw.index
        m = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}
        daily_index_map[ticker] = m
    return m.get(t)


def replay_trade(bars: pd.DataFrame, stop: float | None, exit_bar: int | None):
    """Walk one session's hourly bars. Entry = first bar's open.
    Returns (return, outcome, hit_bar_idx). Ambiguity inside a single bar
    (target AND stop in the same bar) resolves to the stop — still conservative,
    but at hourly granularity that case is rare and we count it."""
    o = float(bars["Open"].iloc[0])
    if not np.isfinite(o) or o <= 0:
        return None
    target = o * (1.0 + TARGET_MOVE)
    stop_px = o * (1.0 - stop) if stop is not None else -np.inf

    last = min(len(bars), exit_bar) if exit_bar is not None else len(bars)
    for i in range(last):
        hi = float(bars["High"].iloc[i])
        lo = float(bars["Low"].iloc[i])
        hit_t = hi >= target
        hit_s = lo <= stop_px
        if hit_t and hit_s:
            return -stop, "ambiguous_stop", i
        if hit_s:
            return -stop, "stopped", i
        if hit_t:
            return TARGET_MOVE, "target", i
    exit_px = float(bars["Close"].iloc[last - 1])
    return exit_px / o - 1.0, "exit", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    a = ap.parse_args()
    cost = a.cost_bps / 1e4

    preds_path = MODEL_DIR / "oos_preds.parquet"
    if not preds_path.exists():
        sys.exit("No models/oos_preds.parquet — run `cli.py evaluate` first.")
    preds = pd.read_parquet(preds_path)

    # Only the window where hourly data can exist.
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=720)
    preds = preds[pd.to_datetime(preds["date"]) >= cutoff]
    if preds.empty:
        sys.exit("No OOS predictions inside the hourly-data window.")
    print(f"[sim] {preds['date'].nunique()} decision days in hourly window, "
          f"top-{a.k}/day, cost={a.cost_bps:.0f}bps")

    top = (preds.sort_values("score", ascending=False)
                .groupby("date", sort=True).head(a.k))

    daily_map: dict = {}
    trades = []          # (ticker, t1, bars)
    n_no_next, n_no_bars = 0, 0
    open_diffs = []
    for r in top.itertuples():
        t1 = _next_session(daily_map, r.ticker, pd.Timestamp(r.date))
        if t1 is None:
            n_no_next += 1
            continue
        bars = day_bars(r.ticker, t1)
        if bars is None or len(bars) < 2:
            n_no_bars += 1
            continue
        raw = load_cached(r.ticker)
        if raw is not None and t1 in raw.index:
            d_open = float(raw.loc[t1, "Open"])
            h_open = float(bars["Open"].iloc[0])
            if d_open > 0:
                open_diffs.append(abs(h_open / d_open - 1.0))
        trades.append((r.ticker, t1, bars))

    total = len(top)
    print(f"[sim] coverage: {len(trades)}/{total} trades replayable "
          f"({n_no_next} no next session, {n_no_bars} no hourly bars)")
    if open_diffs:
        print(f"[sim] hourly-vs-daily open px: median diff "
              f"{np.median(open_diffs):.3%}, p90 {np.quantile(open_diffs, 0.9):.3%}")
    if not trades:
        sys.exit("Nothing to replay — run scripts/fetch_hourly.py first.")

    # Pop-timing histogram (no stop, hold to close)
    timing = Counter()
    for _, _, bars in trades:
        out = replay_trade(bars, None, None)
        if out and out[1] == "target":
            timing[out[2]] += 1
    n_hits = sum(timing.values())
    print(f"\n=====  WHEN DO THE +{TARGET_MOVE:.0%} PRINTS HAPPEN? "
          f"({n_hits} hits / {len(trades)} trades)  =====")
    for bar_idx in sorted(timing):
        share = timing[bar_idx] / max(n_hits, 1)
        print(f"  bar {bar_idx} ({9 + bar_idx}:30-ish ET): {timing[bar_idx]:5d}  "
              f"{share:6.1%}  {'#' * int(share * 60)}")

    print(f"\n=====  EXIT-RULE GRID (top-{a.k}/day, {a.cost_bps:.0f}bps cost)  =====")
    print(f"  {'stop':>6} {'exit':>10} {'avg ret':>9} {'win%':>7} {'target%':>8} "
          f"{'stopped%':>9} {'ambig%':>7}")
    results = []
    for stop in STOPS:
        for exit_bar in EXIT_BARS:
            rets, outcomes = [], Counter()
            for _, _, bars in trades:
                out = replay_trade(bars, stop, exit_bar)
                if out is None:
                    continue
                rets.append(out[0] - cost)
                outcomes[out[1]] += 1
            rets = np.array(rets)
            n = len(rets)
            row = {
                "stop": stop, "exit_bar": exit_bar, "n_trades": n,
                "avg_ret": float(rets.mean()),
                "win_rate": float((rets > 0).mean()),
                "target_rate": outcomes["target"] / n,
                "stop_rate": (outcomes["stopped"] + outcomes["ambiguous_stop"]) / n,
                "ambiguous_rate": outcomes["ambiguous_stop"] / n,
            }
            results.append(row)
            print(f"  {('none' if stop is None else f'{stop:.0%}'):>6} "
                  f"{('close' if exit_bar is None else f'bar {exit_bar}'):>10} "
                  f"{row['avg_ret']:>+9.3%} {row['win_rate']:>7.1%} "
                  f"{row['target_rate']:>8.1%} {row['stop_rate']:>9.1%} "
                  f"{row['ambiguous_rate']:>7.1%}")

    out = {
        "k": a.k, "cost_bps": a.cost_bps, "n_trades": len(trades),
        "coverage": len(trades) / max(total, 1),
        "pop_timing_by_bar": {str(k_): v for k_, v in sorted(timing.items())},
        "grid": results,
    }
    (MODEL_DIR / "intraday_sim.json").write_text(json.dumps(out, indent=2))
    print(f"\n[sim] saved -> {MODEL_DIR / 'intraday_sim.json'}")
    print("[sim] note: hourly bars start 9:30 ET; bar 0 = the open hour. "
          "Ambiguous (target+stop same bar) counts as stopped — conservative.")


if __name__ == "__main__":
    main()
