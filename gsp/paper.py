"""Paper-trading ledger — the bridge from backtest to reality.

Every research number in this repo describes the past. The ledger runs the model
FORWARD: each day it logs the scan's top picks (decision made at that close),
and on later days it settles them against what actually happened at the next
session — entry at the open, +TARGET_MOVE limit, close-exit otherwise, plus
conservative stopped variants. After a month or three, the settled stats here
are the only numbers that deserve real trust.

CSV at data/paper_ledger.csv. Append-only by (decision_date, ticker).

Used by:  python cli.py paper          # settle pending + log today's picks
          python cli.py paper --summary-only
          python cli.py daily          # refresh data, then the above
"""
from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR, TARGET_MOVE
from .data import load_cached

LEDGER_PATH = DATA_DIR / "paper_ledger.csv"

COLS = ["decision_date", "ticker", "close", "score", "exp_hit", "logged_at",
        "status", "entry_date", "entry_open", "day_high", "day_low", "day_close",
        "hit_target", "ret_limit", "ret_stop3", "ret_stop5"]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH, parse_dates=["decision_date", "entry_date"])
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def save_ledger(df: pd.DataFrame) -> None:
    df.to_csv(LEDGER_PATH, index=False)


def log_picks(picks: pd.DataFrame) -> int:
    """Append today's scan picks (columns: ticker, asof, close, score[, exp_hit]).
    Dedupes on (decision_date, ticker) so rerunning the same evening is safe."""
    led = load_ledger()
    new_rows = []
    for r in picks.itertuples():
        ddate = pd.Timestamp(r.asof)
        dup = ((led["decision_date"] == ddate) & (led["ticker"] == r.ticker)).any()
        if dup:
            continue
        new_rows.append({
            "decision_date": ddate,
            "ticker": r.ticker,
            "close": float(r.close),
            "score": float(r.score),
            "exp_hit": float(getattr(r, "exp_hit", np.nan))
            if getattr(r, "exp_hit", None) is not None else np.nan,
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "pending",
        })
    if new_rows:
        led = pd.concat([led, pd.DataFrame(new_rows)], ignore_index=True)
        save_ledger(led)
    return len(new_rows)


def settle() -> int:
    """Fill outcomes for pending rows whose next session now exists in the
    daily cache. The trade rule mirrors the backtest exactly: buy the open,
    +TARGET_MOVE limit; if the high never reaches it, exit at the close.
    Stop variants assume (conservatively) the stop fills first when both the
    stop and the target were touched."""
    led = load_ledger()
    pend = led[led["status"] == "pending"]
    if pend.empty:
        return 0
    n_settled = 0
    for i, r in pend.iterrows():
        raw = load_cached(str(r["ticker"]))
        if raw is None or raw.empty:
            # give delisted/missing names 10 days before writing them off
            if (pd.Timestamp.now() - r["decision_date"]).days > 10:
                led.loc[i, "status"] = "no_data"
            continue
        after = raw[raw.index > r["decision_date"]]
        if after.empty:
            continue
        bar = after.iloc[0]
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        if not np.isfinite(o) or o <= 0:
            led.loc[i, "status"] = "no_data"
            continue
        hit = h >= o * (1.0 + TARGET_MOVE)
        ret_limit = TARGET_MOVE if hit else c / o - 1.0
        outs = {"ret_limit": ret_limit}
        for s_name, s in (("ret_stop3", 0.03), ("ret_stop5", 0.05)):
            stopped = l <= o * (1.0 - s)
            outs[s_name] = -s if stopped else ret_limit
        led.loc[i, ["status", "entry_date", "entry_open", "day_high", "day_low",
                    "day_close", "hit_target", "ret_limit", "ret_stop3",
                    "ret_stop5"]] = [
            "settled", after.index[0], o, h, l, c, bool(hit),
            outs["ret_limit"], outs["ret_stop3"], outs["ret_stop5"]]
        n_settled += 1
    if n_settled:
        save_ledger(led)
    return n_settled


def summary(cost_bps: float = 25.0) -> None:
    led = load_ledger()
    if led.empty:
        print("[paper] ledger is empty — run `cli.py paper` after a scan.")
        return
    cost = cost_bps / 1e4
    st = led["status"].value_counts().to_dict()
    print(f"\n=====  PAPER LEDGER  ({LEDGER_PATH})  =====")
    print(f" rows: {len(led)}  {st}")
    s = led[led["status"] == "settled"]
    if s.empty:
        print(" no settled trades yet — outcomes fill in after the next session.")
        return
    print(f" settled trades       : {len(s)}")
    print(f" hit +{TARGET_MOVE:.0%} intraday   : {s['hit_target'].mean():.1%}")
    for col, label in (("ret_limit", "no stop     "),
                       ("ret_stop3", "3% stop     "),
                       ("ret_stop5", "5% stop     ")):
        r = s[col].astype(float) - cost
        print(f" avg trade ({label}): {r.mean():+.3%}   win {(r > 0).mean():.1%}")
    m = s.copy()
    m["month"] = pd.to_datetime(m["decision_date"]).dt.to_period("M")
    by = m.groupby("month").agg(n=("ticker", "size"),
                                hit=("hit_target", "mean"),
                                avg=("ret_limit", "mean"))
    print("\n by month (no stop, pre-cost):")
    for mo, row in by.iterrows():
        print(f"   {mo}: n={int(row['n']):<4} hit {row['hit']:.1%}  avg {row['avg']:+.3%}")
    print("==========================================\n")
