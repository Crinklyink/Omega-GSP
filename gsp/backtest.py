"""Honest evaluation of out-of-sample walk-forward predictions.

Two views, both reported:

1) RANKING QUALITY (does the model find the pops?)
   - ROC-AUC and PR-AUC over all rows
   - precision@K per day vs the base rate  ->  LIFT. Lift > 1 means the model's
     top picks pop more often than a random liquid stock. This is the edge.

2) REAL MONEY (what a tradeable rule actually earns)
   We buy the top-K names at the NEXT OPEN (the earliest you could act on a close
   signal) and place a +8% limit. If the day's high reaches it -> +8%. Otherwise
   we exit at that day's close. This bakes in the overnight gap you cannot trade,
   which is the difference between a backtest and a fantasy.

Costs: set `cost_bps` for round-trip slippage+commission (default 10 bps).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import TARGET_MOVE, TOP_K


def ranking_metrics(preds: pd.DataFrame) -> dict:
    y = preds["y"].to_numpy()
    s = preds["score"].to_numpy()
    out = {"n_rows": int(len(preds)), "base_rate": float(y.mean())}
    if y.min() != y.max():
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def precision_at_k(preds: pd.DataFrame, k: int = TOP_K) -> dict:
    """Each day take the K highest-scoring names; what fraction actually hit y=1?"""
    df = preds.reset_index()
    hits, picks = 0, 0
    per_day = []
    for _, day in df.groupby("date"):
        top = day.nlargest(k, "score")
        h = float(top["y"].sum())
        hits += h
        picks += len(top)
        per_day.append(h / len(top))
    prec = hits / max(picks, 1)
    base = float(df["y"].mean())
    return {
        "k": k,
        "precision_at_k": prec,
        "base_rate": base,
        "lift": prec / base if base > 0 else float("nan"),
        "days": len(per_day),
        "avg_daily_precision": float(np.mean(per_day)) if per_day else float("nan"),
    }


def precision_curve(preds: pd.DataFrame, ks=(1, 2, 3, 5, 10)) -> list[dict]:
    """How precision (fraction of picks that actually popped) changes as we get
    MORE selective. Fewer picks/day -> higher hit rate. This is the honest answer
    to 'can it hit +8% almost always?' — only by being very selective, and even
    then nowhere near 100%."""
    df = preds.reset_index()
    base = float(df["y"].mean())
    out = []
    for k in ks:
        hits = picks = 0
        for _, day in df.groupby("date"):
            top = day.nlargest(k, "score")
            hits += float(top["y"].sum()); picks += len(top)
        prec = hits / max(picks, 1)
        out.append({"k": k, "precision": prec, "lift": prec / base if base else float("nan")})
    return out


def precision_by_threshold(preds: pd.DataFrame, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
    """Precision and coverage if we only act when score >= threshold. Higher bar
    = higher hit rate but fewer days you trade at all."""
    y = preds["y"].to_numpy(); s = preds["score"].to_numpy()
    n_days = preds.reset_index()["date"].nunique()
    out = []
    for th in thresholds:
        m = s >= th
        n = int(m.sum())
        out.append({
            "threshold": th,
            "n_signals": n,
            "precision": float(y[m].mean()) if n else float("nan"),
            "signals_per_day": n / max(n_days, 1),
        })
    return out


def simulate_strategy(preds: pd.DataFrame, k: int = TOP_K, cost_bps: float = 10.0,
                      stop: float | None = None) -> dict:
    """Buy top-K at next open, +TARGET_MOVE limit, else exit at close. Equal weight.

    If `stop` is set (e.g. 0.03), also place a stop-loss `stop` below the open. When
    the next-day LOW pierces the stop we exit at -stop. Intraday order of stop vs
    target is unknown from daily bars, so if BOTH are touched we conservatively
    assume the stop filled first (a lower bound on the strategy's true P&L)."""
    df = preds.reset_index()
    cost = cost_bps / 1e4
    daily_returns = []
    n_trades = 0
    n_hit_target = 0
    n_stopped = 0
    trade_rets = []
    has_low = "fwd_low_ret" in df.columns

    for date, day in df.groupby("date"):
        top = day.nlargest(k, "score")
        if top.empty:
            continue
        # entry at next open (relative to close_t), target +X% above THAT open.
        entry = 1.0 + top["fwd_open_ret"].to_numpy()          # open / close_t
        high_mult = 1.0 + top["fwd_high_ret"].to_numpy()      # high / close_t
        close_mult = 1.0 + top["fwd_close_ret"].to_numpy()    # close / close_t
        target = entry * (1.0 + TARGET_MOVE)
        hit = high_mult >= target
        if stop is not None and has_low:
            low_mult = 1.0 + top["fwd_low_ret"].to_numpy()    # low / close_t
            stopped = low_mult <= entry * (1.0 - stop)
            # conservative: stop first if both touched
            realized = np.where(stopped, -stop,
                                np.where(hit, TARGET_MOVE, close_mult / entry - 1.0))
            n_stopped += int(stopped.sum())
        else:
            realized = np.where(hit, TARGET_MOVE, close_mult / entry - 1.0)
        realized = realized - cost  # round-trip cost
        trade_rets.extend(realized.tolist())
        n_trades += len(realized)
        n_hit_target += int(hit.sum())
        daily_returns.append(float(np.mean(realized)))  # equal weight that day

    if not daily_returns:
        return {"error": "no trades"}

    dr = np.array(daily_returns)
    equity = np.cumprod(1.0 + dr)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
    tr = np.array(trade_rets)

    return {
        "k": k,
        "cost_bps": cost_bps,
        "stop": stop,
        "stop_rate": n_stopped / max(n_trades, 1),
        "n_trading_days": len(dr),
        "n_trades": n_trades,
        "hit_8pct_rate": n_hit_target / max(n_trades, 1),
        # Per-trade stats are the trustworthy core — no compounding assumptions.
        "avg_trade_return": float(tr.mean()),
        "median_trade_return": float(np.median(tr)),
        "win_rate": float((tr > 0).mean()),
        "avg_daily_return": float(dr.mean()),
        # Additive (un-compounded) sum of daily returns: conservative, realistic.
        "additive_return_per_yr": float(dr.mean() * 252),
        # Compounded figures ASSUME perfect daily 100% reinvestment & fills =>
        # wildly optimistic. Reported for completeness, NOT to be believed.
        "compounded_ann_return_FANTASY": float(equity[-1] ** (252 / len(dr)) - 1.0),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "final_equity_mult": float(equity[-1]),
    }


def full_report(preds: pd.DataFrame, k: int = TOP_K, cost_bps: float = 10.0) -> dict:
    rep = {
        "ranking": ranking_metrics(preds),
        "precision_at_k": precision_at_k(preds, k),
        "precision_curve": precision_curve(preds),
        "precision_by_threshold": precision_by_threshold(preds),
        "strategy": simulate_strategy(preds, k, cost_bps),
    }
    return rep


def print_report(rep: dict) -> None:
    r, p, s = rep["ranking"], rep["precision_at_k"], rep["strategy"]
    print("\n================  OUT-OF-SAMPLE REPORT  ================")
    print(f" rows scored          : {r['n_rows']:,}")
    print(f" base rate (8% pops)  : {r['base_rate']:.3%}")
    if "roc_auc" in r:
        print(f" ROC-AUC              : {r['roc_auc']:.4f}  (0.5 = no skill)")
        print(f" PR-AUC               : {r['pr_auc']:.4f}  (vs base {r['base_rate']:.3%})")
    print(f"\n picks/day (K)        : {p['k']}")
    print(f" precision@K          : {p['precision_at_k']:.3%}")
    print(f" LIFT over base       : {p['lift']:.2f}x   <-- the edge")
    if "precision_curve" in rep:
        print("\n selectivity -> hit rate (how often picks actually popped +8%):")
        for c in rep["precision_curve"]:
            print(f"   top-{c['k']:<2}/day : {c['precision']:6.2%}  ({c['lift']:.1f}x base)")
        print("   ^ even the single best pick/day is far from 'always'. That's reality.")
    if "error" not in s:
        print(f"\n --- realistic strategy (buy next open, +8% limit, {s['cost_bps']:.0f}bps cost) ---")
        print(f" trades               : {s['n_trades']:,} over {s['n_trading_days']:,} days")
        print(f" hit +8% intraday     : {s['hit_8pct_rate']:.2%}")
        print(f" avg trade return     : {s['avg_trade_return']:+.3%}   <-- trust this")
        print(f" win rate             : {s['win_rate']:.2%}")
        print(f" additive return/yr   : {s['additive_return_per_yr']:+.2%}   (no compounding, conservative)")
        print(f" Sharpe (annualized)  : {s['sharpe_annualized']:.2f}")
        print(f" max drawdown         : {s['max_drawdown']:.2%}")
        print(f" compounded ann.      : {s['compounded_ann_return_FANTASY']:+.1%}  <-- FANTASY (assumes perfect")
        print(f"                        daily reinvest+fills; ignore it)")
    print("=======================================================\n")
