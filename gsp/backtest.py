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


def risk_on_dates() -> set | None:
    """Days where SPY closed above its 50-day SMA — a dumb-but-honest regime
    gate, known at each day's close (point-in-time)."""
    from .data import load_market
    mkt = load_market()
    if mkt is None or mkt.empty:
        return None
    c = mkt["Close"]
    sma50 = c.rolling(50, min_periods=50).mean()
    return set(c.index[c > sma50])


def simulate_strategy(preds: pd.DataFrame, k: int = TOP_K, cost_bps: float = 10.0,
                      stop: float | None = None, gate_dates: set | None = None,
                      adaptive_cost: bool = False) -> dict:
    """Buy top-K at next open, +TARGET_MOVE limit, else exit at close. Equal weight.

    If `stop` is set (e.g. 0.03), also place a stop-loss `stop` below the open. When
    the next-day LOW pierces the stop we exit at -stop. Intraday order of stop vs
    target is unknown from daily bars, so if BOTH are touched we conservatively
    assume the stop filled first (a lower bound on the strategy's true P&L).

    gate_dates: if given, only trade on those decision dates (e.g. risk-on days).
    adaptive_cost: charge thin names more. Crude ADV-based slippage proxy —
    +15 bps per unit of log-dollar-volume below ~$5M ADV, capped at +45 bps —
    because a flat 25 bps for a $1M-ADV microcap is a fantasy."""
    df = preds.reset_index()
    cost = cost_bps / 1e4
    daily_returns = []
    n_trades = 0
    n_hit_target = 0
    n_stopped = 0
    trade_rets = []
    has_low = "fwd_low_ret" in df.columns
    has_liq = adaptive_cost and "log_dollar_vol_20" in df.columns

    for date, day in df.groupby("date"):
        if gate_dates is not None and date not in gate_dates:
            continue
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
        if has_liq:
            logdv = top["log_dollar_vol_20"].to_numpy()
            extra = np.clip((15.5 - logdv) * 15.0, 0.0, 45.0) / 1e4
            realized = realized - cost - np.nan_to_num(extra)
        else:
            realized = realized - cost  # flat round-trip cost
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
        "adaptive_cost": bool(has_liq),
        "gated": gate_dates is not None,
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


CALIBRATION_QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)


def calibration_table(preds: pd.DataFrame,
                      quantiles=CALIBRATION_QUANTILES) -> list[dict]:
    """Score-percentile -> historical hit rate, from out-of-sample predictions.
    Lets the live scan say 'names scoring in this band hit +8% X% of the time'
    instead of showing an uncalibrated 0-1 signal. Fine bins at the top because
    that's where all the action (and all the money) is."""
    s = preds["score"].to_numpy()
    y = preds["y"].to_numpy()
    edges = np.quantile(s, quantiles)
    out = []
    for i in range(len(quantiles) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (s >= lo) & ((s < hi) if i < len(quantiles) - 2 else (s <= hi))
        if m.sum() == 0:
            continue
        out.append({
            "pctl_lo": quantiles[i], "pctl_hi": quantiles[i + 1],
            "score_lo": float(lo), "score_hi": float(hi),
            "n": int(m.sum()), "hit_rate": float(y[m].mean()),
        })
    return out


def expected_hit_rate(score: float, table: list[dict]) -> float | None:
    """Look a live score up in the calibration table."""
    if not table:
        return None
    for row in table:
        if score < row["score_hi"]:
            return row["hit_rate"]
    return table[-1]["hit_rate"]


def score_gate_curve(preds: pd.DataFrame, k: int = TOP_K, cost_bps: float = 10.0,
                     window: int = 60,
                     percentiles=(0.5, 0.7, 0.85, 0.95)) -> list[dict]:
    """No-trade filter: some days even the best-ranked name is weak. Each day,
    compare the day's TOP score to the trailing `window` days of daily top
    scores; trade only if it clears the given percentile. Point-in-time honest —
    the threshold uses only PAST out-of-sample scores (shifted one day).

    Score scales drift a little at fold boundaries (each quarter retrains the
    model); a live quarterly-retrain deployment faces exactly the same drift, so
    this is a fair simulation of the rule."""
    df = preds.reset_index()
    daily_top = df.groupby("date")["score"].max().sort_index()
    out = []
    for p in percentiles:
        thr = daily_top.shift(1).rolling(window, min_periods=max(20, window // 2)) \
                       .quantile(p)
        ok = (daily_top >= thr) & thr.notna()
        gate = set(daily_top.index[ok])
        sim = simulate_strategy(preds, k=k, cost_bps=cost_bps, gate_dates=gate)
        out.append({
            "percentile": p,
            "days_traded": sim.get("n_trading_days"),
            "avg_trade_return": sim.get("avg_trade_return"),
            "hit_rate": sim.get("hit_8pct_rate"),
            "win_rate": sim.get("win_rate"),
        })
    return out


def yearly_breakdown(preds: pd.DataFrame, k: int = TOP_K,
                     cost_bps: float = 10.0) -> list[dict]:
    """Same headline metrics, one row per calendar year. A real edge shows up
    in most years; an edge that lives entirely in 2020-21 froth is a warning."""
    years = preds.index.get_level_values("date").year
    out = []
    for yr in sorted(np.unique(years)):
        sub = preds[years == yr]
        if sub.empty or sub["y"].sum() == 0:
            continue
        curve = {c["k"]: c for c in precision_curve(sub, ks=(1, k))}
        strat = simulate_strategy(sub, k, cost_bps)
        out.append({
            "year": int(yr),
            "n_rows": int(len(sub)),
            "base_rate": float(sub["y"].mean()),
            "precision_at_1": curve.get(1, {}).get("precision"),
            "lift_at_1": curve.get(1, {}).get("lift"),
            f"precision_at_{k}": curve.get(k, {}).get("precision"),
            "avg_trade_return": strat.get("avg_trade_return"),
        })
    return out


def full_report(preds: pd.DataFrame, k: int = TOP_K, cost_bps: float = 10.0) -> dict:
    rep = {
        "ranking": ranking_metrics(preds),
        "precision_at_k": precision_at_k(preds, k),
        "precision_curve": precision_curve(preds),
        "precision_by_threshold": precision_by_threshold(preds),
        "strategy": simulate_strategy(preds, k, cost_bps),
        "strategy_adaptive_cost": simulate_strategy(preds, k, cost_bps,
                                                    adaptive_cost=True),
        "score_gate_curve": score_gate_curve(preds, k, cost_bps),
        "yearly": yearly_breakdown(preds, k, cost_bps),
    }
    try:
        gate = risk_on_dates()
        if gate:
            rep["strategy_risk_on"] = simulate_strategy(preds, k, cost_bps,
                                                        gate_dates=gate)
    except Exception as e:  # noqa: BLE001
        print(f"[backtest] regime gate skipped: {e}")
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
    sa = rep.get("strategy_adaptive_cost")
    if sa and "error" not in sa:
        print(f"\n with ADV-based slippage (thin names pay up to +45bps):")
        print(f" avg trade return     : {sa['avg_trade_return']:+.3%}   win rate {sa['win_rate']:.1%}")
    sg = rep.get("strategy_risk_on")
    if sg and "error" not in sg:
        print(f" risk-on days only (SPY>SMA50), flat cost:")
        print(f" avg trade return     : {sg['avg_trade_return']:+.3%}   win rate {sg['win_rate']:.1%}  "
              f"({sg['n_trading_days']:,} days traded)")
    if rep.get("score_gate_curve"):
        print("\n no-trade filter (only trade when today's top score clears the")
        print(" trailing-60d percentile of past top scores):")
        for g in rep["score_gate_curve"]:
            if g.get("avg_trade_return") is None:
                continue
            print(f"   p{int(g['percentile']*100):<3}: avg trade {g['avg_trade_return']:+.3%}  "
                  f"hit {g['hit_rate']:.1%}  win {g['win_rate']:.1%}  "
                  f"({g['days_traded']:,} days traded)")
    if rep.get("yearly"):
        print("\n year-by-year (top-1 hit rate / lift / avg trade ret @K):")
        for yr in rep["yearly"]:
            p1 = yr.get("precision_at_1")
            l1 = yr.get("lift_at_1")
            tr_ = yr.get("avg_trade_return")
            print(f"   {yr['year']}: prec@1 {p1:6.1%}  lift {l1:5.1f}x  "
                  f"avg trade {tr_:+.3%}  (base {yr['base_rate']:.2%})")
    print("=======================================================\n")
