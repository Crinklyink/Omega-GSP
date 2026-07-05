"""Meta-labeling: a second-stage model that looks at each day's top-K picks and
decides TAKE or SKIP.

The first-stage ranker answers "which names are most likely to print +8%?".
This stage answers the money question about exactly those names: "is THIS trade,
on THIS day, likely to be profitable after costs?" — trained only on the picks
the first stage actually surfaced, which is the distribution that matters.

Honesty: strictly walk-forward. The first-stage scores are already out-of-sample
(from models/oos_preds.parquet). The meta model for fold f trains only on picks
from folds < f, so no meta prediction ever sees its own era.

Needs:  models/oos_preds.parquet   (cli.py evaluate)
        data/dataset/dataset.parquet

Run:  python scripts/meta_label.py [--k 5] [--cost-bps 25] [--min-train-folds 8]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsp.config import MODEL_DIR, RANDOM_SEED
from gsp.dataset import load_dataset
from gsp.features import feature_columns
from gsp.intraday import load_cached_hourly
from gsp.model import trade_return

IH_COLS = ["ih_high_early", "ih_bar0_vol_share", "ih_bar0_range", "ih_post1_ret"]


def intraday_history(ticker: str) -> pd.DataFrame | None:
    """Per-session intraday shape stats, rolled over the trailing 10 sessions.
    Point-in-time: the row dated d aggregates sessions up to and including d,
    all fully known at d's close. Only ~2 years exist (yfinance 60m limit) —
    older picks get NaN and LightGBM routes them through its default branch.

      ih_high_early     : share of sessions whose HIGH printed in the first 2 bars
      ih_bar0_vol_share : how front-loaded volume is into the open hour
      ih_bar0_range     : first-hour range as % of the open
      ih_post1_ret      : average drift AFTER the first hour (fade vs grind)
    """
    df = load_cached_hourly(ticker)
    if df is None or len(df) < 40:
        return None
    idx = df.index
    days = pd.Index(idx.tz_convert("America/New_York").date if idx.tz is not None
                    else idx.normalize().date)
    g = df.groupby(days)
    rows = {}
    for day, bars in g:
        if len(bars) < 3:
            continue
        hi_bar = int(np.argmax(bars["High"].to_numpy()))
        vol = bars["Volume"].to_numpy()
        vsum = vol.sum()
        o0 = bars["Open"].iloc[0]
        c0 = bars["Close"].iloc[0]
        rows[pd.Timestamp(day)] = {
            "high_early": 1.0 if hi_bar <= 1 else 0.0,
            "bar0_vol_share": vol[0] / vsum if vsum > 0 else np.nan,
            "bar0_range": (bars["High"].iloc[0] - bars["Low"].iloc[0]) / o0 if o0 > 0 else np.nan,
            "post1_ret": bars["Close"].iloc[-1] / c0 - 1.0 if c0 > 0 else np.nan,
        }
    if len(rows) < 15:
        return None
    sess = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out = sess.rolling(10, min_periods=5).mean()
    out.columns = IH_COLS
    return out

META_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_jobs": -1,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--min-train-folds", type=int, default=8,
                    help="meta predictions start once this many folds of picks exist")
    ap.add_argument("--no-intraday", action="store_true",
                    help="skip the hourly-history features")
    a = ap.parse_args()
    cost = a.cost_bps / 1e4

    preds_path = MODEL_DIR / "oos_preds.parquet"
    if not preds_path.exists():
        sys.exit("No models/oos_preds.parquet — run `cli.py evaluate` first.")
    preds = pd.read_parquet(preds_path)
    preds["date"] = pd.to_datetime(preds["date"])

    # The universe of this exercise: the first stage's daily top-K picks.
    picks = (preds.sort_values("score", ascending=False)
                  .groupby("date", sort=True).head(a.k)
                  .set_index(["date", "ticker"]).sort_index())
    picks["net_ret"] = trade_return(picks) - cost
    picks["y_meta"] = (picks["net_ret"] > 0).astype("float32")
    print(f"[meta] {len(picks):,} picks over {picks.index.get_level_values('date').nunique():,} days; "
          f"{picks['y_meta'].mean():.1%} profitable after {a.cost_bps:.0f}bps")

    # Join the full feature vector for each picked row.
    ds = load_dataset()
    feat_cols = [c for c in feature_columns(ds.reset_index())
                 if c not in ("date", "ticker")]
    X_all = ds[feat_cols].reindex(picks.index)
    X_all["stage1_score"] = picks["score"]
    feat_cols = feat_cols + ["stage1_score"]
    del ds

    # Intraday shape history (last ~2 years only; NaN elsewhere).
    if not a.no_intraday:
        for c in IH_COLS:
            X_all[c] = np.nan
        n_have = 0
        for tk, sub in X_all.groupby(level="ticker", sort=False):
            ih = intraday_history(str(tk))
            if ih is None:
                continue
            dts = sub.index.get_level_values("date")
            vals = ih.reindex(dts)
            X_all.loc[sub.index, IH_COLS] = vals.to_numpy()
            n_have += 1
        feat_cols = feat_cols + IH_COLS
        cov = X_all["ih_high_early"].notna().mean()
        print(f"[meta] intraday history joined for {n_have} tickers "
              f"({cov:.0%} of picks covered)")

    # Expanding walk-forward over the first stage's folds (time-ordered).
    folds = np.sort(picks["fold"].unique())
    rows = []
    for f in folds[a.min_train_folds:]:
        tr_mask = picks["fold"] < f
        te_mask = picks["fold"] == f
        if tr_mask.sum() < 500 or te_mask.sum() == 0:
            continue
        dtrain = lgb.Dataset(X_all[tr_mask], label=picks.loc[tr_mask, "y_meta"])
        booster = lgb.train(META_PARAMS, dtrain, num_boost_round=400)
        te = picks[te_mask].copy()
        te["meta_p"] = booster.predict(X_all[te_mask])
        rows.append(te)
    if not rows:
        sys.exit("Not enough folds for meta training.")
    m = pd.concat(rows)
    print(f"[meta] scored {len(m):,} picks out-of-sample "
          f"(folds {folds[a.min_train_folds]}..{folds[-1]})")

    base = m["net_ret"].mean()
    print(f"\n=====  TAKE/SKIP CURVE (meta threshold on the daily top-{a.k})  =====")
    print(f"  take-all baseline: avg net {base:+.3%} over {len(m):,} trades")
    print(f"  {'thr':>5} {'taken%':>7} {'avg net (taken)':>16} {'win%':>7} {'avg net (skipped)':>18}")
    curve = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        take = m["meta_p"] >= thr
        if take.sum() < 50:
            continue
        row = {
            "threshold": thr,
            "frac_taken": float(take.mean()),
            "n_taken": int(take.sum()),
            "avg_net_taken": float(m.loc[take, "net_ret"].mean()),
            "win_rate_taken": float((m.loc[take, "net_ret"] > 0).mean()),
            "avg_net_skipped": float(m.loc[~take, "net_ret"].mean()) if (~take).any() else None,
        }
        curve.append(row)
        print(f"  {thr:>5.2f} {row['frac_taken']:>7.1%} {row['avg_net_taken']:>+16.3%} "
              f"{row['win_rate_taken']:>7.1%} "
              f"{(row['avg_net_skipped'] if row['avg_net_skipped'] is not None else float('nan')):>+18.3%}")

    out = {"k": a.k, "cost_bps": a.cost_bps, "n_scored": len(m),
           "take_all_avg_net": float(base), "curve": curve}
    (MODEL_DIR / "meta_curve.json").write_text(json.dumps(out, indent=2))
    print(f"\n[meta] saved -> {MODEL_DIR / 'meta_curve.json'}")
    print("[meta] read: if 'avg net (taken)' at some threshold is clearly positive while")
    print("       'skipped' is negative, the meta filter is doing real work — trade only")
    print("       when it says take. If not, the edge isn't separable at daily granularity.")


if __name__ == "__main__":
    main()
