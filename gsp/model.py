"""LightGBM model + walk-forward (time-ordered) evaluation.

Walk-forward is the only honest way to test this. We train on a block of the past,
leave an embargo gap, then predict a forward test block we never touched. Then we
roll the whole window forward and repeat. Concatenating the test-block predictions
gives a fully out-of-sample track record across years.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import (LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING,
                     WALKFORWARD_TRAIN_YEARS, WALKFORWARD_TEST_MONTHS,
                     EMBARGO_DAYS, RANDOM_SEED)
from .features import feature_columns


def _xy(df: pd.DataFrame, feat_cols: list[str]):
    X = df[feat_cols]
    y = df["y"].astype("float32")
    return X, y


def train_lgb(train: pd.DataFrame, feat_cols: list[str],
              params: dict | None = None, val_frac: float = 0.15,
              num_round: int = NUM_BOOST_ROUND) -> lgb.Booster:
    params = {**LGB_PARAMS, **(params or {})}
    train = train.dropna(subset=["y"]).sort_index()

    # Time-ordered validation tail for early stopping (no shuffling!).
    dates = train.index.get_level_values("date")
    uniq = np.sort(dates.unique())
    cut = uniq[min(int(len(uniq) * (1 - val_frac)), len(uniq) - 1)]
    tr = train[dates <= cut]
    va = train[dates > cut]
    if len(va) < 1000 or va["y"].sum() < 20:
        tr, va = train, train.iloc[-min(len(train), 5000):]

    pos = max(tr["y"].sum(), 1.0)
    neg = max(len(tr) - pos, 1.0)
    params = {**params, "scale_pos_weight": float(neg / pos), "seed": RANDOM_SEED}

    Xtr, ytr = _xy(tr, feat_cols)
    Xva, yva = _xy(va, feat_cols)
    dtrain = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dval = lgb.Dataset(Xva, label=yva, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        params, dtrain, num_boost_round=num_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return booster


def walk_forward(dataset: pd.DataFrame, params: dict | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Return out-of-sample predictions for every test fold, with the forward
    returns attached so backtest.py can simulate a strategy."""
    feat_cols = feature_columns(dataset.reset_index())
    feat_cols = [c for c in feat_cols if c not in ("date", "ticker")]

    df = dataset.dropna(subset=["y"]).copy()
    all_dates = df.index.get_level_values("date")
    start, end = all_dates.min(), all_dates.max()

    train_off = pd.DateOffset(years=WALKFORWARD_TRAIN_YEARS)
    test_off = pd.DateOffset(months=WALKFORWARD_TEST_MONTHS)
    embargo = pd.Timedelta(days=EMBARGO_DAYS)

    preds = []
    fold = 0
    test_start = start + train_off + embargo
    while test_start < end:
        train_end = test_start - embargo
        train_start = train_end - train_off
        test_end = test_start + test_off

        tr_mask = (all_dates >= train_start) & (all_dates < train_end)
        te_mask = (all_dates >= test_start) & (all_dates < test_end)
        train = df[tr_mask]
        test = df[te_mask]

        if len(train) < 5000 or len(test) < 200 or train["y"].sum() < 50:
            test_start = test_start + test_off
            continue

        booster = train_lgb(train, feat_cols, params)
        p = booster.predict(test[feat_cols], num_iteration=booster.best_iteration)
        fwd = [c for c in ("fwd_high_ret", "fwd_open_ret", "fwd_close_ret",
                           "fwd_low_ret") if c in test.columns]
        res = test[["y"] + fwd].copy()
        res["score"] = p
        res["fold"] = fold
        preds.append(res)

        if verbose:
            base = test["y"].mean()
            print(f"[wf] fold {fold}: train {train_start.date()}..{train_end.date()} "
                  f"test {test_start.date()}..{test_end.date()} "
                  f"n_test={len(test):,} base_rate={base:.3%}")
        fold += 1
        test_start = test_start + test_off

    if not preds:
        raise RuntimeError("Walk-forward produced no folds — not enough history?")
    return pd.concat(preds)


def train_final(dataset: pd.DataFrame, params: dict | None = None) -> tuple[lgb.Booster, list[str]]:
    """Train one model on ALL available labelled data, for live scanning."""
    feat_cols = feature_columns(dataset.reset_index())
    feat_cols = [c for c in feat_cols if c not in ("date", "ticker")]
    booster = train_lgb(dataset, feat_cols, params)
    return booster, feat_cols
