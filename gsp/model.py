"""LightGBM models + walk-forward (time-ordered) evaluation.

Walk-forward is the only honest way to test this. We train on a block of the past,
leave an embargo gap, then predict a forward test block we never touched. Then we
roll the whole window forward and repeat. Concatenating the test-block predictions
gives a fully out-of-sample track record across years.

Two upgrades over the original single-model version:

1. Early stopping uses average_precision (PR-AUC) instead of ROC-AUC. With a ~4%
   positive class, PR-AUC is the metric that actually tracks top-of-ranking
   quality — which is what precision@K cares about.
2. Seed-bagged ENSEMBLE: `n_models` LightGBM boosters trained with different
   seeds (different row/column subsample draws), predictions averaged. Averaging
   uncorrelated tree noise reliably tightens the top of the ranking, which is
   exactly where we harvest precision@1..5.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import (LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING, ENSEMBLE_N,
                     WALKFORWARD_TRAIN_YEARS, WALKFORWARD_TEST_MONTHS,
                     EMBARGO_DAYS, RANDOM_SEED, TARGET_MOVE)
from .features import feature_columns


def _xy(df: pd.DataFrame, feat_cols: list[str], label_col: str = "y"):
    X = df[feat_cols]
    y = df[label_col].astype("float32")
    return X, y


def trade_return(df: pd.DataFrame) -> np.ndarray:
    """Realized (cost-free) return of the standard trade rule: buy next open,
    +TARGET_MOVE limit, else exit at that day's close. This is the EV-mode
    regression label — built ONLY from fwd_* columns, i.e. the same forward
    data the classification label already uses."""
    entry = 1.0 + df["fwd_open_ret"].to_numpy()
    hi = 1.0 + df["fwd_high_ret"].to_numpy()
    cl = 1.0 + df["fwd_close_ret"].to_numpy()
    hit = hi >= entry * (1.0 + TARGET_MOVE)
    return np.where(hit, TARGET_MOVE, cl / entry - 1.0).astype("float32")


def _time_split(train: pd.DataFrame, val_frac: float):
    """Time-ordered validation tail for early stopping (no shuffling!)."""
    dates = train.index.get_level_values("date")
    uniq = np.sort(dates.unique())
    cut = uniq[min(int(len(uniq) * (1 - val_frac)), len(uniq) - 1)]
    tr = train[dates <= cut]
    va = train[dates > cut]
    if len(va) < 1000 or va["y"].sum() < 20:
        tr, va = train, train.iloc[-min(len(train), 5000):]
    return tr, va


def train_lgb(train: pd.DataFrame, feat_cols: list[str],
              params: dict | None = None, val_frac: float = 0.15,
              num_round: int = NUM_BOOST_ROUND,
              seed: int = RANDOM_SEED, label_col: str = "y",
              regression: bool = False) -> lgb.Booster:
    params = {**LGB_PARAMS, **(params or {})}
    train = train.dropna(subset=[label_col]).sort_index()
    tr, va = _time_split(train, val_frac)

    spw_mult = float(params.pop("spw_mult", 1.0))  # searchable multiplier
    max_bin = int(params.pop("max_bin", 255))      # dataset-level param, not train-level
    if regression:
        # Huber loss: the trade-return label has fat tails (halts, crashes);
        # squared error would let a few -60% days dominate the fit.
        params = {**params, "objective": "huber", "metric": ["l1"]}
        params.pop("scale_pos_weight", None)
    else:
        pos = max(tr[label_col].sum(), 1.0)
        neg = max(len(tr) - pos, 1.0)
        params = {**params,
                  "scale_pos_weight": float(neg / pos) * spw_mult,
                  "metric": ["average_precision"]}
    params = {**params,
              "seed": seed,
              "bagging_seed": seed + 1,
              "feature_fraction_seed": seed + 2}

    ds_params = {"max_bin": max_bin, "verbosity": -1}
    Xtr, ytr = _xy(tr, feat_cols, label_col)
    Xva, yva = _xy(va, feat_cols, label_col)
    dtrain = lgb.Dataset(Xtr, label=ytr, params=ds_params, free_raw_data=False)
    dval = lgb.Dataset(Xva, label=yva, reference=dtrain, params=ds_params,
                       free_raw_data=False)

    booster = lgb.train(
        params, dtrain, num_boost_round=num_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True,
                                      verbose=False),
                   lgb.log_evaluation(0)],
    )
    return booster


def train_ensemble(train: pd.DataFrame, feat_cols: list[str],
                   params: dict | None = None, n_models: int = ENSEMBLE_N,
                   val_frac: float = 0.15, label_col: str = "y",
                   regression: bool = False) -> list[lgb.Booster]:
    """Train `n_models` seed-varied boosters on the same window."""
    models = []
    for i in range(max(1, n_models)):
        models.append(train_lgb(train, feat_cols, params, val_frac=val_frac,
                                seed=RANDOM_SEED + 101 * i,
                                label_col=label_col, regression=regression))
    return models


def predict_ensemble(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X, num_iteration=m.best_iteration) for m in models]
    return np.mean(preds, axis=0)


def walk_forward(dataset: pd.DataFrame, params: dict | None = None,
                 verbose: bool = True, n_models: int = 1,
                 on_fold: Callable[[int, pd.DataFrame], None] | None = None,
                 target: str = "hit") -> pd.DataFrame:
    """Return out-of-sample predictions for every test fold, with the forward
    returns attached so backtest.py can simulate a strategy.

    n_models=1 keeps single-model speed (leakage tests, optuna trials);
    n_models=ENSEMBLE_N is used for the real evaluation and final training.
    on_fold(fold_idx, fold_preds) is called after each fold — the optimizer uses
    it to report intermediate scores and prune hopeless trials early.

    target="hit" (default): binary classifier on y = touched +TARGET_MOVE.
    target="ev": Huber regression on the realized trade return — ranks names by
    what the trade is WORTH, not how likely it is to print. The classification
    y stays in the output so ranking metrics remain comparable across modes.
    """
    feat_cols = feature_columns(dataset.reset_index())
    feat_cols = [c for c in feat_cols if c not in ("date", "ticker")]

    df = dataset.dropna(subset=["y"]).copy()
    regression = target == "ev"
    label_col = "y"
    if regression:
        if "fwd_open_ret" not in df.columns:
            raise ValueError("EV mode needs fwd_* columns in the dataset")
        df["ev_label"] = trade_return(df)
        label_col = "ev_label"
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

        models = train_ensemble(train, feat_cols, params, n_models=n_models,
                                label_col=label_col, regression=regression)
        p = predict_ensemble(models, test[feat_cols])
        # keep liquidity so the backtest can price slippage per name
        extra = [c for c in ("fwd_high_ret", "fwd_open_ret", "fwd_close_ret",
                             "fwd_low_ret", "log_dollar_vol_20")
                 if c in test.columns]
        res = test[["y"] + extra].copy()
        res["score"] = p
        res["fold"] = fold
        preds.append(res)

        if verbose:
            base = test["y"].mean()
            print(f"[wf] fold {fold}: train {train_start.date()}..{train_end.date()} "
                  f"test {test_start.date()}..{test_end.date()} "
                  f"n_test={len(test):,} base_rate={base:.3%}", flush=True)
        if on_fold is not None:
            on_fold(fold, res)
        fold += 1
        test_start = test_start + test_off

    if not preds:
        raise RuntimeError("Walk-forward produced no folds — not enough history?")
    return pd.concat(preds)


def train_final(dataset: pd.DataFrame, params: dict | None = None,
                n_models: int = ENSEMBLE_N) -> tuple[list[lgb.Booster], list[str]]:
    """Train the live-scanning ensemble on ALL available labelled data."""
    feat_cols = feature_columns(dataset.reset_index())
    feat_cols = [c for c in feat_cols if c not in ("date", "ticker")]
    models = train_ensemble(dataset, feat_cols, params, n_models=n_models)
    return models, feat_cols
