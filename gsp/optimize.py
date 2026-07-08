"""Hyperparameter search — this is what eats the 12-hour budget productively.

Each Optuna trial runs a full walk-forward and is scored by its OUT-OF-SAMPLE
precision@K lift (with small stabilizers). Because every trial is judged on data
it never trained on, the search optimizes for a *real* edge, not for memorizing
the past.

Two things make the 12 hours go further than the original version:

1. Fold-level PRUNING: after each walk-forward fold the trial reports its
   cumulative precision@1; trials clearly below the median of past trials at the
   same fold get killed early. Bad configs cost minutes instead of an hour.
2. Trials train a SINGLE model (fast); the winning params are later retrained as
   the full seed-bagged ensemble by `train --best`, which only helps.
"""
from __future__ import annotations
import json
import warnings

import numpy as np
import optuna
import pandas as pd

from .config import MODEL_DIR, RANDOM_SEED, TOP_K
from .model import walk_forward
from .backtest import precision_at_k, ranking_metrics, precision_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Don't prune before this many folds have been scored — early folds are noisy.
PRUNE_WARMUP_FOLDS = 6


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 3000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
        # multiplier on the auto class weight: <1 leans on ranking loss shape,
        # >1 pushes harder on recall of the rare positives.
        "spw_mult": trial.suggest_float("spw_mult", 0.5, 4.0, log=True),
    }


def optimize(dataset, timeout_hours: float = 12.0, k: int = 1,
             study_name: str = "gsp_cp_v1", sample_frac: float = 1.0,
             holdout_months: int = 0) -> dict:
    # Study names are tied to the LABEL ERA: trials scored on different labels
    # can never share a leaderboard. "gsp_p1_v2" = the old any-touch +8% label;
    # "gsp_cp_v1" = the clean_pop label (2026-07-08). Bump this whenever the
    # target definition changes.
    """Hunt hyperparameters that maximize OUT-OF-SAMPLE precision@`k`. Default k=1:
    we optimize for the single best pick each day being a real +8% mover — exactly
    the 'I only care about the top 1' objective. A small precision@3 + PR-AUC bonus
    stabilizes the noisy top-1 signal without changing what we're chasing.

    sample_frac<1 trains each trial on a fixed random SUBSET of tickers so the 12-hour
    budget buys many more trials on a big universe. The same subset is used for every
    trial (so scores are comparable); the winning params are later retrained on the
    FULL universe by `train --best`.

    holdout_months>0 removes the most recent N months from the search entirely —
    a vault the param selection never touches, so `evaluate --best` on that
    window is an honest final exam. The study name gets a suffix so vaulted and
    unvaulted trials never share a leaderboard (their scores aren't comparable)."""
    if holdout_months > 0:
        dates = dataset.index.get_level_values("date")
        cutoff = dates.max() - pd.DateOffset(months=holdout_months)
        dataset = dataset[dates < cutoff]
        study_name = f"{study_name}_h{holdout_months}"
        print(f"[opt] holding out data from {cutoff.date()} onward "
              f"({len(dataset):,} rows remain; study '{study_name}')", flush=True)
    if sample_frac < 1.0:
        tickers = dataset.index.get_level_values("ticker").unique().to_numpy()
        rng = np.random.default_rng(RANDOM_SEED)
        keep = set(rng.choice(tickers, size=max(50, int(len(tickers) * sample_frac)),
                              replace=False).tolist())
        mask = dataset.index.get_level_values("ticker").isin(keep)
        dataset = dataset[mask]
        print(f"[opt] sampling {len(keep)} of {len(tickers)} tickers for the search "
              f"({len(dataset):,} rows)", flush=True)

    storage = f"sqlite:///{MODEL_DIR / 'optuna.db'}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=15,
                                           multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                           n_warmup_steps=PRUNE_WARMUP_FOLDS,
                                           interval_steps=2),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        fold_preds: list[pd.DataFrame] = []

        def on_fold(fold_idx: int, res: pd.DataFrame) -> None:
            fold_preds.append(res)
            if fold_idx < PRUNE_WARMUP_FOLDS:
                return
            cum = pd.concat(fold_preds)
            p1 = precision_curve(cum, ks=(1,))[0]["precision"]
            trial.report(p1 if np.isfinite(p1) else 0.0, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at fold {fold_idx} (cum prec@1={p1:.3%})")

        try:
            preds = walk_forward(dataset, params=params, verbose=False,
                                 n_models=1, on_fold=on_fold)
        except optuna.TrialPruned:
            raise
        except Exception as e:  # noqa: BLE001
            raise optuna.TrialPruned() from e

        curve = {c["k"]: c for c in precision_curve(preds, ks=(1, 3, 5))}
        rk = ranking_metrics(preds)
        p1 = curve.get(k, curve.get(1, {})).get("precision", 0.0)
        p3 = curve.get(3, {}).get("precision", 0.0)
        trial.set_user_attr("precision_at_1", curve.get(1, {}).get("precision"))
        trial.set_user_attr("precision_at_1_lift", curve.get(1, {}).get("lift"))
        trial.set_user_attr("precision_at_3", p3)
        trial.set_user_attr("pr_auc", rk.get("pr_auc", float("nan")))
        # Primary precision@1; small stabilizers so a lucky-1-day spike can't win.
        return (p1 if np.isfinite(p1) else 0.0) + 0.15 * p3 + 0.02 * rk.get("pr_auc", 0.0)

    def cb(study, trial):
        state = trial.state.name
        p1 = trial.user_attrs.get("precision_at_1") or 0.0
        try:
            b = study.best_trial
            bp1 = b.user_attrs.get("precision_at_1") or 0.0
            best_txt = f"best prec@1={bp1:.1%} (trial {b.number})"
        except ValueError:
            best_txt = "no completed trials yet"
        val = f"{trial.value:.3f}" if trial.value is not None else state
        print(f"[opt] trial {trial.number}: value={val} (prec@1={p1:.1%}) | {best_txt}",
              flush=True)

    study.optimize(objective, timeout=int(timeout_hours * 3600), callbacks=[cb],
                   gc_after_trial=True)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
    }
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    print(f"[opt] done. {best['n_trials']} trials. best precision@1="
          f"{best['best_user_attrs'].get('precision_at_1')}. saved best_params.json")
    return best
