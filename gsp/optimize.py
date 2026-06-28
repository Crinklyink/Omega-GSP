"""Hyperparameter search — this is what eats the 12-hour budget productively.

Each Optuna trial runs a full walk-forward and is scored by its OUT-OF-SAMPLE
precision@K lift (with a small PR-AUC tie-breaker). Because every trial is judged
on data it never trained on, the search optimizes for a *real* edge, not for
memorizing the past. Runs until `timeout_hours` elapses, then reports the best
config. Use `--quick` folds for laptops.
"""
from __future__ import annotations
import json
import warnings

import numpy as np
import optuna

from .config import MODEL_DIR, RANDOM_SEED, TOP_K
from .model import walk_forward
from .backtest import precision_at_k, ranking_metrics, precision_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


def optimize(dataset, timeout_hours: float = 12.0, k: int = 1,
             study_name: str = "gsp_p1", sample_frac: float = 1.0) -> dict:
    """Hunt hyperparameters that maximize OUT-OF-SAMPLE precision@`k`. Default k=1:
    we optimize for the single best pick each day being a real +8% mover — exactly
    the 'I only care about the top 1' objective. A small precision@3 + PR-AUC bonus
    stabilizes the noisy top-1 signal without changing what we're chasing.

    sample_frac<1 trains each trial on a fixed random SUBSET of tickers so the 12-hour
    budget buys many more trials on a big universe. The same subset is used for every
    trial (so scores are comparable); the winning params are later retrained on the
    FULL universe by `train --best`."""
    if sample_frac < 1.0:
        tickers = dataset.index.get_level_values("ticker").unique().to_numpy()
        rng = np.random.default_rng(RANDOM_SEED)
        keep = set(rng.choice(tickers, size=max(50, int(len(tickers) * sample_frac)),
                              replace=False).tolist())
        mask = dataset.index.get_level_values("ticker").isin(keep)
        dataset = dataset[mask]
        print(f"[opt] sampling {len(keep)} of {len(tickers)} tickers for the search "
              f"({len(dataset):,} rows)")

    storage = f"sqlite:///{MODEL_DIR / 'optuna.db'}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=15),
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        try:
            preds = walk_forward(dataset, params=params, verbose=False)
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
        b = study.best_trial
        p1 = trial.user_attrs.get("precision_at_1") or 0.0
        bp1 = b.user_attrs.get("precision_at_1") or 0.0
        print(f"[opt] trial {trial.number}: value={trial.value:.3f} "
              f"(prec@1={p1:.1%}) | best prec@1={bp1:.1%}")

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
