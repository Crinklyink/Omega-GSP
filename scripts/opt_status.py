"""Peek at the running (or finished) Optuna study without disturbing it.

Run:  python scripts/opt_status.py
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna

from gsp.config import MODEL_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = f"sqlite:///{MODEL_DIR / 'optuna.db'}"

for summary in optuna.study.get_all_study_summaries(storage):
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    trials = study.trials
    by_state: dict[str, int] = {}
    for t in trials:
        by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
    print(f"\n=== study '{summary.study_name}' — {len(trials)} trials {by_state} ===")

    done = [t for t in trials if t.state.name == "COMPLETE"]
    recent = [t for t in trials
              if t.datetime_complete
              and t.datetime_complete > datetime.now() - timedelta(hours=2)]
    if recent:
        print(f"  pace: {len(recent)} trials finished in the last 2h")
    if not done:
        print("  no completed trials yet")
        continue
    b = study.best_trial
    print(f"  best trial #{b.number}: value={b.value:.4f}")
    for k, v in b.user_attrs.items():
        print(f"    {k:22s} {v}")
    print("  best params:")
    for k, v in b.params.items():
        print(f"    {k:22s} {v}")
