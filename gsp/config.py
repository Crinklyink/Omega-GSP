"""Central configuration. Everything tunable lives here so experiments are reproducible."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # per-ticker cached OHLCV parquet
DATASET_DIR = DATA_DIR / "dataset"  # assembled feature matrices
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for _d in (RAW_DIR, DATASET_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- The prediction target -------------------------------------------------
# Decision is made at the CLOSE of day t using only information available then.
# Label y = 1 if the NEXT trading day's intraday HIGH reaches at least
#   Close_t * (1 + TARGET_MOVE).  i.e. "tomorrow this stock pops >= 8% above
#   where it closed today, at some point during the day."
TARGET_MOVE = 0.08

# Minimums to keep the universe tradeable and the labels meaningful.
MIN_PRICE = 1.50          # ignore sub-$1.50 names (data is garbage, hard to trade)
MIN_DOLLAR_VOLUME = 1_000_000   # 20d avg dollar volume floor

# ---- Data ------------------------------------------------------------------
HISTORY_START = "2015-01-01"   # how far back to pull
MARKET_INDEX = "SPY"           # used for market-regime / relative features

# ---- Walk-forward evaluation ----------------------------------------------
# Time-ordered. Train on the past, test on the future, then roll forward.
WALKFORWARD_TRAIN_YEARS = 3
WALKFORWARD_TEST_MONTHS = 3
EMBARGO_DAYS = 2   # gap between train end and test start to avoid label leakage

# ---- Model defaults (LightGBM) --------------------------------------------
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100

# How many names we "buy" per day when simulating the strategy.
TOP_K = 5

RANDOM_SEED = 42
