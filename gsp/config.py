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
# We then act at NEXT day's OPEN.
TARGET_MOVE = 0.08

# TARGET_MODE controls what "a win" means:
#   "high_vs_open"  -> y=1 if  High_{t+1} >= Open_{t+1} * (1 + TARGET_MOVE)
#       i.e. AFTER you buy at the open, the stock climbs >= 8% intraday that day.
#       This is the TRADEABLE goal: buy at open, place a +8% limit, it fills when
#       the stock rises 8% during the day. It explicitly EXCLUDES overnight gaps
#       (a stock that opened already up doesn't count unless it keeps climbing).
#   "high_vs_close" -> y=1 if  High_{t+1} >= Close_t * (1 + TARGET_MOVE)
#       includes the un-tradeable overnight gap. Kept only for comparison.
TARGET_MODE = "high_vs_open"

# Minimums to keep the universe tradeable and the labels meaningful.
MIN_PRICE = 15.00         # drop penny/low-priced names (< $15): noisy data, hard to trade
MIN_DOLLAR_VOLUME = 1_000_000   # 20d avg dollar volume floor

# ---- Data ------------------------------------------------------------------
HISTORY_START = "2012-01-01"   # how far back to pull (more history = more folds)
MARKET_INDEX = "SPY"           # used for market-regime / relative features

# ---- Walk-forward evaluation ----------------------------------------------
# Time-ordered. Train on the past, test on the future, then roll forward.
WALKFORWARD_TRAIN_YEARS = 3
WALKFORWARD_TEST_MONTHS = 3
EMBARGO_DAYS = 2   # gap between train end and test start to avoid label leakage

# ---- Model defaults (LightGBM) --------------------------------------------
LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",   # PR-AUC: the right early-stop metric for a ~4% positive class
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
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
NUM_BOOST_ROUND = 4000
EARLY_STOPPING = 200

# Seed-bagged ensemble size used by `evaluate` and `train` (optuna trials use 1
# model per trial so the 12h budget buys more trials; the params transfer).
ENSEMBLE_N = 5

# How many names we "buy" per day when simulating the strategy.
TOP_K = 5

RANDOM_SEED = 42
