"""Produce TODAY's ranked candidates from a trained model.

For each ticker we rebuild the point-in-time features and score the most recent
row — i.e. the decision made at the latest available close. Output is a ranked
table: the names the model thinks are most likely to pop >= 8% the next session.
These are PROBABILITIES, not promises. Read the README before risking money.

The trained model is a seed-bagged ENSEMBLE (models/model_0.txt ...); scores are
the mean probability across members. Cross-sectional features are recomputed on
the live panel exactly as dataset.py computes them on the training panel.
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import MODEL_DIR, MIN_PRICE, MIN_DOLLAR_VOLUME, MAX_ATR_PCT
from .data import load_cached, load_market
from .features import make_features, market_features, add_cross_sectional

BOOSTER_PATH = MODEL_DIR / "model.txt"          # legacy single-model path
FEATCOLS_PATH = MODEL_DIR / "feature_columns.json"
ENSEMBLE_META_PATH = MODEL_DIR / "ensemble.json"


def save_model(models: list[lgb.Booster] | lgb.Booster, feat_cols: list[str]) -> None:
    if isinstance(models, lgb.Booster):
        models = [models]
    for old in MODEL_DIR.glob("model_*.txt"):
        old.unlink()
    for i, m in enumerate(models):
        m.save_model(str(MODEL_DIR / f"model_{i}.txt"), num_iteration=m.best_iteration)
    ENSEMBLE_META_PATH.write_text(json.dumps({"n_models": len(models)}, indent=2))
    FEATCOLS_PATH.write_text(json.dumps(feat_cols, indent=2))


def load_model() -> tuple[list[lgb.Booster], list[str]]:
    members = sorted(MODEL_DIR.glob("model_*.txt"),
                     key=lambda p: int(p.stem.split("_")[1]))
    if members:
        models = [lgb.Booster(model_file=str(p)) for p in members]
    elif BOOSTER_PATH.exists():  # legacy single-model fallback
        models = [lgb.Booster(model_file=str(BOOSTER_PATH))]
    else:
        raise FileNotFoundError("No trained model. Run `train` first.")
    feat_cols = json.loads(FEATCOLS_PATH.read_text())
    return models, feat_cols


def predict_scores(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X, num_iteration=m.best_iteration) for m in models]
    return np.mean(preds, axis=0)


def latest_row(ticker: str, mkt_feats) -> pd.DataFrame | None:
    raw = load_cached(ticker)
    if raw is None or len(raw) < 260:
        return None
    feats = make_features(raw, mkt_feats)
    if feats.empty:
        return None
    row = feats.iloc[[-1]].copy()
    row["ticker"] = ticker
    row["asof"] = feats.index[-1]
    row["close"] = raw["Close"].iloc[-1]
    return row


def build_live_panel(tickers: list[str], mkt_feats) -> pd.DataFrame:
    """Latest feature row per ticker, with cross-sectional features computed on
    this very panel (names with the same as-of date rank against each other)."""
    rows = [latest_row(t, mkt_feats) for t in tickers]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise RuntimeError("No scannable rows. Run `download` first.")
    df = pd.concat(rows, ignore_index=True)
    # Same tradeability filter as the TRAINING data (dataset.py). Without this
    # the model scores penny stocks it was never trained or validated on, and
    # the calibrated hit rates don't transfer. Applied BEFORE cross-sectional
    # features so live ranks see the same universe shape training saw.
    n0 = len(df)
    dollar_vol = np.expm1(df["log_dollar_vol_20"])
    df = df[(df["close"] >= MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOLUME)
            & (df["atr14_pct"] <= MAX_ATR_PCT)]
    if df.empty:
        raise RuntimeError("No candidates pass the price/liquidity/volatility filter.")
    print(f"[scan] {len(df)} of {n0} names pass the tradeability filter "
          f"(close >= ${MIN_PRICE:.0f}, ADV >= ${MIN_DOLLAR_VOLUME / 1e6:.0f}M, "
          f"ATR <= {MAX_ATR_PCT:.0%})")
    panel = df.rename(columns={"asof": "date"}).set_index(["date", "ticker"]).sort_index()
    panel = add_cross_sectional(panel)
    df = panel.reset_index().rename(columns={"date": "asof"})
    return df.replace([np.inf, -np.inf], np.nan)


def load_calibration() -> list[dict]:
    p = MODEL_DIR / "calibration.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return []


def scan(tickers: list[str], top: int = 20) -> pd.DataFrame:
    models, feat_cols = load_model()
    mkt = load_market()
    mkt_feats = market_features(mkt) if mkt is not None else None

    df = build_live_panel(tickers, mkt_feats)
    X = df.reindex(columns=feat_cols)
    df["score"] = predict_scores(models, X)

    # Calibrated read: what fraction of OOS names in this score band hit +8%.
    calib = load_calibration()
    if calib:
        from .backtest import expected_hit_rate
        df["exp_hit"] = df["score"].map(lambda s: expected_hit_rate(float(s), calib))

    # Interpretable context columns for the human reading the list.
    if "log_dollar_vol_20" in df.columns:
        df["adv_musd"] = np.expm1(df["log_dollar_vol_20"]) / 1e6
    out = df.sort_values("score", ascending=False)
    cols = [c for c in ("ticker", "asof", "close", "score", "exp_hit",
                        "pop_ho_60", "hi_open_avg_20", "adv_musd")
            if c in out.columns]
    return out[cols].head(top).reset_index(drop=True)
