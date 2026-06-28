"""Produce TODAY's ranked candidates from a trained model.

For each ticker we rebuild the point-in-time features and score the most recent
row — i.e. the decision made at the latest available close. Output is a ranked
table: the names the model thinks are most likely to pop >= 8% the next session.
These are PROBABILITIES, not promises. Read the README before risking money.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import MODEL_DIR
from .data import load_cached, load_market
from .features import make_features, market_features

BOOSTER_PATH = MODEL_DIR / "model.txt"
FEATCOLS_PATH = MODEL_DIR / "feature_columns.json"


def save_model(booster: lgb.Booster, feat_cols: list[str]) -> None:
    booster.save_model(str(BOOSTER_PATH), num_iteration=booster.best_iteration)
    FEATCOLS_PATH.write_text(json.dumps(feat_cols, indent=2))


def load_model() -> tuple[lgb.Booster, list[str]]:
    if not BOOSTER_PATH.exists():
        raise FileNotFoundError("No trained model. Run `train` first.")
    booster = lgb.Booster(model_file=str(BOOSTER_PATH))
    feat_cols = json.loads(FEATCOLS_PATH.read_text())
    return booster, feat_cols


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


def scan(tickers: list[str], top: int = 20) -> pd.DataFrame:
    booster, feat_cols = load_model()
    mkt = load_market()
    mkt_feats = market_features(mkt) if mkt is not None else None

    rows = []
    for t in tickers:
        r = latest_row(t, mkt_feats)
        if r is not None:
            rows.append(r)
    if not rows:
        raise RuntimeError("No scannable rows. Run `download` first.")

    df = pd.concat(rows, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    X = df.reindex(columns=feat_cols)
    df["score"] = booster.predict(X, num_iteration=booster.best_iteration)
    out = df.sort_values("score", ascending=False)
    cols = ["ticker", "asof", "close", "score"]
    return out[cols].head(top).reset_index(drop=True)
