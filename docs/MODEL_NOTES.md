# GSP Model Notes — the complete record of the 2026-07 build

This is the technical memory of how the current model was made: what it
predicts, what it eats, how it was trained, how it was validated, what the
experiments proved, and how to care for it. The README is the user manual;
this is the lab notebook. Nothing here changes code — if these notes and the
code ever disagree, the code is the truth and this file needs updating.

Build dates: 2026-07-02 (night) → 2026-07-04 (morning), on a Ryzen 7 5800X3D
(16 threads, 32 GB RAM), Python 3.12, LightGBM 4.6.0, pandas 3.0.3.

---

## 1. What the model predicts

One number per stock per day, decided at the close of day `t`:

    P( High[t+1] >= Open[t+1] * 1.08 )

In words: *if you bought at tomorrow's open and placed a +8% limit sell, would
it fill during that same session?* This is the "tradeable" target — it
deliberately excludes the overnight gap (a stock that opens already +8% counts
only if it climbs another 8% from that open). Configured by `TARGET_MOVE=0.08`
and `TARGET_MODE="high_vs_open"` in `gsp/config.py`.

The model is a **ranker, not a strategy**. Section 7 records the proof.

## 2. Data

- **Source**: yfinance daily OHLCV, auto-adjusted for splits/dividends.
- **Universe**: NASDAQ Trader symbol directory (NASDAQ + NYSE/other common
  stock, ETFs/warrants/units/preferreds stripped) → 5,101 symbols fetched,
  5,086 with usable history. **Current-membership list → survivorship bias is
  the #1 known validity caveat** (delisted names since 2012 are invisible).
- **History**: 2012-01-01 → 2026-07-02 (`HISTORY_START` moved from 2015 to
  2012 during this build for ~3 more years of walk-forward folds).
- **Tradeability filter** (training AND live scan, identical): close ≥ $15 and
  20-day average dollar volume ≥ $1M, both point-in-time. On 2026-07-02 this
  passed 2,307 of 4,642 scannable names.
- **Bad-bar sanitizer** (`_sanitize_bars` in `gsp/data.py`): a High more than
  3× BOTH the candle body top and the prior close is treated as a bad print
  and clipped to the body top. Rationale: a fake high creates a fake +8%
  label. Clipping (not row-dropping) preserves day-adjacency so labels never
  skip a session. Applied at download AND on cache load.
- **Hourly tier**: 4,924 tickers × ~2 years of 60-minute bars (yfinance's
  730-day hard limit) in `data/raw_60m/` — used only for the intraday replay
  and meta-model features, never for the daily model itself.
- **Final dataset**: 5,972,331 rows × 96 features, 106,016 positives
  (**1.776% base rate**), float32, `data/dataset/dataset.parquet`.

## 3. Features (96 total, all point-in-time)

The iron rule: a feature for day `t` may use only information known at the
close of day `t`. Enforced three ways (section 6). Families:

| Family | Examples | Why |
| --- | --- | --- |
| Momentum | `ret_1..ret_120`, vol-scaled variants | classic trend/reversal |
| Trend context | `px_to_sma10/20/50/200`, `macd_hist` | where price sits in its trend |
| Volatility/range | `vol_5..60`, `atr14_pct`, `range_trend`, `nr7`, `bb_z_20`, `bb_squeeze_120` | vol clustering IS the signal for pops |
| **Label-aligned pop history** | `pop_ho_20/60/120` (rolling freq of High/Open−1 ≥ 8%), `hi_open_avg_10/20`, `hi_open_max_20` | the single best predictor: how often has THIS name already done exactly what the label asks |
| Intraday/overnight split | `gap_today`, `gap_vol_20`, `intraday_ret_avg_5/20` | gappers vs grinders behave differently |
| Candle anatomy | `upper_shadow`, `lower_shadow`, `body_pct` + rolling means | wicks measure intraday reach |
| Breakout proximity | `stoch_20`, `atr_dist_hi20`, `new_hi20_count_10`, `dist_52w_high/low` | pops cluster near highs |
| Volume/liquidity | `vol_ratio_20`, `vol_z_20`, `vol_mom`, `upvol_ratio_20`, `log_dollar_vol_20`, `amihud_20` | volume precedes price; thin names move more |
| Streaks/seasonality | `up_streak`, `down_streak`, `dow`, `month`, `dom` | weak but free |
| Market regime | `mkt_ret_*`, `mkt_above_sma50`, `mkt_vol_20/60`, `mkt_dist_hi20`, `rs_5/20`, `beta_60`, `corr_mkt_60` | SPY-derived context |
| **Cross-sectional (per-day)** | `cs_rank_*` (12 pct-ranks vs the whole universe THAT day), `cs_breadth_up`, `cs_median_vol20`, `cs_mean_pop60` | the daily top-K selection is cross-sectional, so features should be too |

What the final model actually leans on (share of total gain):
`range_avg_10` 43.4%, `atr14_pct` 21.8%, `hi_open_avg_20` 10.8%,
`range_today` 4.6%, `vol_ratio_20` 2.1% — i.e. ~80% of the model is "how wide
does this name swing lately," refined by its own pop history. Full table in
`models/feature_importance.json`.

## 4. Model

- **Architecture**: 5 × LightGBM binary classifiers ("seed-bagged ensemble"),
  identical hyperparameters, seeds `42 + 101*i` (varying seed, bagging seed,
  feature-fraction seed). Live score = mean probability across members.
  Averaging uncorrelated tree noise tightens the top of the ranking, which is
  where precision@1..5 lives.
- **Early stopping**: `average_precision` (PR-AUC) on a time-ordered 15%
  validation tail — never shuffled. With a ~2% positive class, PR-AUC tracks
  top-of-ranking quality; ROC-AUC (the old metric) barely moves when the top
  of the list improves.
- **Class weighting**: `scale_pos_weight = neg/pos × spw_mult`, spw_mult was
  searched (winner: 0.53 — the search preferred LESS positive-class pressure
  than the auto weight).
- **Tuned hyperparameters** (trial #168 of 298, `models/best_params.json`):
  `learning_rate 0.0231, num_leaves 112, max_depth 6, min_child_samples 264,
  subsample 0.756, colsample_bytree 0.646, reg_alpha 0.028, reg_lambda 2.64,
  min_split_gain 0.329, max_bin 127, spw_mult 0.532`. Shallow-but-wide, mildly
  regularized, coarse bins — trains ~3x faster than the defaults did.
- **Files**: `models/model_0.txt … model_4.txt` + `ensemble.json` +
  `feature_columns.json` (written 2026-07-03 21:33).

## 5. How it was trained (the actual timeline)

| When (EDT) | What | Duration |
| --- | --- | --- |
| Jul 2 night | universe fetch + 14y daily download (5,086 tickers, 476 MB) | ~1 h |
| Jul 3 early AM | dataset build (5.97M rows), leakage audit (0.35 sample) | ~1.5 h |
| Jul 3 ~04-08 | baseline evaluation, default params, 5-model ensemble, 46 folds | ~4 h |
| Jul 3 08:50–20:50 | **12-hour Optuna search**: 298 trials, 31 complete / 267 pruned, on a fixed 35% ticker sample (2.03M rows), single model per trial, MedianPruner on cumulative fold-level precision@1 | 12 h |
| Jul 3 20:50–23:15 | post-optimize chain: selftest (6 s) → rebuild with sanitizer (9 min) → leakage re-audit (26 min) → **final train `--best`** (11 min) → full hit-mode evaluation (80 min) → intraday replay (24 s) → meta-label (3 min) → EV evaluation (19 min) → dashboard (4 min) | ~2.5 h |

Search objective: out-of-sample `precision@1 + 0.15·precision@3 + 0.02·PR-AUC`
per full walk-forward — every trial judged only on data it never trained on.
The winner beat trial 0 (near-defaults) by ~1.4pp of precision@1: with strong
features, tuning polishes rather than rescues.

**Walk-forward geometry** (all evaluation): train 3 years → 2-day embargo →
test 3 months → roll forward 3 months. 46 folds, 2015-2026. Pooled test-fold
predictions = 5,150,780 fully out-of-sample rows.

## 6. Why the numbers are believable (leakage defenses)

1. **Construction**: features use only backward-looking pandas ops; labels
   live in `labels.py` and are the only code allowed to `shift(-1)`; anything
   `fwd_*` is rejected from the feature list by prefix, with an assert.
2. **Truncation-invariance selftest** (`scripts/selftest.py`, 6 s): recompute
   features on history cut off at day `t` — row `t` must be bit-identical to
   the full-history computation. A feature that peeks forward CANNOT pass.
   This is the strongest cheap lookahead test and runs before every big job.
3. **Label-shuffle audits** (`scripts/leakage_test.py`): global shuffle
   collapses to AUC ≈ 0.50 / lift ≈ 0.9x (no lookahead); within-day shuffle
   collapses to lift ≈ 0.95x (no per-name leak); single-feature AUC audit
   flags anything > 0.92 alone (top legit feature: 0.913, vol-clustering).
   Re-run on the final sanitized dataset: **all clean**.
4. **Embargoed walk-forward** everywhere; the optimizer's model selection is
   the one residual overfit channel (it saw all years) — future searches
   should use `optimize --holdout-months 12`, added for exactly this.

## 7. Results and what the experiments proved

**Ranking (the model's job — it is genuinely good at it):**

| Metric | Old model | Final 2026-07 model |
| --- | ---: | ---: |
| ROC-AUC | 0.868 | **0.920** |
| Top-1/day hit rate | 50.9% | **53.0%** |
| Top-1 lift | 12.1x | **27.0x** |
| Top-5 hit rate / lift | 46.7% / 11.1x | 43.4% / **22.1x** |

Stable in every year 2015→2026 (top-1 hit 29%→69%, lift 14x–65x every year).
Calibration ladder (`models/calibration.json`): scores in the p98 band hit
~24%, p99 ~30%, p99.5 ~37%, p99.9 ~50%.

**Strategy (the honest part — three experiments, one conclusion):**

- Naive rule (buy next open, +8% limit, close exit, 25 bps): **−0.97%/trade**
  (52% win rate — the average miss loses more than the average win gains).
- ADV-based slippage, risk-on gating, trailing score gates: all still negative.
- **Intraday replay** (1,765 trades vs real hourly bars): 74% of +8% prints
  happen in the FIRST HOUR; with a 3% stop ~79% of trades stop out; ~21%
  touch stop AND target inside the opening bar. Every stop/exit combination
  in the grid: negative.
- **Meta-labeling** (take/skip model on the picks): monotonic improvement
  (taken −0.57% vs skipped −1.15% at thr 0.6, win rate → 61%) but never
  positive.
- **EV ranking** (Huber regression on realized trade return, free to pick any
  stock): −0.25%/trade — it learns to hide in low-vol names and lose slowly.

**Conclusion: the trade construction has negative expectancy universe-wide;
no ranker fixes it.** The pops the model finds print at the open, and the
open is exactly where entry+stop mechanics get shredded. The next frontier is
a different entry/exit structure (opening-hour participation), not a better
model.

## 8. Live operation

- **Evening routine**: `cli.py daily --universe file [--min-exp-hit 0.4]` —
  incremental download → settle pending paper trades → scan (tradeability
  filter → cross-sectional features → ensemble mean → calibrated `exp_hit`)
  → log top-5 to `data/paper_ledger.csv` → print ledger stats.
- **Dashboard**: `cli.py report --universe file --top 10` →
  `reports/report.html` (self-contained; deep sections appear as their
  artifacts exist).
- **First real ledger entries**: decision 2026-07-02 → SDOT, FCEL, MAAS,
  ARQQ, MOVE (settle Monday 2026-07-06).
- **The ledger is the final judge.** Backtests said the naive trade loses;
  if the forward ledger agrees after 4-6 weeks, that's settled truth.

## 9. Care and feeding

- **Routine retrain** (monthly/quarterly): `download` → `build` → `train
  --best` ≈ 1.5–2.5 h. Add `evaluate --best` + `report` (~1.5 h more) when
  fresh calibration/honest numbers are wanted.
- **Re-tune** (rare): `optimize --hours N --sample-frac 0.35
  --holdout-months 12` — resumable via `models/optuna.db` (vaulted studies
  get a `_h12` suffix; never compare across vault settings).
- **After ANY change to features/labels/data code**: run
  `scripts/selftest.py` (6 s) and `scripts/leakage_test.py --sample-frac
  0.25` (~25 min) before believing anything.
- **One-shot full chain**: `scripts/post_optimize.py`.
- **Known footguns fixed during this build — do not reintroduce:**
  - The dashboard template must never declare `const top` (or `location`,
    `name`, …) at global script scope — `window.top` is non-configurable and
    the whole script dies with a silent SyntaxError.
  - The live scan must apply the training tradeability filter BEFORE
    cross-sectional features, or the model scores penny stocks it never saw
    and calibration lies.
  - `gsp/secrets.py` is gitignored — a fresh clone silently loses enrichment
    until it's recreated (it just reads `.env`).
  - LightGBM `max_bin` is a Dataset param, not a train param.

## 10. Artifact map

| File | What it is |
| --- | --- |
| `models/model_0..4.txt`, `ensemble.json`, `feature_columns.json` | the live ensemble |
| `models/best_params.json` | tuned hyperparameters + search stats |
| `models/last_report.json` / `last_report_ev.json` | full OOS reports (hit / EV mode) |
| `models/oos_preds.parquet` / `oos_preds_ev.parquet` | raw OOS predictions (5.15M rows) |
| `models/calibration.json` | score band → historical hit rate |
| `models/intraday_sim.json` | hourly replay: pop timing + exit grid |
| `models/meta_curve.json` | take/skip threshold curve |
| `models/feature_importance.json` | ensemble mean gain per feature |
| `models/optuna.db` | all 298 search trials, resumable |
| `data/dataset/dataset.parquet` | 5.97M-row training table |
| `data/raw/`, `data/raw_60m/` | daily (14y) and hourly (2y) bar caches |
| `data/paper_ledger.csv` | the forward test — the scoreboard |
| `reports/report.html` | the mission-control dashboard |
