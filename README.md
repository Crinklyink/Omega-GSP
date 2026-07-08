# GSP - ML Intraday Pop Ranker
<p align="center">"The stock market is solvable through machine learning." -Crinklyink</p>

<p align="center">
  <img
    src="https://media1.tenor.com/m/L83Vfm7Rb3oAAAAd/%E0%B8%9E%E0%B8%B5%E0%B9%88%E0%B8%8B%E0%B8%B5%E0%B9%8A%E0%B8%94-opztv.gif"
    alt="Stonks meme with a rising chart"
    width="360"
  >
</p>

`gsp` is a leakage-aware research pipeline for ranking US stocks by the chance
that they climb at least 8% from the next session's open during that same day —
**without ever dipping more than 4% below that open first** (a "clean pop").

The default target is the safe tradeable version:

```text
y = 1 if High[t+1] >= Open[t+1] * 1.08  AND  Low[t+1] >= Open[t+1] * 0.96
```

That means the model scores stocks after today's close, assumes entry at
tomorrow's open, and asks whether a +8% limit order would fill intraday while a
stop-loss under the entry survives. Rewarding *any* +8% touch (the old target,
kept as `TARGET_MODE="high_vs_open"`) mathematically selects the most volatile
names in the market; dip-conditioning the label selects pops you can actually
sit through. The target is configured in [gsp/config.py](gsp/config.py).

This is a research tool, not financial advice. It finds names likely to move; it
does not yet provide a complete profitable trading system.

> **Live ledger:** the latest scan's dashboard is published at
> <https://crinklyink.github.io/Omega-GSP/> — `cli.py daily` and `cli.py report`
> push it automatically after every scan (opt out with `--no-publish`).
> Only the self-contained dashboard is published; data, models, and the paper
> ledger stay local. If `PAGES_PASSWORD` is set in the gitignored `.env`, the
> page is published AES-encrypted with a browser-side unlock — without the
> passphrase the content is ciphertext. (A short passphrase deters casual
> visitors, not determined ones.)
>
> Deep technical record — how the model was built, trained, validated, and
> what the experiments proved: [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md)

## Read This First

The model's ranking skill is real in the saved out-of-sample report, but the
naive strategy still loses money. That distinction is the most important thing
in the repo.

- Strong ranker: the best-scored names hit the clean +8% target far more often
  than the base rate (top pick: 13.2% of days at 19.8x the 0.67% base rate).
- Weak strategy: buying at the next open, taking +8% if hit, and otherwise
  exiting at the close still loses a little after costs (-0.38%/trade; the old
  any-touch target lost -0.97%) — and the 2026-07 experiments below show WHY,
  which is more useful than the fact itself.
- Practical next step: run the paper-trading ledger (`cli.py daily`) and let
  forward results, not backtests, have the final word.

## Latest Saved Out-of-Sample Report (2026-07-08 "clean pop" retarget)

Tuned 5-seed LightGBM ensemble, ~96 point-in-time features, 2012-2026 history,
4,165,623-row dataset / 3,618,434 out-of-sample rows across 46 embargoed
walk-forward folds. Universe: close ≥ $15, ADV ≥ $10M, 14d ATR ≤ 8% of price.
From `models/last_report.json`.

| Metric | Value |
| --- | ---: |
| Base rate for the clean +8% target | 0.67% |
| ROC-AUC | 0.877 |
| PR-AUC | 0.051 |
| Top-1 hit rate | 13.23% |
| Top-1 lift over base rate | 19.8x |
| Top-5 hit rate | 10.12% |
| Top-5 lift over base rate | 15.1x |
| Naive strategy avg trade return | -0.38% (win rate 46.4%) |
| Naive strategy cost assumption | 25 bps round trip |

Selectivity curve from the same report:

| Picks per day | Hit rate | Lift |
| ---: | ---: | ---: |
| 1 | 13.23% | 19.8x |
| 2 | 12.06% | 18.0x |
| 3 | 10.97% | 16.4x |
| 5 | 10.12% | 15.1x |
| 10 | 8.64% | 12.9x |

The edge is stable year by year (top-1 hit rate 7.3% in 2015 rising to 16-19%
in 2024-2026, with 12x-47x lift every single year; 2026 YTD avg trade is
~breakeven at +0.007%), and the leakage suite — label shuffles, single-feature
audit, and a truncation-invariance selftest — passes clean on the exact dataset
behind these numbers.

Hit rates look smaller than the previous any-touch target's (53% top-1) because
the event is 3x rarer and far harder: the pop must arrive without the dip. In
exchange, picks moved from micro-cap lottery tickets to liquid names (first
scan under the new target: DELL, MRNA, APP), and naive-strategy expectancy
improved by ~0.6%/trade.

## What The 2026-07 Experiments Established

Three experiments asked whether the strong ranking can be turned into a
profitable naive strategy. All three came back negative, and together they
pinpoint the real problem:

1. **Intraday replay** (`scripts/intraday_sim.py`, real hourly bars): 74% of
   +8% prints happen in the FIRST HOUR. With a 3% stop, ~79% of trades stop
   out, and in a fifth of trades the stop AND target are touched in the same
   opening bar. These names whip violently right at the open.
2. **Meta-labeling** (`scripts/meta_label.py`): a second model choosing
   take/skip on the top picks improves expectancy monotonically (skipped
   trades lose ~2x more than taken ones) but never crosses zero.
3. **EV ranking** (`cli.py evaluate --target ev`): a regression trained to
   maximize expected trade return, free to pick ANY stock, still lands at
   -0.25%/trade — it learns to avoid volatile names entirely and hide in the
   least-bad losers.

Conclusion: the trade construction itself (buy next open, +8% limit, exit at
close, 25 bps) has negative expectancy across the whole liquid universe. No
ranker fixes that; a different entry/exit structure might. That is the honest
frontier of this project.

The 2026-07-08 retarget acted on these findings from the label side: requiring
the pop to arrive without a >4% dip (plus the $10M ADV floor and 8% ATR
ceiling) moved naive expectancy from -0.97% to -0.38%/trade — still not
positive on the full 2015-2026 history, but ~breakeven in 2026, with picks the
stop-loss experiments above would no longer shake out. The paper ledger
(`cli.py daily`) is now the arbiter.

## Why The Backtest Is Less Likely To Be Fake

Most ML stock projects fail because the model accidentally sees the future.
This repo has several guardrails:

- Point-in-time features in [gsp/features.py](gsp/features.py): every feature for
  day `t` uses only OHLCV available at or before the close of day `t`.
- Forward columns are excluded by construction: `feature_columns()` rejects
  labels and anything starting with `fwd_`.
- Embargoed walk-forward validation in [gsp/model.py](gsp/model.py): train on a
  past window, skip an embargo gap, then test on a future window.
- Leak tests in [scripts/leakage_test.py](scripts/leakage_test.py): label shuffle
  checks and a single-feature AUC audit catch future-derived columns.

Known caveat: a current-membership universe is still survivorship biased. Use the
Polygon universe helper with delisted tickers when possible, and treat live
paper trading as the real validation.

## Setup

From the repo root:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generated data, models, reports, and secrets are ignored by git.

## Quickstart: S&P 500 Smoke Run

This is the fastest way to prove the pipeline works end to end.

```powershell
.venv\Scripts\python.exe cli.py download --universe sp500
.venv\Scripts\python.exe cli.py build --universe sp500
.venv\Scripts\python.exe cli.py evaluate --universe sp500
.venv\Scripts\python.exe scripts\leakage_test.py
.venv\Scripts\python.exe cli.py train --universe sp500
.venv\Scripts\python.exe cli.py report --universe sp500 --top 10
start reports\report.html
```

The report is a self-contained HTML dashboard. It does not require a server.

## Full-Market Run

Eight-percent intraday movers are much more common outside the S&P 500, so the
broad universe is the more interesting run.

Without API keys, fetch a broad current common-stock list:

```powershell
.venv\Scripts\python.exe scripts\fetch_universe.py
.venv\Scripts\python.exe cli.py download --universe file
.venv\Scripts\python.exe cli.py build --universe file
.venv\Scripts\python.exe cli.py evaluate --universe file
.venv\Scripts\python.exe scripts\leakage_test.py
.venv\Scripts\python.exe cli.py train --universe file
.venv\Scripts\python.exe cli.py report --universe file --top 10
start reports\report.html
```

With a Polygon-compatible key in `MASSIVE_API_KEY`, fetch a broader
survivorship-aware universe:

```powershell
.venv\Scripts\python.exe scripts\fetch_universe_polygon.py --delisted
.venv\Scripts\python.exe cli.py download --universe file
.venv\Scripts\python.exe cli.py build --universe file
.venv\Scripts\python.exe cli.py evaluate --universe file
```

Optional market-cap filter using Finnhub:

```powershell
.venv\Scripts\python.exe scripts\filter_universe_mcap.py --min-mcap 50
```

## Retraining On New Data (the routine)

Do this monthly or quarterly — more often buys little, since the walk-forward
validates the model on a quarterly retrain cadence anyway. Total ~1.5-2.5 h
on a modern 8-core box; steps must run in this order:

```powershell
# 1. Pull fresh bars (incremental — only fetches days you don't have)
.venv\Scripts\python.exe cli.py download --universe file

# 2. Rebuild the feature/label table so it includes the new bars
.venv\Scripts\python.exe cli.py build --universe file

# 3. Six-second invariant check (cheap insurance; MANDATORY after any code change)
.venv\Scripts\python.exe scripts\selftest.py

# 4. Retrain the 5-model ensemble with the saved tuned params
.venv\Scripts\python.exe cli.py train --best --universe file
```

That's it — the new model (`models/model_0..4.txt`) is live for scans
immediately. Two optional add-ons:

- **Fresh honest numbers + calibration** (adds ~1.5 h): rerun
  `cli.py evaluate --best --universe file` — this refreshes
  `models/last_report.json`, the score→hit-rate calibration behind `exp_hit`,
  and the dashboard's stats. Do it quarterly or after a market-character
  shift; skipping it just means scans keep using the previous calibration.
- **Leakage re-audit** (adds ~30 min): `scripts\leakage_test.py
  --sample-frac 0.25`. Only needed after changes to features/labels/data
  code — a routine retrain on unchanged code doesn't require it.
- **Full re-tune** (rare — new hyperparameter search, ~12 h+):
  `cli.py optimize --hours 12 --universe file --sample-frac 0.35
  --holdout-months 12`, then `train --best`. Resumable via `models/optuna.db`.
  After it finishes, `scripts\post_optimize.py` runs the whole
  rebuild→audit→train→evaluate→replay→meta→report chain in one command.

## Running Scans (the evening routine)

Run after the market close — the model decides on closing bars and its picks
are for the NEXT session. The scan automatically applies the same
tradeability filter the model was trained on (close ≥ $15, ADV ≥ $10M,
14d ATR ≤ 8% of price), so it never scores names outside its training
distribution.

```powershell
# Recommended one-liner: refresh bars -> settle pending paper trades ->
# scan -> log top-3 to the ledger -> print running forward stats
.venv\Scripts\python.exe cli.py daily --universe file --top 3

# Just the ranked list (no ledger logging)
.venv\Scripts\python.exe cli.py scan --universe file --top 20

# Regenerate the dashboard and open it
.venv\Scripts\python.exe cli.py report --universe file --top 10
start reports\report.html
```

Reading the output: `score` is the raw 0-1 signal; `exp_hit` is the honest
column — the fraction of out-of-sample names in that score band that actually
printed a clean +8% pop. `--min-exp-hit` enforces no-trade discipline: if
nothing clears the calibrated hit-rate bar, the scan declares a NO-TRADE DAY
and the ledger stays closed — that's a feature, not a failure. Calibrate the
threshold to the current target: under "clean_pop" the top pick averages a
13% hit rate, so `--min-exp-hit 0.1` keeps only above-average days (the old
0.4 gate belongs to the old any-touch label and would block everything). The
paper ledger (`data/paper_ledger.csv`) settles each pick against the next
session automatically on the following `daily` run.

## Hyperparameter Search

The optimizer runs full walk-forward trials and scores them by out-of-sample
precision. The default objective is precision at 1: the single best pick each
day.

```powershell
.venv\Scripts\python.exe -u cli.py optimize --hours 12 --universe file --sample-frac 0.35
.venv\Scripts\python.exe cli.py train --best --universe file
.venv\Scripts\python.exe cli.py evaluate --best --universe file
.venv\Scripts\python.exe cli.py report --universe file --top 10
```

Optimization is resumable through `models/optuna.db`. The winning parameters are
saved to `models/best_params.json`.

## Dashboard Enrichment And API Keys

The dashboard works without keys. API keys add company names, earnings dates,
news headlines, analyst recommendations, and market telemetry when available.

Create a gitignored `.env` file in the repo root:

```text
MASSIVE_API_KEY=...        # Polygon-compatible key used by the Polygon helper
FINNHUB_API_KEY=...        # earnings, profiles, recommendations, telemetry
ALPHAVANTAGE_API_KEY=...   # occasional quote/news checks
```

Probe what each key unlocks:

```powershell
.venv\Scripts\python.exe scripts\probe_keys.py
```

Security note: if a key was ever pasted into a shared transcript or ticket,
rotate it.

## Command Reference

| Command | Purpose |
| --- | --- |
| `cli.py download` | Download and cache OHLCV bars under `data/raw/`. |
| `cli.py build` | Assemble the feature/label matrix in `data/dataset/`. |
| `cli.py evaluate` | Embargoed walk-forward eval; saves report, OOS preds, calibration. `--target ev` for the expected-value experiment. |
| `cli.py optimize` | Optuna search (fold-pruned). `--holdout-months 12` keeps a vault for honest final validation. |
| `cli.py train` | Train the final seed-bagged ensemble; saves feature importances. |
| `cli.py scan` | Rank latest rows. `--min-exp-hit 0.4` = calibrated no-trade filter. |
| `cli.py paper` | Settle pending paper trades + log today's picks to the ledger. |
| `cli.py daily` | The evening routine: refresh bars -> settle -> scan -> log -> summary. |
| `cli.py report` | Generate the `reports/report.html` mission-control dashboard. |
| `cli.py all` | Run download, build, evaluate, and train. |
| `scripts/selftest.py` | 6-second invariant suite (truncation-invariance lookahead proof). |
| `scripts/post_optimize.py` | One command: rebuild -> audit -> train -> evaluate -> replay -> meta -> EV -> report. |
| `scripts/intraday_sim.py` | Replay OOS picks bar-by-bar against hourly data. |
| `scripts/meta_label.py` | Walk-forward take/skip second-stage model. |
| `scripts/opt_status.py` | Peek at the Optuna study. |
| `scripts/fetch_hourly.py` | Cache ~2 years of hourly bars (yfinance 60m limit). |

Common options:

```text
--universe sp500|file|fallback|default
--limit N
--best
--k N
--top N
```

## Pipeline Map

```text
gsp/universe.py  -> choose tickers
gsp/data.py      -> download and cache daily OHLCV
gsp/features.py  -> point-in-time technical and market-regime features
gsp/labels.py    -> next-session clean +8% intraday target
gsp/dataset.py   -> stack tickers into one training table
gsp/model.py     -> LightGBM and embargoed walk-forward evaluation
gsp/backtest.py  -> ranking metrics and naive strategy simulation
gsp/optimize.py  -> Optuna hyperparameter search
gsp/scan.py      -> score latest rows and rank candidates
gsp/report.py    -> self-contained HTML dashboard
gsp/enrich.py    -> optional live API enrichment
```

## Configuration

Edit [gsp/config.py](gsp/config.py) for experiment-level settings.

Important knobs:

- `TARGET_MOVE`: default `0.08` for an 8% target.
- `TARGET_MODE`: default `"clean_pop"` — the pop must arrive without dipping
  more than `MAX_DIP` below the entry open (`"high_vs_open"` = any touch).
- `MAX_DIP`: default `0.04` — the dip tolerance that defines a clean pop.
- `MIN_PRICE`: default `15.00` — excludes penny/low-priced names (< $15) from training.
- `MIN_DOLLAR_VOLUME`: default `10_000_000`.
- `MAX_ATR_PCT`: default `0.08` — volatility ceiling; wilder names are excluded
  from training and the scan.
- `TOP_K`: default number of names used in strategy simulation.
- `WALKFORWARD_TRAIN_YEARS`, `WALKFORWARD_TEST_MONTHS`, `EMBARGO_DAYS`: validation
  geometry.

After changing target settings, rebuild labels and rerun leakage tests:

```powershell
.venv\Scripts\python.exe cli.py build --universe file
.venv\Scripts\python.exe scripts\leakage_test.py
```

To see how hit rate changes as the target is raised:

```powershell
.venv\Scripts\python.exe scripts\target_curve.py
```

## Limitations

- Free price data can contain bad bars and adjustment quirks.
- Current-membership universes inflate backtests through survivorship bias.
- Limit-fill assumptions are optimistic for thin, fast-moving names.
- Catalyst data is displayed in the dashboard but is not yet modeled as features.
- The current strategy has no stop-loss, sizing, or no-trade filter.
- A high hit rate does not guarantee positive expected value.

## Next Work

Done in the 2026-07 rebuild: stop-loss/intraday simulation (hourly replay),
meta-label take/skip model, score calibration, no-trade filters, paper-trading
ledger, bad-bar sanitizer, invariant selftest, seed-bagged ensemble, calibrated
dashboard. What remains, in order of value:

1. Redesign the trade construction — the experiments show entry-at-open with a
   fixed +8% limit is the broken piece (74% of pops print in the first hour;
   opening ranges swallow tight stops). Candidates: participate in the opening
   hour directly, later entries, or scaled exits.
2. Kill survivorship bias: `scripts/fetch_universe_polygon.py --delisted` with
   a Polygon-compatible key, then rebuild and re-evaluate.
3. Promote catalyst data (earnings proximity, float, short interest) into
   features — needs a Finnhub key in `.env`.
4. Paper-trade with `cli.py daily` for 1-3 months; the Forward Ledger section
   of the dashboard is the scoreboard that decides everything.
