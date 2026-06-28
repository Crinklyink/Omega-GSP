# GSP - ML Intraday Pop Ranker
<"The stock market is solvable through machine learning.">
<p align="center">
  <img
    src="https://media1.tenor.com/m/L83Vfm7Rb3oAAAAd/%E0%B8%9E%E0%B8%B5%E0%B9%88%E0%B8%8B%E0%B8%B5%E0%B9%8A%E0%B8%94-opztv.gif"
    alt="Stonks meme with a rising chart"
    width="360"
  >
</p>

`gsp` is a leakage-aware research pipeline for ranking US stocks by the chance
that they climb at least 8% from the next session's open during that same day.

The default target is the tradeable version:

```text
y = 1 if High[t+1] >= Open[t+1] * 1.08
```

That means the model scores stocks after today's close, assumes entry at
tomorrow's open, and asks whether a +8% limit order would fill intraday. The
target is configured in [gsp/config.py](gsp/config.py).

This is a research tool, not financial advice. It finds names likely to move; it
does not yet provide a complete profitable trading system.

## Read This First

The model's ranking skill is real in the saved out-of-sample report, but the
naive strategy still loses money. That distinction is the most important thing
in the repo.

- Strong ranker: the best-scored names hit the +8% intraday target far more
  often than the base rate.
- Weak strategy: buying at the next open, taking +8% if hit, and otherwise
  exiting at the close loses money after costs.
- Main gap: risk management. These are volatile stocks; misses can be large.
- Practical next step: stop-loss, sizing, and paper trading before risking
  capital.

## Latest Saved Out-of-Sample Report

These numbers come from the latest local `models/last_report.json`, a
broad-universe walk-forward run over 5,688,778 scored rows.

| Metric | Value |
| --- | ---: |
| Base rate for +8% intraday target | 4.20% |
| ROC-AUC | 0.868 |
| PR-AUC | 0.241 |
| Top-1 hit rate | 50.85% |
| Top-1 lift over base rate | 12.12x |
| Top-5 hit rate | 46.71% |
| Top-5 lift over base rate | 11.13x |
| Naive strategy avg trade return | -0.66% |
| Naive strategy cost assumption | 25 bps round trip |

Selectivity curve from the same report:

| Picks per day | Hit rate | Lift |
| ---: | ---: | ---: |
| 1 | 50.85% | 12.12x |
| 2 | 49.69% | 11.84x |
| 3 | 48.47% | 11.55x |
| 5 | 46.71% | 11.13x |
| 10 | 43.47% | 10.36x |

Interpretation: the ranker is useful at finding stocks that can move sharply.
It is not, by itself, a complete trading plan.

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
| `cli.py evaluate` | Run embargoed walk-forward evaluation and save `models/last_report.json`. |
| `cli.py optimize` | Run Optuna search over walk-forward precision. |
| `cli.py train` | Train a final model on all labeled data. |
| `cli.py scan` | Rank the latest row for each ticker in the selected universe. |
| `cli.py report` | Generate `reports/report.html`. |
| `cli.py all` | Run download, build, evaluate, and train. |

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
gsp/labels.py    -> next-session +8% intraday target
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
- `TARGET_MODE`: default `"high_vs_open"`, the tradeable intraday target.
- `MIN_PRICE`: default `1.50`.
- `MIN_DOLLAR_VOLUME`: default `1_000_000`.
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

1. Add stop-loss and intraday risk-management simulation.
2. Add a meta-label model for take/skip/size decisions on the highest-ranked names.
3. Calibrate probabilities and add capped position sizing.
4. Promote catalyst data into model features.
5. Validate with point-in-time survivorship-bias-free data.
6. Paper-trade live for at least 1 to 3 months before risking capital.
