# gsp — ML intraday "pop" ranker

A leakage-safe machine-learning pipeline that scans the whole US market every day and
**ranks stocks by how likely they are to climb ≥ 8% from the next open, intraday** —
the *tradeable* target: buy at tomorrow's open, set a +8% limit, and it fills when the
stock rises 8% during the day. (Set in `gsp/config.py`: `TARGET_MODE="high_vs_open"`.)

It is built to be **honest**: every claim is measured out-of-sample with walk-forward
testing, two leak tests guard against peeking at the future, and the dashboard tells
you plainly where the edge is real and where it isn't.

---

## Read this first — what this is and isn't

**What it IS — a strong predictor.** On the broad US universe (~5,000 names, 7M
rows), the single highest-ranked pick climbs +8% from the open on roughly **half of
all days** out-of-sample — about **15–17× the base rate**. That is a real, measured
edge at *finding names that move*.

**What it is NOT (yet) — a money-printer.** This is the honest core finding: a great
predictor is not automatically a profitable strategy. When the #1 pick *misses*, these
volatile names fall hard, and a naive "buy at open, exit at close" rule **loses money
at every target level** (we measured +5% → +20%; all negative). Moving ≠ moving *up
and holdable*. Closing that gap needs **risk management** (a stop-loss to cut losers),
which is the next build — see "Honest results" below.

**Aiming higher doesn't make stocks climb more.** The target % is the *definition of a
win*, not a dial. Raise it and the win just gets **rarer**: the #1 pick hits +8% ~50%
of days, +12% ~36%, +20% ~20%. Pick the level you want with the trade-off in view
(`python scripts/target_curve.py`).

**The edge decays.** Markets adapt; retrain continuously (the `optimize` loop).

**This is not financial advice.** Trading 8% movers is among the hardest, most
adversarial things in finance. You can lose everything. Paper-trade first.

---

## Why you should trust the numbers (and where to distrust them)

The single biggest reason ML stock projects "work" and then lose money is **data
leakage** — accidentally feeding the model information from the future. This project
has a war story about exactly that, told honestly below.

**Defenses:**
1. **Point-in-time features** (`gsp/features.py`): every feature for day *t* uses
   only OHLCV up to and including day *t*. No forward shifts. `feature_columns()`
   hard-excludes anything named `fwd_*` so a future column cannot silently leak in.
2. **Embargoed walk-forward** (`gsp/model.py`): train on a past window, skip a
   2-day embargo, then test on a *future* window the model never saw. Roll forward.
   All reported metrics are stitched from these out-of-sample test blocks.
3. **Two leak tests** (`scripts/leakage_test.py`):
   - *Label shuffle* — destroys the signal; pipeline must collapse to AUC≈0.50.
   - *Single-feature AUC audit* — catches a leaked **feature** (which label-shuffle
     CANNOT, because shuffling y also de-correlates the leaked feature from the
     shuffled label). Any feature scoring ~perfect alone is a target leak.

### Honest out-of-sample results (broad US universe, ~5,000 names, 7M rows)

Target = **+8% from the open, intraday** (the tradeable one). All walk-forward, OOS:

```
ranking skill         the #1 pick climbs +8% from the open on ~50% of days (15.7× base)
ROC-AUC               0.89
selectivity curve     top-1 50% · top-3 44% · top-5 40% · top-10 ...  (pickier = higher)
naive strategy P&L    -0.26%/trade  (buy open, +8% limit else exit close, 25 bps)  -> LOSES
```

**The target-level trade-off** (`scripts/target_curve.py`) — and every level loses
money *with the naive exit*, which is the point:

```
target from open | #1 hit rate | avg $/trade (untuned)
     +5%         |    64%      |   -0.39%
     +8%         |    50%      |   -0.26%
    +10%         |    42%      |   -0.32%
    +12%         |    36%      |   -0.25%   <- "profit peak", still negative
    +15%         |    29%      |   -0.36%
    +20%         |    20%      |   -0.43%
```

**Read this carefully — it's the whole point:**

- ✅ **The ranking skill is real and strong.** The #1 pick climbs +8% from the open
  ~50% of days, ~16× chance. The model genuinely finds names poised to move.
- ❌ **The naive strategy still loses**, at *every* target. When the pick misses, these
  volatile names fall hard, and holding to the close eats the winners. High hit-rate,
  negative P&L. **A good predictor is not a profitable strategy** — the real lesson.
- 🔧 **The fix is risk management, not a better target or longer search.** A stop-loss
  that cuts losers early (~−3%) is the next build: math intuition is 50%×(+8%) +
  50%×(−3%) − costs ≈ **+2%/trade**. It bolts onto the *same* trained model — no
  retrain. (Whether daily-bar data is enough to model stops without whipsaw is the
  open question; paper-trading is the only real test.)

### The leak we caught earlier (why we have two leak tests)

An earlier build accidentally fed `fwd_close_ret` (tomorrow's close) in as a feature.
The label-shuffle test **passed anyway** (shuffling y de-correlates the leaked feature
too) — false confidence. The dashboard's **SHAP panel** exposed it. Fix: `feature_columns()`
hard-excludes every `fwd_*` column, plus a single-feature-AUC audit. **A passing
null-test is necessary, not sufficient — audit your attributions.**

**Also still inflating any positive result:** survivorship bias (the universe is
*today's* members backtested from 2015 — delistings missing). Polygon's delisted-ticker
list (`fetch_universe_polygon.py --delisted`) pushes back on it; a full fix needs paid
point-in-time data. The only honest validation is forward **paper-trading**.

---

## Your hardware

Tuned for: **Ryzen 7 5800X3D (16 threads) + 32 GB RAM + Radeon RX 7900 XT**.

For this *tabular* problem, gradient-boosted trees (LightGBM) on your CPU are the
right tool — they beat deep nets here and train a fold in seconds. Your 5800X3D's
huge cache is ideal for it. The AMD GPU is **not** used: CUDA-only DL libraries
don't run on it on Windows, and you don't need them. The "12-hour budget" is spent
on `optimize` — hundreds of walk-forward hyperparameter trials hunting a real edge —
which is exactly where the value is.

---

## Quickstart

Fast path on the S&P 500 (smaller, quicker to prove the pipeline):

```powershell
# from c:\Users\bentl\Desktop\gsp
.venv\Scripts\python.exe cli.py download --universe sp500
.venv\Scripts\python.exe cli.py build    --universe sp500
.venv\Scripts\python.exe cli.py evaluate --universe sp500     # HONEST out-of-sample report
.venv\Scripts\python.exe scripts\leakage_test.py              # prove no leakage
.venv\Scripts\python.exe cli.py train    --universe sp500
.venv\Scripts\python.exe cli.py report   --universe sp500 --top 10   # luxury dashboard
start reports\report.html
```

Full US market (what you actually want — micro-caps are where 8% intraday moves live):

```powershell
.venv\Scripts\python.exe scripts\fetch_universe_polygon.py    # ~5k active US tickers
.venv\Scripts\python.exe cli.py download --universe file      # yfinance, ~30 min
.venv\Scripts\python.exe cli.py build    --universe file      # ~7M rows
.venv\Scripts\python.exe cli.py evaluate --universe file
.venv\Scripts\python.exe scripts\target_curve.py              # hit-rate vs target level
```

Overnight 12-hour edge-hunt (precision@1; ticker-subsampled so it fits many trials;
the winner retrains on the full universe; resumable via `models/optuna.db`):

```powershell
.venv\Scripts\python.exe -u cli.py optimize --hours 12 --universe file --sample-frac 0.35
.venv\Scripts\python.exe cli.py train --best --universe file
.venv\Scripts\python.exe cli.py evaluate --best --universe file
.venv\Scripts\python.exe cli.py report --universe file
```

## Live dashboard + API keys

`cli.py report` writes a **self-contained, luxury "Night Ledger" dashboard** —
black-and-gold, gold-foil masthead, a **TOP PICK hero card** with the #1 name's honest
OOS hit-rate, the ranked book, a SHAP "why it leads" panel, a hit-rate curve, and a
live "Markets at a Glance" strip. Open `reports/report.html` in any browser (no server).

It is **enriched with live data** when API keys are present — real company names,
**earnings dates**, news headlines, analyst recommendations, index telemetry. Put the
keys in a **gitignored `.env`** at the project root:

```
MASSIVE_API_KEY=...       # this is a Polygon.io key
ALPHAVANTAGE_API_KEY=...
FINNHUB_API_KEY=...
```

```powershell
.venv\Scripts\python.exe scripts\probe_keys.py     # see what each key unlocks
.venv\Scripts\python.exe cli.py report --universe sp500 --top 10
start reports\report.html
```

What each key gives you (verified by `probe_keys.py`):

| Key | Good for | Limits |
|-----|----------|--------|
| **Polygon** (`MASSIVE`) | survivorship-aware **universe** (active + **delisted** tickers), news, reference | ~2 yrs history, ~5 calls/min — too slow to power the long backtest |
| **Finnhub** | dashboard enrichment: **earnings dates**, news headlines, analyst recs, ETF telemetry | ~60 calls/min; daily candles are paid |
| **Alpha Vantage** | occasional quotes | **25 calls/day** — too limited for bulk |

So the keys power the **UI** and the **universe list**; the price history that feeds
the model still comes from yfinance (free, long history, no rate cap). To build a
broad, survivorship-aware universe from Polygon:

```powershell
.venv\Scripts\python.exe scripts\fetch_universe_polygon.py --delisted   # writes data\universe.txt
.venv\Scripts\python.exe cli.py download --universe file
.venv\Scripts\python.exe cli.py build --universe file
.venv\Scripts\python.exe cli.py evaluate --universe file
```

> Security: `.env` is gitignored. You pasted these keys in chat once — rotate them
> if that transcript could ever be seen by others.

## Scanning the *whole* market (>$50M cap)

Big up-moves cluster in smaller, more volatile names the S&P 500 excludes, so the
broad universe is the real target. `fetch_universe_polygon.py` (above) writes the full
active list to `data/universe.txt`. To enforce a market-cap floor (e.g. $50M) via
Finnhub — cached and resumable, ~1–2 h for the full list:

```powershell
.venv\Scripts\python.exe scripts\filter_universe_mcap.py --min-mcap 50
```

A broad universe gives more positives to learn from and a more realistic edge. The
liquidity filter (`MIN_PRICE`, `MIN_DOLLAR_VOLUME`) already removes untradeable junk,
so the mcap filter is a refinement, not a prerequisite.

---

## How it works (pipeline)

```
universe.py  -> which tickers to scan
data.py      -> download & cache daily OHLCV (auto-adjusted) to data/raw/*.parquet
features.py  -> ~44 point-in-time technical features (momentum, vol, volume,
                gaps, RSI, distance-from-MAs, 52w context, "pop history", market regime)
labels.py    -> y = 1 if next-day High >= next-day Open * 1.08  (tradeable, intraday)
dataset.py   -> stack all tickers into one table + liquidity filter
model.py     -> LightGBM + embargoed walk-forward (the honest evaluator)
backtest.py  -> precision@K, the target-level curve, and a buy-next-open +8%-limit P&L
optimize.py  -> Optuna search over walk-forward OOS precision@1 (the 12-hour job)
scan.py      -> score the latest bar per ticker -> today's ranked candidates
report.py    -> render the luxury HTML dashboard (+ enrich.py for live API data)
```

## Tuning the target

Edit `gsp/config.py`:
- `TARGET_MODE` — `"high_vs_open"` (tradeable: +8% from the open, intraday; default)
  or `"high_vs_close"` (includes the un-tradeable overnight gap).
- `TARGET_MOVE` — the move size (0.08 = 8%). Higher = bigger but **rarer** wins; see
  `scripts/target_curve.py` for the hit-rate trade-off before you change it.
- `MIN_PRICE`, `MIN_DOLLAR_VOLUME` — tradeability filters (≈ the >$50M-cap floor).
- `TOP_K` — how many names you'd "buy" per day in the backtest.

> After changing the target, recompute labels and re-run the leak test:
> `cli.py build …` then `scripts/leakage_test.py`.

## Honest limitations / known gaps

- **Free data is imperfect.** yfinance has occasional bad bars/adjustment quirks;
  garbage prices → garbage features. A paid, point-in-time, survivorship-bias-free
  data source (e.g. Polygon, Norgate, Sharadar) is the biggest real upgrade.
- **Survivorship bias** in any "current tickers" universe: delisted/bankrupt names
  are missing, which flatters results. This is the #1 thing inflating backtests.
- **Fill assumptions** (buy at open, +8% limit fills) are optimistic for thin names.
- **Catalysts are shown but not yet modeled**: earnings dates / news / analyst recs
  appear on the dashboard (via the APIs) but aren't features in the model yet.
- Predicts *opportunity*, not a full trade plan — **no stop-loss / sizing yet** (the
  reason the naive strategy loses; it's upgrade #1 below).

## Next upgrades that actually move the needle (in order)
1. **Stop-loss / risk management** — the one thing standing between the proven 50%
   hit-rate and a positive P&L. Cut losers ~−3% instead of riding to the close.
   Bolts onto the *same* model, no retrain. **This is the next build.**
2. **Meta-labeling** — a second model that decides *take it / skip it / size* on the
   #1 pick. Directly targets "the single best pick" and gives a no-trade-today signal.
3. Probability **calibration** + position sizing (Kelly-capped) on the ranker.
4. **Catalyst features** — earnings proximity, short interest, news/sentiment spikes
   (the API keys already reach these); per-regime models (calm vs. volatile markets).
5. Survivorship-bias-free, point-in-time **data** (paid) — the honest ceiling.
6. **Paper-trade live** 1–3 months before risking a cent. The only real validation.
