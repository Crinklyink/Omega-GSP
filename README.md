# gsp — ML next-day "pop" ranker

A leakage-safe machine-learning pipeline that scans a universe of stocks every day
and **ranks them by the probability that they pop ≥ 8% the next trading session**
(next-day intraday high ≥ today's close × 1.08).

It is built to be **honest**: every claim it makes about itself is measured
out-of-sample with walk-forward testing, and there is a null-test that proves the
framework doesn't cheat by peeking at the future.

---

## Read this first — what this is and isn't

**What it IS:** a probability machine. It learns patterns (volatility regime,
momentum, volume spikes, "this name has been popping a lot lately", market state)
that historically preceded big up-days, and ranks today's stocks by that
probability. Used on a broad universe it can put its top picks meaningfully above
the base rate — that is a real, measurable *edge*.

**What it is NOT:** a crystal ball that says "BUY XYZ, it will be +8% tomorrow."
On any given day only ~2–6% of liquid stocks pop ≥8%. A *good* model lifts its top
picks to maybe ~10–25% — a 2–5× edge over chance. That is genuinely valuable and
also nowhere near certainty. Most of your top picks will NOT pop. Position and risk
accordingly, or don't risk money at all.

**The edge decays.** Markets adapt. Any edge you find will weaken; it must be
re-trained continuously (that's what the walk-forward + `optimize` loop is for).

**This is not financial advice.** It's a research tool. Trading 8% movers is among
the hardest, most adversarial things in finance. You can lose everything.

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

### The leak we found (this is why both tests exist)

The first build accidentally fed `fwd_close_ret` (tomorrow's close return) into the
model as a feature. The label shuffle test **passed anyway** (AUC→0.50) and gave
false confidence — because shuffling the label breaks the leaked feature too. The
SHAP panel in the dashboard is what exposed it (`fwd_close_ret` topping the
attribution). Lesson: **a passing null-test is necessary, not sufficient.** Audit
your feature attributions.

### Honest out-of-sample results (S&P 500, 1.03M rows, after the fix)

```
                       LEAKED (wrong)     FIXED (true)
ROC-AUC                0.968              0.880
precision@5 / day      38.9%              11.3%   (base rate 1.0%)
lift vs base           39x                11.3x
best single pick/day   —                  16.9% hit  (17x base)
realistic strategy P&L Sharpe ~20 (!!)    -0.13%/trade, Sharpe -0.72  (LOSES money)
```

**Read this carefully — it's the whole point:**

- ✅ **The ranking skill is real.** Out-of-sample, the model's single best pick each
  day pops +8% about **17× more often than chance**. It genuinely finds names poised
  to move. That is a real, measured edge in *prediction*.
- ❌ **The naive strategy still loses money.** Buying that pick at the next open and
  exiting on a +8% limit / close earns **−0.13%/trade**. Why the contradiction?
  1. The move is largely in the **overnight gap you can't trade** — the +8%-from-
     prior-close is partly gone by the open (intraday-from-open hit rate is only ~8%).
  2. Violent names drop about as often as they pop; buy-at-open has no edge in
     *direction*, only in *volatility*.
  3. Costs (25 bps) finish it off.

**This is the real lesson of quant trading: a good predictor is not a profitable
strategy.** Turning 17× ranking lift into money needs a smarter entry/exit, risk
sizing, and almost certainly better data — and it may simply not be there with free
daily bars. Anyone showing you a clean equity curve here is hiding a leak.

**Also still inflating any positive result:** survivorship bias (the universe is
*today's* members, backtested from 2015 — losers/delistings missing). Fixing it
needs point-in-time, survivorship-bias-free data (paid). The only honest validation
is forward **paper-trading**.

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

```powershell
# from c:\Users\bentl\Desktop\gsp
.venv\Scripts\python.exe cli.py download --universe sp500     # ~1-3 min
.venv\Scripts\python.exe cli.py build --universe sp500        # assemble features+labels
.venv\Scripts\python.exe cli.py evaluate --universe sp500     # HONEST out-of-sample report
.venv\Scripts\python.exe scripts\leakage_test.py              # prove no leakage
.venv\Scripts\python.exe cli.py train --universe sp500        # fit final model
.venv\Scripts\python.exe cli.py scan  --universe sp500 --top 20   # today's candidates
```

Overnight edge-hunt (uses your CPU fully for the time budget):

```powershell
.venv\Scripts\python.exe cli.py optimize --hours 12 --universe sp500
.venv\Scripts\python.exe cli.py train --best --universe sp500     # train with found params
.venv\Scripts\python.exe cli.py evaluate --best --universe sp500  # re-check OOS
```

## Live dashboard + API keys

The HTML dashboard (`cli.py report`) is enriched with live data when API keys are
present. Put them in a **gitignored `.env`** at the project root:

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

## Scanning the *whole* market (recommended for actually finding 8% movers)

Big up-moves cluster in smaller, cheaper, more volatile names that the S&P 500
mostly excludes. To scan thousands of tickers, put one symbol per line in
`data/universe.txt`, then use `--universe file`:

```powershell
.venv\Scripts\python.exe scripts\fetch_universe.py     # writes a broad NASDAQ+NYSE list
.venv\Scripts\python.exe cli.py download --universe file
.venv\Scripts\python.exe cli.py build    --universe file
.venv\Scripts\python.exe cli.py evaluate --universe file
```

A broad universe gives more positives to learn from and a more realistic edge.

---

## How it works (pipeline)

```
universe.py  -> which tickers to scan
data.py      -> download & cache daily OHLCV (auto-adjusted) to data/raw/*.parquet
features.py  -> ~45 point-in-time technical features (momentum, vol, volume,
                gaps, RSI, distance-from-MAs, 52w context, "pop history", market regime)
labels.py    -> y = 1 if next-day High >= today's Close * 1.08
dataset.py   -> stack all tickers into one table + liquidity filter
model.py     -> LightGBM + embargoed walk-forward (the honest evaluator)
backtest.py  -> precision@K, lift, and a realistic buy-next-open +8%-limit P&L
optimize.py  -> Optuna search over walk-forward OOS lift (the 12-hour job)
scan.py      -> score the latest bar per ticker -> today's ranked candidates
```

## Tuning the target

Edit `gsp/config.py`:
- `TARGET_MOVE` — the pop size (0.08 = 8%). Smaller targets are easier/more common.
- `MIN_PRICE`, `MIN_DOLLAR_VOLUME` — tradeability filters.
- `TOP_K` — how many names you'd "buy" per day in the backtest.
- walk-forward windows, LightGBM defaults.

## Honest limitations / known gaps

- **Free data is imperfect.** yfinance has occasional bad bars/adjustment quirks;
  garbage prices → garbage features. A paid, point-in-time, survivorship-bias-free
  data source (e.g. Polygon, Norgate, Sharadar) is the biggest real upgrade.
- **Survivorship bias** in any "current tickers" universe: delisted/bankrupt names
  are missing, which flatters results. This is the #1 thing inflating backtests.
- **Fill assumptions** (buy at open, +8% limit fills) are optimistic for thin names.
- **No fundamentals / news / options flow / earnings dates** yet — all natural
  next features.
- Predicts *direction/magnitude opportunity*, not a full trade plan (no stops,
  sizing, or exits beyond the simulated rule).

## Next upgrades that actually move the needle (in order)
1. Broad, survivorship-bias-free universe + better data.
2. Add earnings-date / news / short-interest / options features.
3. Per-regime models (calm vs. volatile markets behave differently).
4. Probability calibration + position sizing (Kelly-capped) on top of the ranker.
5. Paper-trade live for 1–3 months before risking a cent.
