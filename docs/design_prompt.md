# Claude Design prompt — GSP "Platinum Ledger" luxury dashboard

Paste everything below the line into claude.ai/design.

---

Design a luxury single-page dashboard for **GSP**, a private machine-learning stock scanner. Every evening after market close it ranks ~2,300 liquid US stocks by the probability each one climbs +8% intraday during the next session, and the owner reads this page for five minutes before deciding anything. It should feel like a **private bank's night desk**: hushed, expensive, typographic — closer to a hand-set financial broadsheet or an architect's monograph than a Bloomberg terminal. One viewer, one page, no marketing.

## Aesthetic direction — "Platinum Ledger"

- **No gold. No warm metallics anywhere.** The luxury is cold and understated: graphite, platinum, bone.
- Palette: near-black graphite backgrounds (#0b0c0e, panels #121417), hairlines and accents in cool platinum/silver (#9aa3ad, bright #cfd6dd), bone/ivory text (#e8e6e1, muted #8b8f94). Semantic color only where money is at stake: dry sage (#9aa97f) for gains, muted oxblood (#a8544c) for losses/warnings. Accents are jewelry: hairline rules, fine borders, numerals — never large filled areas, never gradients that read as metallic shine.
- Typography does the luxury work: a restrained engraved-serif masthead, high-contrast serif display for numbers and headlines (Playfair Display class), refined text serif for body (EB Garamond / Cormorant class), tabular numerals everywhere data appears. Letter-spaced small caps in silver for labels. No sans-serif anywhere.
- Texture: engraved-paper restraint — hairline double rules, borders inset from panel edges, minimal geometric ornaments (thin diamonds, rules) as section separators. Generous whitespace; density lives in the tables, not the layout.
- Motion: restrained. Slow fades on load, nothing bouncy, nothing that glows.

## Page structure (single scroll, sticky nav)

1. **Sticky nav**: thin bar — wordmark "SP", section links (The Book, Tape Replay, Gatekeeper, Engine Room, Forward Ledger), and on the right three small regime chips: `Risk-On · SPY above SMA50` (sage), `SPY 20d +2.1%`, `Vol 14% ann.` (silver).
2. **Masthead**: "SP — The Ledger · Night Edition", dateline "AS OF CLOSE 2026-07-02 · LONG-ONLY · RESEARCH LEDGER, NOT ADVICE", headline "The House Backs 5 Names; SDOT Leads The Ledger."
3. **Stat line**: Universe scanned 2,307 · ROC-AUC 0.920 · Top-pick hit rate 53.0% · Lift 27× base · Base rate 1.97%.
4. **Hero — the top pick**: SDOT, $50.55, signal 0.963. Big calibrated number: "≈50% historical hit rate for signals in this band." Six-month price chart with SMA20/50 overlays (thin platinum line on graphite, elegant, no gridlines). Below it a **calibration ladder**: nine score bands (p0–50 → p99.9) with historical hit rates 0.2% → 24% → 30% → 37% → 50%, the active band edged in bright platinum with a small "▲ tonight" marker.
5. **The Book** — the ranked table (5–10 rows): rank, ticker + one-line "why" in italic serif (e.g. "wide ATR range · pops 42% of days · $133M/day"), 60-day sparkline (sage/oxblood by trend), last price, signal bar 0–1 in silver, calibrated hit %, percentile. Sortable columns. Sample rows: SDOT 0.963/50%, FCEL $28.11 0.951/37%, MAAS $15.03 0.943/37%, ARQQ $23.52 0.939/37%, MOVE $17.37 0.931/37%.
6. **Right rail** beside The Book: "Why SDOT leads" — seven SHAP attribution bars (platinum positive, oxblood negative); honest hit-rate curve (top-1 53.0%, top-2 49.0%, top-3 46.7%, top-5 43.4%, top-10 37.7%); year-by-year top-1 hit rate bars 2015→2026 (29% rising to 69%).
7. **The Tape Replay**: horizontal histogram "when +8% prints happen" — 74% in the 9:30 bar, 11% at 10:30, tapering to 1.5% at 15:30; beside it an exit-rule grid table (stop × exit vs avg return/win/target/stopped) where every value is honestly negative in oxblood.
8. **The Gatekeeper**: take/skip table — meta threshold 0.30→0.70, fraction taken 94%→11%, avg net of taken −0.90%→−0.60%, of skipped −1.94%→−1.01%; caption: "the filter separates better from worse, but nothing crosses zero — that is the finding."
9. **The Engine Room**: feature-importance bars (range_avg_10 43%, atr14_pct 22%, hi_open_avg_20 11%, …) and a short engraved-plaque style note: "298 trials · 12 hours · best configuration 45.6% top-pick precision on its search sample."
10. **The Forward Ledger**: paper-trading scoreboard — settled trades, hit rate, avg net/trade, win rate as four stat cells; a thin cumulative-return line chart; by-month mini table. Include an elegant **empty state**: "Five names rest on the ledger. The market opens Monday; the ledger settles Monday night."
11. **Footer**: a thin-ruled circular "SP" seal in platinum, and the honesty plaque: "This ledger shows the model's ranking — real, measured, out-of-sample. A naive strategy on these picks loses money after costs. Ranking skill ≠ profit. Research tool, not financial advice."

## Design principles

- **Honesty is the brand.** Negative numbers are shown beautifully, never hidden: oxblood, plainly labeled. The page must feel like it respects the reader too much to flatter them.
- Numbers are the heroes; charts are quiet and gridless; labels are small caps; the only chromatic color on the page is sage/oxblood where money is at stake — everything else is graphite, platinum, bone.
- Include a **no-trade day** variant of the hero: "The desk sits tonight — no name clears the bar," styled as calmly confident, not empty.
- Dark mode only. Desktop-first (~1140px content column), with a graceful single-column mobile collapse.

## Don'ts

No gold, brass, bronze, champagne, or any warm metallic. No neon, no glassmorphism, no rounded-pill buttons, no emerald-green "profit" dashboards, no rocket/graph-up iconography, no sans-serif, no cards floating on drop shadows, no metallic sheen gradients. This is a ledger, not a fintech app.
