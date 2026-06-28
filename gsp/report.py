"""Generate a self-contained HTML dashboard from the trained model.

Unlike the design mockup, this is REAL: every number comes from your model. It
runs the live scan, computes genuine per-pick SHAP attributions (LightGBM native),
pulls the honest out-of-sample stats from the last evaluation, and writes one
standalone .html file you can open in any browser (no server, no framework).

Run pre-market:  python cli.py report --universe sp500 --top 10
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORT_DIR, MODEL_DIR, TARGET_MOVE
from .data import load_market
from .features import market_features
from .scan import load_model, latest_row

# Map raw feature names -> human "why" phrases for the lead/attribution panel.
PHRASE = {
    "ret_1": "1-day momentum", "ret_2": "2-day momentum", "ret_3": "3-day momentum",
    "ret_5": "5-day momentum", "ret_10": "10-day momentum", "ret_20": "20-day momentum",
    "ret_60": "3-month trend", "ret_120": "6-month trend",
    "px_to_sma10": "above 10-day MA", "px_to_sma20": "above 20-day MA",
    "px_to_sma50": "above 50-day MA", "px_to_sma200": "above 200-day MA",
    "macd_hist": "MACD turn", "vol_5": "5-day volatility", "vol_10": "10-day volatility",
    "vol_20": "elevated volatility", "vol_60": "volatility regime",
    "atr14_pct": "wide ATR range", "range_today": "wide daily range",
    "range_avg_10": "expanding range", "close_in_range": "strong close",
    "gap_today": "gap up", "gap_avg_5": "recent gaps", "rsi_14": "RSI thrust",
    "stoch_20": "stochastic thrust", "dist_52w_high": "near 52-week high",
    "dist_52w_low": "off the lows", "vol_ratio_20": "relative-volume surge",
    "vol_z_20": "volume spike", "log_dollar_vol_20": "dollar-volume surge",
    "pops_20": "repeat popper (20d)", "pops_60": "repeat popper (60d)",
    "max_up_10": "recent pop", "max_up_20": "recent big up-day",
    "log_price": "low price level", "dow": "weekday effect", "month": "seasonality",
    "mkt_ret_1": "market tailwind", "mkt_ret_5": "market trend",
    "mkt_ret_20": "market regime", "mkt_above_sma50": "market uptrend",
    "mkt_vol_20": "market volatility", "rs_5": "relative strength (1w)",
    "rs_20": "relative strength (1m)",
}


def _build_scored(tickers: list[str], top: int):
    booster, feat_cols = load_model()
    mkt = load_market()
    mkt_feats = market_features(mkt) if mkt is not None else None

    rows = [latest_row(t, mkt_feats) for t in tickers]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise RuntimeError("No scannable rows. Run `download` first.")
    df = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)

    X = df.reindex(columns=feat_cols)
    df["score"] = booster.predict(X, num_iteration=booster.best_iteration)
    df["pct"] = df["score"].rank(pct=True) * 100.0

    contrib = booster.predict(X, num_iteration=booster.best_iteration, pred_contrib=True)
    contrib = np.asarray(contrib)[:, :-1]  # drop base-value column

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    order = df.index.to_list()
    top_df = df.head(top)

    picks = []
    for i, (_, r) in enumerate(zip(range(len(top_df)), top_df.itertuples())):
        ci = contrib[order[i]] if i < len(order) else contrib[i]
        top_feat_idx = int(np.argmax(ci))
        lead = PHRASE.get(feat_cols[top_feat_idx], feat_cols[top_feat_idx])
        picks.append({
            "rank": i + 1,
            "ticker": r.ticker,
            "score": round(float(r.score), 3),
            "pct": round(float(r.pct), 1),
            "close": round(float(r.close), 2),
            "lead": lead,
        })

    # SHAP detail for the #1 name.
    shap = []
    if len(top_df):
        c0 = contrib[order[0]]
        idx = np.argsort(-np.abs(c0))[:7]
        for j in idx:
            shap.append({
                "f": PHRASE.get(feat_cols[int(j)], feat_cols[int(j)]),
                "v": round(float(c0[int(j)]), 3),
            })

    asof = str(pd.to_datetime(df["asof"].iloc[0]).date()) if "asof" in df else ""
    return picks, shap, asof, len(df)


def _load_stats() -> dict:
    p = MODEL_DIR / "last_report.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def generate(tickers: list[str], top: int = 10,
             out_path: Path | None = None, enrich: bool = True) -> Path:
    picks, shap, asof, universe_n = _build_scored(tickers, top)
    stats = _load_stats()

    telemetry = []
    if enrich:
        try:
            from .enrich import enrich_picks, market_telemetry
            info = enrich_picks([p["ticker"] for p in picks])
            for p in picks:
                p.update({k: v for k, v in info.get(p["ticker"], {}).items()
                          if v is not None})
            telemetry = market_telemetry()
        except Exception as e:  # noqa: BLE001
            print(f"[report] enrichment skipped: {e}")
    pk = stats.get("precision_at_k", {})
    rk = stats.get("ranking", {})
    curve = stats.get("precision_curve", [])

    payload = {
        "picks": picks,
        "shap": shap,
        "telemetry": telemetry,
        "asof": asof,
        "universe_n": universe_n,
        "target_pct": int(TARGET_MOVE * 100),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stats": {
            "base_rate": rk.get("base_rate"),
            "auc": rk.get("roc_auc"),
            "precision_at_k": pk.get("precision_at_k"),
            "k": pk.get("k"),
            "lift": pk.get("lift"),
            "curve": curve,
        },
    }
    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(payload))
    out_path = out_path or (REPORT_DIR / "report.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Standalone HTML template. Data is injected as JSON at /*__DATA__*/ and
# rendered by the vanilla JS at the bottom. Aesthetic mirrors the "Night
# Edition" mockup but runs with zero dependencies beyond Google Fonts.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SP — The Ledger · Night Edition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cinzel+Decorative:wght@700;900&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0809;--panel:#110f13;--ink:#e7dcc6;--ink2:#d6cbb2;--bright:#f6efdd;
    --mut:#948b77;--mut2:#bdb091;--gold:#c9a44c;--gold2:#ecd28a;--ox:#b35349;--sage:#9aa97f;
  }
  html,body{margin:0;padding:0;background:var(--bg);font-family:'EB Garamond',Georgia,serif;color:var(--ink)}
  *{box-sizing:border-box}
  body{background-image:radial-gradient(1100px 460px at 50% -8%,rgba(201,164,76,.08),transparent 62%)}
  .wrap{position:relative;max-width:1140px;margin:36px auto 50px;
    background:linear-gradient(180deg,#14110d,#0d0b0e 62%);
    background-image:repeating-linear-gradient(0deg,rgba(201,164,76,.02) 0 1px,transparent 1px 4px);
    border:1px solid rgba(201,164,76,.5);
    box-shadow:0 34px 90px rgba(0,0,0,.72), 0 0 0 1px rgba(0,0,0,.5) inset;
    padding:36px 48px 46px}
  .wrap::before{content:"";position:absolute;inset:9px;border:1px solid rgba(201,164,76,.22);pointer-events:none}
  .gold{color:var(--gold)}
  .grule{height:2px;border:0;margin:0;background:linear-gradient(90deg,transparent,var(--gold) 16%,var(--gold2) 50%,var(--gold) 84%,transparent)}
  .hrule{height:1px;border:0;margin:0;background:linear-gradient(90deg,transparent,rgba(201,164,76,.5),transparent)}
  .double{border-top:1px solid var(--gold);border-bottom:1px solid var(--gold);height:4px}
  .orn{color:var(--gold);text-align:center;font-size:14px;letter-spacing:.5em;margin:14px 0 6px;opacity:.85}
  .masthead{font-family:'Cinzel Decorative',serif;font-weight:900;font-size:88px;letter-spacing:.06em;line-height:1;margin:0;text-align:center;
    background:linear-gradient(180deg,#f8e4ab,#d9b45e 46%,#a67d31);-webkit-background-clip:text;background-clip:text;color:transparent}
  .kicker{font-family:'Cinzel',serif;font-size:10.5px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;color:var(--mut)}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th{font-family:'Cinzel',serif;font-size:10px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);border-bottom:1px solid var(--gold);padding:9px 8px}
  td{padding:12px 8px;border-bottom:1px solid rgba(201,164,76,.15)}
  .tk{font-family:'Playfair Display',serif;font-weight:700;font-size:19px;color:var(--bright);letter-spacing:.01em}
  .lead{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:14px;color:var(--gold);margin-left:8px}
  .bar{flex:1;max-width:92px;height:4px;background:rgba(201,164,76,.16)}
  .barf{height:100%;background:linear-gradient(90deg,var(--gold),var(--gold2))}
  .stat{flex:1;text-align:center;padding:18px 8px;border-right:1px solid rgba(201,164,76,.22)}
  .statv{font-family:'Playfair Display',serif;font-weight:900;font-size:34px;color:var(--bright)}
  .statl{font-family:'Cinzel',serif;font-size:9px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);margin-top:9px}
  h2{font-family:'Playfair Display',serif;font-weight:900;font-size:46px;line-height:1.05;margin:20px 0 0;color:var(--bright)}
  h3{font-family:'Playfair Display',serif;font-weight:900;font-size:25px;margin:0 0 12px;color:var(--bright)}
  a{color:inherit;text-decoration:none;border-bottom:1px dotted rgba(201,164,76,.4)}
  @media(max-width:820px){.masthead{font-size:48px}.grid{grid-template-columns:1fr !important}.rail{border-left:none !important;padding-left:0 !important;border-top:1px solid var(--gold);padding-top:18px}.wrap{padding:22px}}
</style></head>
<body><div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:center" class="kicker">
    <span id="vol">Vol. III · No. 24</span>
    <span class="gold" style="letter-spacing:.3em">❦&nbsp; The Ledger · Night Edition &nbsp;❦</span>
    <span id="modeltag">stockpred</span>
  </div>
  <div class="double" style="margin:11px 0"></div>

  <header style="text-align:center;padding:14px 0 4px">
    <div style="display:flex;align-items:center;justify-content:center;gap:28px">
      <span style="flex:0 0 130px;height:1px;background:linear-gradient(90deg,transparent,var(--gold))"></span>
      <h1 class="masthead">SP</h1>
      <span style="flex:0 0 130px;height:1px;background:linear-gradient(90deg,var(--gold),transparent)"></span>
    </div>
    <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:16px;color:var(--mut2);margin-top:9px">
      The pre-market ledger — names ranked by signal for a +<span id="tgt"></span>% climb the coming session
    </div>
  </header>

  <div class="grule" style="margin-top:8px"></div>
  <div class="kicker" style="text-align:center;padding:10px 0" id="dateline"></div>
  <div class="hrule"></div>

  <h2 id="headline"></h2>
  <p style="font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:500;font-size:20px;line-height:1.42;color:var(--ink2);max-width:790px" id="subhead"></p>

  <div style="display:flex;border-top:1px solid var(--gold);border-bottom:1px solid var(--gold);margin-top:16px" id="statline"></div>

  <!-- TOP PICK hero (the one name you actually care about) -->
  <div id="hero" style="position:relative;margin-top:26px;border:1px solid var(--gold);
    background:radial-gradient(720px 220px at 18% -10%,rgba(201,164,76,.12),transparent 70%),linear-gradient(180deg,rgba(201,164,76,.05),transparent);
    box-shadow:0 0 0 1px rgba(201,164,76,.18) inset;padding:24px 28px"></div>

  <div class="orn">✦ &nbsp;❖&nbsp; ✦</div>

  <div class="grid" style="display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:34px;padding-top:8px">
    <section style="min-width:0">
      <h3>The Book</h3>
      <table><thead><tr>
        <th style="text-align:left;width:34px">#</th><th style="text-align:left">Ticker</th>
        <th style="text-align:right">Last</th><th style="text-align:right;width:150px">Signal</th>
        <th style="text-align:right;width:58px">Pctl</th>
      </tr></thead><tbody id="book"></tbody></table>
      <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:13.5px;color:var(--mut);margin-top:9px">
        Signal is a 0–1 ranking score (not a calibrated probability). The honest expectation
        for how often the top names truly climb is the hit-rate panel at right.
      </div>
    </section>

    <aside class="rail" style="min-width:0;border-left:1px solid var(--gold);padding-left:30px">
      <div style="font-family:'Cinzel',serif;font-size:12px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);border-bottom:1px solid rgba(201,164,76,.4);padding-bottom:8px;margin-bottom:14px" id="whytitle">Why № 1 Leads</div>
      <div id="shap"></div>
      <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:12.5px;color:var(--mut);margin-top:9px">Fig. I — SHAP attribution (log-odds). <span style="color:var(--ox)">Oxblood</span> detracts.</div>

      <div style="border-top:1px solid rgba(201,164,76,.3);padding-top:14px;margin-top:24px">
        <div style="font-family:'Cinzel',serif;font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin-bottom:12px">Honest Hit-Rate · out-of-sample</div>
        <div id="curve"></div>
        <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:12.5px;color:var(--mut);margin-top:8px">Fig. II — Share of top-N picks/day that truly climbed +<span id="tgt2"></span>%. Pickier raises the rate; nothing reaches 100%.</div>
      </div>
    </aside>
  </div>

  <div class="grule" style="margin-top:28px"></div>
  <div style="padding:14px 0">
    <div style="font-family:'Cinzel',serif;font-size:11px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);margin-bottom:13px">Markets at a Glance · live</div>
    <div id="telem" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))"></div>
  </div>
  <div class="grule"></div>

  <footer style="margin-top:24px;text-align:center">
    <div style="display:inline-flex;align-items:center;justify-content:center;width:54px;height:54px;border:1px solid var(--gold);border-radius:50%;
      box-shadow:0 0 0 4px rgba(201,164,76,.08), 0 0 0 1px rgba(201,164,76,.4) inset;margin-bottom:12px">
      <span style="font-family:'Cinzel Decorative',serif;font-weight:900;font-size:19px;color:var(--gold)">SP</span>
    </div>
    <div style="font-family:'Cormorant Garamond',serif;font-size:13.5px;color:var(--mut)">
      <span id="gen"></span> &nbsp;·&nbsp; <span id="uni"></span> names scanned &nbsp;·&nbsp; A US-equity research ledger
    </div>
    <div style="font-family:'Cinzel',serif;font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--ox);margin-top:10px">
      Research Tool — Not Financial Advice
    </div>
    <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:13px;color:var(--mut);margin-top:8px;max-width:780px;margin-left:auto;margin-right:auto">
      This ledger shows the model's <b style="color:var(--ink2)">ranking</b> — which names are likeliest to move. That ranking is
      real, but a naive buy-at-open strategy on these picks <b style="color:var(--ox)">lost money</b> in honest out-of-sample tests.
      Ranking skill ≠ profit. Paper-trade before risking a cent.
    </div>
  </footer>
</div>

<script id="data" type="application/json">/*__DATA__*/</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const pct = v => v==null ? '—' : (v*100).toFixed(1)+'%';
document.getElementById('tgt').textContent = D.target_pct;
document.getElementById('tgt2').textContent = D.target_pct;
document.getElementById('gen').textContent = 'Struck ' + D.generated;
document.getElementById('uni').textContent = D.universe_n;
document.getElementById('dateline').textContent = 'As of close ' + D.asof + '  ·  Long-Only  ·  Research Ledger, Not Advice';
const top = D.picks[0] || {ticker:'—'};
document.getElementById('headline').textContent =
  'The House Backs ' + D.picks.length + ' Names; ' + top.ticker + ' Leads The Ledger';
const s = D.stats || {};
document.getElementById('subhead').textContent =
  'Out-of-sample, the top-' + (s.k||5) + ' picks climbed +' + D.target_pct + '% on ' +
  pct(s.precision_at_k) + ' of days — ' + (s.lift? s.lift.toFixed(1):'?') +
  '× the ' + pct(s.base_rate) + ' base rate. A real edge, not a certainty.';
document.getElementById('whytitle').textContent = 'Why ' + top.ticker + ' Leads';

// TOP PICK hero — the single name the user cares about most.
const curveMap = {}; (s.curve||[]).forEach(c=>curveMap[c.k]=c);
const p1 = curveMap[1];
const hitTxt = p1 ? (p1.precision*100).toFixed(1)+'%' : '—';
const liftTxt = p1 ? p1.lift.toFixed(0)+'×' : '—';
const eb = (top.days_to_earnings!=null)
  ? '<span style="font-family:Cinzel,serif;font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:2px 9px;margin-left:10px;border:1px solid var(--gold);color:var(--gold2)">⚜ Earnings '+(top.days_to_earnings<=0?'today':'in '+top.days_to_earnings+'d')+'</span>' : '';
document.getElementById('hero').innerHTML =
  '<div style="font-family:Cinzel,serif;font-size:11px;font-weight:600;letter-spacing:.24em;text-transform:uppercase;color:var(--gold)">✦ Top Pick · Highest Conviction ✦</div>'+
  '<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin-top:10px">'+
  '<span style="font-family:Playfair Display,serif;font-weight:900;font-size:54px;color:var(--bright);line-height:1">'+top.ticker+'</span>'+
  '<span style="font-family:Cormorant Garamond,serif;font-size:19px;color:var(--ink2)">'+(top.name||'')+(top.industry?' · '+top.industry:'')+'</span>'+eb+'</div>'+
  (top.headline?'<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:15px;color:var(--mut2);margin-top:9px">“'+(top.news_url?'<a href="'+top.news_url+'" target="_blank">'+top.headline+'</a>':top.headline)+'”</div>':'')+
  '<div style="display:flex;flex-wrap:wrap;gap:34px;margin-top:18px">'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:31px;color:var(--bright)">'+top.score.toFixed(3)+'</div><div class="statl">Signal (0–1)</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:31px;color:var(--gold2)">'+hitTxt+'</div><div class="statl">Top-pick hit rate · OOS</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:31px;color:var(--bright)">'+liftTxt+'</div><div class="statl">vs random ('+pct(s.base_rate)+' base)</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:31px;color:var(--bright)">$'+top.close.toLocaleString(undefined,{minimumFractionDigits:2})+'</div><div class="statl">Last close</div></div>'+
  '</div>'+
  '<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:13px;color:var(--mut);margin-top:15px">Honest read: out-of-sample, the №1 name climbed +'+D.target_pct+'% about <b style="color:var(--ink2)">'+hitTxt+'</b> of days — '+liftTxt+' better than chance, but it MISSES most days. Size for that; never wager the estate on one signal.</div>';

// statline
const stat = [
  {v: D.universe_n, l:'Universe'},
  {v: D.picks.length, l:'Long Names'},
  {v: s.lift? s.lift.toFixed(1)+'×':'—', l:'Lift vs base', gold:true},
  {v: pct(s.precision_at_k), l:'Hit rate (top-'+(s.k||5)+')'},
  {v: s.auc? s.auc.toFixed(3):'—', l:'OOS ROC-AUC'},
];
document.getElementById('statline').innerHTML = stat.map((x,i)=>
  '<div class="stat" style="'+(i==stat.length-1?'border-right:none':'')+(x.gold?';background:rgba(201,164,76,.09)':'')+'">'+
  '<div class="statv" style="'+(x.gold?'color:var(--gold2)':'')+'">'+x.v+'</div><div class="statl">'+x.l+'</div></div>').join('');

// book
const earnBadge = p => {
  if (p.days_to_earnings==null) return '';
  const d = p.days_to_earnings;
  const hot = d>=0 && d<=10;
  return '<span style="font-family:Cinzel,serif;font-size:9px;letter-spacing:.08em;text-transform:uppercase;padding:1px 7px;margin-left:8px;border:1px solid '+(hot?'var(--gold)':'rgba(201,164,76,.35)')+';color:'+(hot?'var(--gold2)':'var(--mut)')+'">⚜ Earnings '+(d<=0?'today':'in '+d+'d')+'</span>';
};
const recTag = r => {
  if(!r) return '';
  const bull = (r.strongBuy||0)+(r.buy||0), bear=(r.sell||0)+(r.strongSell||0), tot=bull+bear+(r.hold||0)||1;
  const pc = Math.round(bull/tot*100);
  return '<span style="font-family:Cormorant Garamond,serif;font-size:12.5px;color:var(--mut);margin-left:8px">'+pc+'% buy</span>';
};
document.getElementById('book').innerHTML = D.picks.map(p=>{
  const name = p.name ? '<div style="font-family:Cormorant Garamond,serif;font-size:13.5px;color:var(--ink2)">'+p.name+(p.industry?' · '+p.industry:'')+earnBadge(p)+recTag(p.rec)+'</div>':'';
  const head = p.headline ? '<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:12.5px;color:var(--mut);margin-top:3px">“'+(p.news_url?'<a href="'+p.news_url+'" target="_blank">'+p.headline+'</a>':p.headline)+'”</div>':'';
  return '<tr><td style="font-family:Playfair Display,serif;font-weight:900;font-size:19px;vertical-align:top;padding-top:13px;'+
  (p.rank==1?'color:var(--gold)':'color:var(--mut2)')+'">'+p.rank+'</td>'+
  '<td><span class="tk">'+p.ticker+'</span><span class="lead">'+p.lead+'</span>'+name+head+'</td>'+
  '<td style="text-align:right;vertical-align:top;padding-top:14px;color:var(--ink)">'+p.close.toLocaleString(undefined,{minimumFractionDigits:2})+'</td>'+
  '<td style="vertical-align:top;padding-top:14px"><div style="display:flex;align-items:center;justify-content:flex-end;gap:9px">'+
  '<div class="bar"><div class="barf" style="width:'+Math.round(p.score*100)+'%"></div></div>'+
  '<span style="font-weight:700;min-width:46px;text-align:right;color:var(--bright)">'+p.score.toFixed(3)+'</span></div></td>'+
  '<td style="text-align:right;color:var(--mut2);vertical-align:top;padding-top:14px">'+p.pct.toFixed(0)+'</td></tr>';
}).join('');

// telemetry
const tel = D.telemetry || [];
document.getElementById('telem').innerHTML = tel.length ? tel.map(t=>{
  const up = t.chg>=0, c = up?'var(--sage)':'var(--ox)';
  return '<div style="padding:0 18px;border-left:1px solid rgba(201,164,76,.22)">'+
  '<div style="font-family:Cinzel,serif;font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut)">'+t.sym+'</div>'+
  '<div style="display:flex;align-items:baseline;gap:9px;margin-top:6px"><span style="font-size:17px;color:var(--ink)">'+t.px.toLocaleString()+'</span>'+
  '<span style="font-weight:700;font-size:13px;color:'+c+'">'+(up?'+':'')+t.chg.toFixed(2)+'%</span></div></div>';
}).join('') : '<div style="color:var(--mut);font-style:italic;padding-left:18px">telemetry unavailable (rate-limited)</div>';

// shap
const mx = Math.max(...D.shap.map(r=>Math.abs(r.v)), 1e-6);
document.getElementById('shap').innerHTML = D.shap.map(r=>{
  const c = r.v>=0 ? 'var(--gold2)' : 'var(--ox)';
  return '<div style="display:grid;grid-template-columns:96px 1fr 48px;align-items:center;gap:8px;padding:5px 0">'+
  '<span style="font-family:Cormorant Garamond,serif;font-size:13px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+r.f+'</span>'+
  '<div style="height:8px;background:rgba(201,164,76,.12)"><div style="height:100%;width:'+Math.round(Math.abs(r.v)/mx*100)+'%;background:'+c+'"></div></div>'+
  '<span style="font-weight:700;font-size:11.5px;text-align:right;color:'+c+'">'+(r.v>=0?'+':'')+r.v.toFixed(3)+'</span></div>';
}).join('');

// hit-rate curve
const cv = s.curve || [];
const cmax = Math.max(...cv.map(c=>c.precision), 1e-6);
document.getElementById('curve').innerHTML = cv.map(c=>
  '<div style="display:grid;grid-template-columns:64px 1fr 52px;align-items:center;gap:8px;padding:5px 0">'+
  '<span style="font-family:Cormorant Garamond,serif;font-size:13px;color:var(--ink2)">top-'+c.k+'/day</span>'+
  '<div style="height:8px;background:rgba(201,164,76,.12)"><div style="height:100%;width:'+Math.round(c.precision/cmax*100)+'%;background:linear-gradient(90deg,var(--gold),var(--gold2))"></div></div>'+
  '<span style="font-weight:700;font-size:11.5px;text-align:right;color:var(--bright)">'+(c.precision*100).toFixed(1)+'%</span></div>').join('');
</script>
</body></html>"""
