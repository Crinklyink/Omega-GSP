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
<title>SP — Night Edition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=PT+Serif:ital,wght@0,400;0,700;1,400&family=Oswald:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;padding:0;background:#07090c;font-family:'PT Serif',Georgia,serif;color:#e9edf3}
  *{box-sizing:border-box}
  .wrap{max-width:1140px;margin:0 auto;background:#11151b;background-image:repeating-linear-gradient(0deg,rgba(214,228,246,.016) 0 1px,transparent 1px 3px);border:1px solid rgba(233,237,243,.22);box-shadow:0 22px 60px rgba(0,0,0,.6);padding:30px 40px 40px;margin-top:30px;margin-bottom:40px}
  .rule{border-top:3px double #e9edf3}
  .masthead{font-family:'Cinzel Decorative',serif;font-weight:900;font-size:84px;letter-spacing:.04em;color:#f6f8fb;line-height:1;margin:0;text-align:center}
  .kicker{font-family:'Oswald',sans-serif;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#8a93a3}
  .red{color:#d24b40}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th{font-family:'Oswald',sans-serif;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#c2cad6;border-bottom:3px double #e9edf3;padding:8px}
  td{padding:11px 8px;border-bottom:1px solid rgba(233,237,243,.2)}
  .tk{font-weight:700;font-size:18px;color:#f6f8fb}
  .lead{font-style:italic;font-size:12px;color:#8a93a3;margin-left:8px}
  .bar{flex:1;max-width:90px;height:5px;background:rgba(233,237,243,.12)}
  .barf{height:100%;background:#9fb0c4}
  .stat{flex:1;text-align:center;padding:18px 8px;border-right:1px solid rgba(233,237,243,.28)}
  .statv{font-family:'Playfair Display',serif;font-weight:900;font-size:36px;color:#f6f8fb}
  .statl{font-family:'Oswald',sans-serif;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#8a93a3;margin-top:8px}
  h2{font-family:'Playfair Display',serif;font-weight:900;font-size:48px;line-height:1.04;margin:18px 0 0;color:#f6f8fb}
  h3{font-family:'Playfair Display',serif;font-weight:900;font-size:26px;margin:0 0 12px;color:#f6f8fb}
  @media(max-width:820px){.masthead{font-size:46px}.grid{grid-template-columns:1fr !important}.rail{border-left:none !important;padding-left:0 !important;border-top:3px double #e9edf3;padding-top:18px}}
</style></head>
<body><div class="wrap">
  <div style="display:flex;justify-content:space-between" class="kicker">
    <span id="vol">Vol. III</span><span class="red">★ Night Edition ★</span><span id="modeltag">stockpred</span>
  </div>
  <div class="rule" style="margin:8px 0"></div>
  <h1 class="masthead">SP</h1>
  <div style="text-align:center;font-style:italic;font-size:14px;color:#8a93a3;margin-top:6px">
    Pre-market scan — names ranked by signal for a +<span id="tgt"></span>% pop the coming session
  </div>
  <div class="rule" style="margin-top:10px"></div>
  <div class="kicker" style="text-align:center;padding:9px 0;border-bottom:1px solid #e9edf3" id="dateline"></div>

  <h2 id="headline"></h2>
  <p style="font-style:italic;font-size:18px;color:#c2cad6;max-width:760px" id="subhead"></p>

  <div style="display:flex;border-bottom:3px double #e9edf3;margin-top:14px" id="statline"></div>

  <!-- TOP PICK hero (the one name you actually care about) -->
  <div id="hero" style="margin-top:24px;border:2px solid #d24b40;background:rgba(210,75,64,.06);padding:22px 26px"></div>

  <div class="grid" style="display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:30px;padding-top:24px">
    <section style="min-width:0">
      <h3>Today's Book</h3>
      <table><thead><tr>
        <th style="text-align:left;width:34px">#</th><th style="text-align:left">Ticker</th>
        <th style="text-align:right">Last</th><th style="text-align:right;width:150px">Signal</th>
        <th style="text-align:right;width:58px">Pctl</th>
      </tr></thead><tbody id="book"></tbody></table>
      <div style="font-style:italic;font-size:11.5px;color:#8a93a3;margin-top:8px">
        Signal is a 0–1 model ranking score (not a calibrated probability). The honest
        expectation for how often the top names actually pop is the hit-rate panel at right.
      </div>
    </section>

    <aside class="rail" style="min-width:0;border-left:1px solid #e9edf3;padding-left:28px">
      <div style="border-top:4px solid #e9edf3;padding-top:7px;font-family:'Oswald',sans-serif;font-size:13px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#f6f8fb;margin-bottom:14px" id="whytitle">Why #1 Leads</div>
      <div id="shap"></div>
      <div style="font-style:italic;font-size:11.5px;color:#8a93a3;margin-top:8px">Fig. 1 — SHAP attribution (log-odds). <span class="red">Red</span> detracts.</div>

      <div style="border-top:1px solid #e9edf3;padding-top:12px;margin-top:22px">
        <div style="font-family:'Oswald',sans-serif;font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#c2cad6;margin-bottom:10px">Honest Hit-Rate (out-of-sample)</div>
        <div id="curve"></div>
        <div style="font-style:italic;font-size:11.5px;color:#8a93a3;margin-top:6px">Fig. 2 — Fraction of top-N picks/day that actually popped +<span id="tgt2"></span>%. Being pickier raises the rate; nothing reaches 100%.</div>
      </div>
    </aside>
  </div>

  <div style="margin-top:26px;border-top:1px solid #e9edf3;border-bottom:1px solid #e9edf3;padding:13px 0">
    <div style="font-family:'Oswald',sans-serif;font-size:12px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#9fb0c4;margin-bottom:12px">Markets at a Glance · live</div>
    <div id="telem" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))"></div>
  </div>

  <footer style="margin-top:26px;text-align:center">
    <div class="rule" style="padding-top:14px;font-size:12.5px;color:#8a93a3">
      <span style="font-family:'Cinzel Decorative',serif;font-weight:900;color:#f6f8fb">SP</span>
      &nbsp;·&nbsp; <span id="gen"></span> &nbsp;·&nbsp; <span id="uni"></span> names scanned
    </div>
    <div style="font-family:'Oswald',sans-serif;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#d24b40;margin-top:8px">
      Research Tool — Not Financial Advice
    </div>
    <div style="font-style:italic;font-size:12px;color:#8a93a3;margin-top:8px;max-width:760px;margin-left:auto;margin-right:auto">
      This panel shows the model's <b>ranking</b> (which names are likeliest to move). That ranking is
      real — but a naive buy-at-open strategy on these picks <b>lost money</b> in honest out-of-sample
      tests (the +8% move is largely in the un-tradeable overnight gap). Ranking skill ≠ profit.
      Validate by paper-trading before risking a cent.
    </div>
  </footer>
</div>

<script id="data" type="application/json">/*__DATA__*/</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const pct = v => v==null ? '—' : (v*100).toFixed(1)+'%';
document.getElementById('tgt').textContent = D.target_pct;
document.getElementById('tgt2').textContent = D.target_pct;
document.getElementById('gen').textContent = 'Generated ' + D.generated;
document.getElementById('uni').textContent = D.universe_n;
document.getElementById('dateline').textContent = 'As of close ' + D.asof + '  ·  Long-Only  ·  Research Tool, Not Advice';
const top = D.picks[0] || {ticker:'—'};
document.getElementById('headline').textContent =
  'Engine Stakes ' + D.picks.length + ' Names; ' + top.ticker + ' Tops The Book';
const s = D.stats || {};
document.getElementById('subhead').textContent =
  'Out-of-sample, the top-' + (s.k||5) + ' picks popped +' + D.target_pct + '% on ' +
  pct(s.precision_at_k) + ' of days — ' + (s.lift? s.lift.toFixed(1):'?') +
  'x the ' + pct(s.base_rate) + ' base rate. Real edge, not certainty.';
document.getElementById('whytitle').textContent = 'Why ' + top.ticker + ' Leads';

// TOP PICK hero — the single name the user cares about most.
const curveMap = {}; (s.curve||[]).forEach(c=>curveMap[c.k]=c);
const p1 = curveMap[1];
const hitTxt = p1 ? (p1.precision*100).toFixed(1)+'%' : '—';
const liftTxt = p1 ? p1.lift.toFixed(0)+'x' : '—';
const eb = (top.days_to_earnings!=null)
  ? '<span style="font-family:Oswald,sans-serif;font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:2px 8px;margin-left:10px;border:1px solid #d24b40;color:#d24b40">⚡ Earnings '+(top.days_to_earnings<=0?'today':'in '+top.days_to_earnings+'d')+'</span>' : '';
document.getElementById('hero').innerHTML =
  '<div style="font-family:Oswald,sans-serif;font-size:11px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#d24b40">★ Top Pick — Highest Conviction ★</div>'+
  '<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin-top:8px">'+
  '<span style="font-family:Playfair Display,serif;font-weight:900;font-size:52px;color:#f6f8fb;line-height:1">'+top.ticker+'</span>'+
  '<span style="font-size:17px;color:#c2cad6">'+(top.name||'')+(top.industry?' · '+top.industry:'')+'</span>'+eb+'</div>'+
  (top.headline?'<div style="font-style:italic;font-size:14px;color:#9fb0c4;margin-top:8px">“'+(top.news_url?'<a href="'+top.news_url+'" target="_blank" style="color:#9fb0c4">'+top.headline+'</a>':top.headline)+'”</div>':'')+
  '<div style="display:flex;flex-wrap:wrap;gap:30px;margin-top:16px">'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:30px;color:#f6f8fb">'+top.score.toFixed(3)+'</div><div class="statl">Signal (0–1)</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:30px;color:#d24b40">'+hitTxt+'</div><div class="statl">Top-pick hit rate (OOS)</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:30px;color:#f6f8fb">'+liftTxt+'</div><div class="statl">vs random ('+pct(s.base_rate)+' base)</div></div>'+
  '<div><div style="font-family:Playfair Display,serif;font-weight:900;font-size:30px;color:#f6f8fb">$'+top.close.toLocaleString(undefined,{minimumFractionDigits:2})+'</div><div class="statl">Last close</div></div>'+
  '</div>'+
  '<div style="font-style:italic;font-size:12px;color:#8a93a3;margin-top:14px">Honest read: out-of-sample, the #1-ranked name popped +'+D.target_pct+'% about <b>'+hitTxt+'</b> of days — '+liftTxt+' better than chance, but it MISSES most days. Size for that, never bet the farm on one signal.</div>';

// statline
const stat = [
  {v: D.universe_n, l:'Universe'},
  {v: D.picks.length, l:'Long Names'},
  {v: s.lift? s.lift.toFixed(1)+'x':'—', l:'Lift vs base', red:true},
  {v: pct(s.precision_at_k), l:'Hit rate (top-'+(s.k||5)+')'},
  {v: s.auc? s.auc.toFixed(3):'—', l:'OOS ROC-AUC'},
];
document.getElementById('statline').innerHTML = stat.map((x,i)=>
  '<div class="stat" style="'+(i==stat.length-1?'border-right:none':'')+(x.red?';background:rgba(210,75,64,.10)':'')+'">'+
  '<div class="statv" style="'+(x.red?'color:#d24b40':'')+'">'+x.v+'</div><div class="statl">'+x.l+'</div></div>').join('');

// book
const earnBadge = p => {
  if (p.days_to_earnings==null) return '';
  const d = p.days_to_earnings;
  const hot = d>=0 && d<=10;
  return '<span style="font-family:Oswald,sans-serif;font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;padding:1px 6px;margin-left:8px;border:1px solid '+(hot?'#d24b40':'#5a6472')+';color:'+(hot?'#d24b40':'#9fb0c4')+'">⚡ Earnings '+(d<=0?'today':'in '+d+'d')+'</span>';
};
const recTag = r => {
  if(!r) return '';
  const bull = (r.strongBuy||0)+(r.buy||0), bear=(r.sell||0)+(r.strongSell||0), tot=bull+bear+(r.hold||0)||1;
  const pc = Math.round(bull/tot*100);
  return '<span style="font-size:11px;color:#8a93a3;margin-left:8px">'+pc+'% buy</span>';
};
document.getElementById('book').innerHTML = D.picks.map(p=>{
  const name = p.name ? '<div style="font-size:12px;color:#c2cad6">'+p.name+(p.industry?' · '+p.industry:'')+earnBadge(p)+recTag(p.rec)+'</div>':'';
  const head = p.headline ? '<div style="font-style:italic;font-size:11.5px;color:#8a93a3;margin-top:2px">“'+(p.news_url?'<a href="'+p.news_url+'" target="_blank" style="color:#8a93a3">'+p.headline+'</a>':p.headline)+'”</div>':'';
  return '<tr><td style="font-family:Playfair Display,serif;font-weight:900;font-size:18px;vertical-align:top;padding-top:13px;'+
  (p.rank==1?'color:#d24b40':'color:#9fb0c4')+'">'+p.rank+'</td>'+
  '<td><span class="tk">'+p.ticker+'</span><span class="lead">'+p.lead+'</span>'+name+head+'</td>'+
  '<td style="text-align:right;vertical-align:top;padding-top:13px">'+p.close.toLocaleString(undefined,{minimumFractionDigits:2})+'</td>'+
  '<td style="vertical-align:top;padding-top:13px"><div style="display:flex;align-items:center;justify-content:flex-end;gap:9px">'+
  '<div class="bar"><div class="barf" style="width:'+Math.round(p.score*100)+'%"></div></div>'+
  '<span style="font-weight:700;min-width:46px;text-align:right">'+p.score.toFixed(3)+'</span></div></td>'+
  '<td style="text-align:right;color:#9fb0c4;vertical-align:top;padding-top:13px">'+p.pct.toFixed(0)+'</td></tr>';
}).join('');

// telemetry
const tel = D.telemetry || [];
document.getElementById('telem').innerHTML = tel.length ? tel.map(t=>{
  const up = t.chg>=0, c = up?'#9fb0c4':'#d24b40';
  return '<div style="padding:0 18px;border-left:1px solid rgba(233,237,243,.22)">'+
  '<div style="font-family:Oswald,sans-serif;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#8a93a3">'+t.sym+'</div>'+
  '<div style="display:flex;align-items:baseline;gap:9px;margin-top:5px"><span style="font-size:16px;color:#e9edf3">'+t.px.toLocaleString()+'</span>'+
  '<span style="font-weight:700;font-size:13px;color:'+c+'">'+(up?'+':'')+t.chg.toFixed(2)+'%</span></div></div>';
}).join('') : '<div style="color:#8a93a3;font-style:italic;padding-left:18px">telemetry unavailable (rate-limited)</div>';

// shap
const mx = Math.max(...D.shap.map(r=>Math.abs(r.v)), 1e-6);
document.getElementById('shap').innerHTML = D.shap.map(r=>{
  const c = r.v>=0 ? '#e9edf3' : '#d24b40';
  return '<div style="display:grid;grid-template-columns:96px 1fr 48px;align-items:center;gap:8px;padding:5px 0">'+
  '<span style="font-size:12px;color:#c2cad6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+r.f+'</span>'+
  '<div style="height:9px;background:rgba(233,237,243,.1)"><div style="height:100%;width:'+Math.round(Math.abs(r.v)/mx*100)+'%;background:'+c+'"></div></div>'+
  '<span style="font-weight:700;font-size:11.5px;text-align:right;color:'+c+'">'+(r.v>=0?'+':'')+r.v.toFixed(3)+'</span></div>';
}).join('');

// hit-rate curve
const cv = s.curve || [];
const cmax = Math.max(...cv.map(c=>c.precision), 1e-6);
document.getElementById('curve').innerHTML = cv.map(c=>
  '<div style="display:grid;grid-template-columns:64px 1fr 52px;align-items:center;gap:8px;padding:4px 0">'+
  '<span style="font-size:12px;color:#c2cad6">top-'+c.k+'/day</span>'+
  '<div style="height:9px;background:rgba(233,237,243,.1)"><div style="height:100%;width:'+Math.round(c.precision/cmax*100)+'%;background:#9fb0c4"></div></div>'+
  '<span style="font-weight:700;font-size:11.5px;text-align:right">'+(c.precision*100).toFixed(1)+'%</span></div>').join('');
</script>
</body></html>"""
