"""Live enrichment for the dashboard, using the Finnhub + Polygon keys.

Best-effort: every call is wrapped so a rate-limit or network hiccup degrades
gracefully (the field is just omitted) and never breaks the report. Finnhub free
allows ~60 calls/min, so we keep per-pick calls small and batch earnings.
"""
from __future__ import annotations
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta

from .secrets import KEYS

FINNHUB = "https://finnhub.io/api/v1"
POLYGON = "https://api.polygon.io"
_TIMEOUT = 12


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "gsp/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001  (best-effort)
        return None


def _fh(path: str, **params) -> object | None:
    tok = KEYS.get("FINNHUB_API_KEY")
    if not tok:
        return None
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return _get(f"{FINNHUB}/{path}?{q}&token={tok}")


def market_telemetry() -> list[dict]:
    """Live ETF proxies for the major indices + a vol gauge. Finnhub quote works
    on ETFs on the free tier."""
    proxies = [("S&P 500", "SPY"), ("Nasdaq", "QQQ"), ("Russell", "IWM"),
               ("Dow", "DIA"), ("Volatility", "VIXY"), ("Semis", "SMH")]
    out = []
    for label, sym in proxies:
        q = _fh("quote", symbol=sym)
        if isinstance(q, dict) and q.get("c"):
            out.append({"sym": label, "px": round(float(q["c"]), 2),
                        "chg": round(float(q.get("dp") or 0.0), 2)})
    return out


def _earnings_map(days_ahead: int = 75) -> dict[str, str]:
    """One batched call -> {ticker: next_earnings_date} for the coming window."""
    today = datetime.now().date()
    to = today + timedelta(days=days_ahead)
    data = _fh("calendar/earnings", **{"from": today.isoformat(), "to": to.isoformat()})
    m: dict[str, str] = {}
    if isinstance(data, dict):
        for e in data.get("earningsCalendar", []) or []:
            sym, dt = e.get("symbol"), e.get("date")
            if sym and dt and (sym not in m or dt < m[sym]):
                m[sym] = dt
    return m


def enrich_picks(tickers: list[str], with_news: bool = True) -> dict[str, dict]:
    """Return {ticker: {name, industry, earnings_date, days_to_earnings,
    headline, rec}} — all fields best-effort."""
    out: dict[str, dict] = {t: {} for t in tickers}
    earnings = _earnings_map()
    today = datetime.now().date()

    for t in tickers:
        info = out[t]
        prof = _fh("stock/profile2", symbol=t)
        if isinstance(prof, dict):
            info["name"] = prof.get("name")
            info["industry"] = prof.get("finnhubIndustry")

        ed = earnings.get(t)
        if ed:
            info["earnings_date"] = ed
            try:
                info["days_to_earnings"] = (datetime.fromisoformat(ed).date() - today).days
            except Exception:  # noqa: BLE001
                pass

        rec = _fh("stock/recommendation", symbol=t)
        if isinstance(rec, list) and rec:
            r = rec[0]
            info["rec"] = {"strongBuy": r.get("strongBuy", 0), "buy": r.get("buy", 0),
                           "hold": r.get("hold", 0), "sell": r.get("sell", 0),
                           "strongSell": r.get("strongSell", 0)}

        if with_news:
            frm = (today - timedelta(days=7)).isoformat()
            news = _fh("company-news", symbol=t, **{"from": frm, "to": today.isoformat()})
            if isinstance(news, list) and news:
                info["headline"] = news[0].get("headline")
                info["news_url"] = news[0].get("url")
    return out
