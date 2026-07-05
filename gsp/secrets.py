"""API keys for optional dashboard enrichment. Reads the gitignored .env in the
repo root (KEY=value lines) plus the process environment. No keys = the
dashboard still works; enrichment fields are simply omitted.
"""
from __future__ import annotations
import os

from .config import ROOT

KEYS: dict[str, str] = {}

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if v.strip():
                KEYS[k.strip()] = v.strip()

for k in ("FINNHUB_API_KEY", "MASSIVE_API_KEY", "ALPHAVANTAGE_API_KEY"):
    if os.environ.get(k):
        KEYS.setdefault(k, os.environ[k])
