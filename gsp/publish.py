"""Publish the dashboard to GitHub Pages.

The Pages site serves `docs/index.html` from the `main` branch. Publishing =
copy the freshly generated `reports/report.html` there, commit, push. Wired
into `cli.py daily` and `cli.py report` (opt out with --no-publish), so the
public page always shows the latest scan.

Best-effort by design: a failed push (no network, auth expired) prints a
warning and never breaks the scan itself. The paper ledger CSV and all
data/models stay local — only the self-contained dashboard is published.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from datetime import datetime

from .config import ROOT, REPORT_DIR

DOCS_INDEX = ROOT / "docs" / "index.html"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT),
                          capture_output=True, text=True)


def pages_url() -> str | None:
    """Derive the github.io URL from the origin remote."""
    r = _git("remote", "get-url", "origin")
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", r.stdout.strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    return f"https://{owner.lower()}.github.io/{repo}/"


def publish_dashboard(push: bool = True) -> bool:
    src = REPORT_DIR / "report.html"
    if not src.exists():
        print("[publish] no reports/report.html yet — run `report` first")
        return False
    DOCS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, DOCS_INDEX)
    print(f"[publish] dashboard -> {DOCS_INDEX}")
    if not push:
        return True

    _git("add", "docs/index.html")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    c = _git("commit", "-m", f"publish: ledger as of {stamp}")
    if c.returncode != 0 and "nothing to commit" in (c.stdout + c.stderr):
        print("[publish] dashboard unchanged — nothing to push")
        return True
    p = _git("push", "origin", "main")
    if p.returncode != 0:
        print(f"[publish] WARNING: push failed (scan results are still local):\n"
              f"          {(p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout).strip() else 'unknown error'}")
        return False
    url = pages_url()
    print(f"[publish] pushed. Live at {url}" if url else "[publish] pushed.")
    return True
