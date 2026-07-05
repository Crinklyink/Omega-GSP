"""Publish the dashboard to GitHub Pages.

The Pages site serves `docs/index.html` from the `main` branch. Publishing =
copy the freshly generated `reports/report.html` there, commit, push. Wired
into `cli.py daily` and `cli.py report` (opt out with --no-publish), so the
public page always shows the latest scan.

Best-effort by design: a failed push (no network, auth expired) prints a
warning and never breaks the scan itself. The paper ledger CSV and all
data/models stay local — only the self-contained dashboard is published.

PASSWORD PROTECTION: if PAGES_PASSWORD is set (in the gitignored .env or the
environment), the published page is AES-256-GCM ENCRYPTED client-side-decrypt
style: docs/index.html becomes a small unlock page + ciphertext, and the real
dashboard only exists after the browser derives the key (PBKDF2, 600k iters)
from the password. Without the password the content is ciphertext, not merely
hidden. Honest caveat: a very short password can be brute-forced by anyone
determined — this deters casual visitors, it is not bank security.
"""
from __future__ import annotations
import base64
import os
import re
import shutil
import subprocess
from datetime import datetime

from .config import ROOT, REPORT_DIR

DOCS_INDEX = ROOT / "docs" / "index.html"

_PBKDF2_ITERS = 600_000

# Unlock page — Platinum Ledger styled, self-contained, WebCrypto decrypt.
_LOCK_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SP — The Ledger · Private</title>
<style>
  :root{--bg:#0b0c0e;--panel:#121417;--ink:#e8e6e1;--bright:#f5f3ee;--mut:#8b8f94;
        --pt:#9aa3ad;--pt2:#cfd6dd;--ox:#a8544c}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font-family:Georgia,'Times New Roman',serif}
  body{display:flex;align-items:center;justify-content:center;
    background-image:radial-gradient(900px 380px at 50% -8%,rgba(154,163,173,.06),transparent 62%)}
  .plate{background:linear-gradient(180deg,#121417,#0b0c0e 62%);border:1px solid rgba(154,163,173,.5);
    box-shadow:0 24px 70px rgba(0,0,0,.7);padding:44px 52px;text-align:center;position:relative;max-width:380px}
  .plate::before{content:"";position:absolute;inset:8px;border:1px solid rgba(154,163,173,.22);pointer-events:none}
  .seal{display:inline-flex;align-items:center;justify-content:center;width:56px;height:56px;
    border:1px solid var(--pt);border-radius:50%;margin-bottom:16px;
    font-weight:900;font-size:20px;color:var(--pt2);letter-spacing:.05em}
  h1{font-size:22px;font-weight:900;color:var(--bright);margin:0 0 6px}
  p{font-size:14px;font-style:italic;color:var(--mut);margin:0 0 22px}
  input{width:100%;box-sizing:border-box;background:#171a1e;border:1px solid rgba(154,163,173,.35);
    color:var(--bright);font:inherit;font-size:18px;text-align:center;letter-spacing:.3em;
    padding:10px 12px;outline:none}
  input:focus{border-color:var(--pt2)}
  button{margin-top:14px;width:100%;background:none;border:1px solid var(--pt);color:var(--pt2);
    font-family:inherit;font-size:11px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;
    padding:11px 0;cursor:pointer}
  button:hover{border-color:var(--pt2)}
  #err{color:var(--ox);font-size:13px;font-style:italic;margin-top:12px;min-height:18px}
</style></head>
<body><form class="plate" id="f">
  <div class="seal">SP</div>
  <h1>The Ledger is sealed.</h1>
  <p>Enter the passphrase to open the night desk.</p>
  <input id="pw" type="password" autocomplete="current-password" autofocus>
  <button type="submit">Unseal</button>
  <div id="err"></div>
</form>
<script>
const BLOB = "/*__BLOB__*/";
const raw = Uint8Array.from(atob(BLOB), c => c.charCodeAt(0));
const salt = raw.slice(0, 16), nonce = raw.slice(16, 28), ct = raw.slice(28);
async function unseal(pw) {
  const km = await crypto.subtle.importKey('raw', new TextEncoder().encode(pw),
    'PBKDF2', false, ['deriveKey']);
  const key = await crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 600000, hash: 'SHA-256' }, km,
    { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
  const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: nonce }, key, ct);
  sessionStorage.setItem('sp_pw', pw);
  document.open(); document.write(new TextDecoder().decode(pt)); document.close();
}
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const err = document.getElementById('err');
  err.textContent = '';
  try { await unseal(document.getElementById('pw').value); }
  catch (_) { err.textContent = 'The desk declines.'; }
});
// same session: unseal silently with the remembered passphrase
const saved = sessionStorage.getItem('sp_pw');
if (saved) unseal(saved).catch(() => sessionStorage.removeItem('sp_pw'));
</script></body></html>"""


def _pages_password() -> str | None:
    try:
        from .secrets import KEYS
        pw = KEYS.get("PAGES_PASSWORD")
    except Exception:  # noqa: BLE001
        pw = None
    return pw or os.environ.get("PAGES_PASSWORD") or None


def _lock_page(html: str, password: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt, nonce = os.urandom(16), os.urandom(12)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=_PBKDF2_ITERS)
    key = kdf.derive(password.encode("utf-8"))
    ct = AESGCM(key).encrypt(nonce, html.encode("utf-8"), None)
    blob = base64.b64encode(salt + nonce + ct).decode("ascii")
    return _LOCK_TEMPLATE.replace("/*__BLOB__*/", blob)


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
    pw = _pages_password()
    if pw:
        DOCS_INDEX.write_text(_lock_page(src.read_text(encoding="utf-8"), pw),
                              encoding="utf-8")
        print(f"[publish] dashboard ENCRYPTED (PAGES_PASSWORD) -> {DOCS_INDEX}")
    else:
        shutil.copyfile(src, DOCS_INDEX)
        print(f"[publish] dashboard (public, no PAGES_PASSWORD set) -> {DOCS_INDEX}")
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
