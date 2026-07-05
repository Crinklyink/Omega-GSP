"""One-shot post-optimization pipeline. Run this after `cli.py optimize` finishes
(or is stopped): it rebuilds the dataset (picking up the bad-bar sanitizer),
re-audits leakage, trains the final ensemble with the best params, produces the
full honest evaluation (hit + EV modes), replays trades against hourly bars,
trains the take/skip meta model, and writes the dashboard.

Steps run sequentially and log to post_optimize.log-style stdout; a failed step
aborts the rest (later steps depend on earlier artifacts).

Run:  python scripts/post_optimize.py [--skip-build] [--skip-leakage] [--skip-ev]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def run(step: str, args: list[str]) -> None:
    print(f"\n{'=' * 66}\n[post] STEP: {step}\n{'=' * 66}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(PY), "-u", *args], cwd=str(ROOT))
    mins = (time.time() - t0) / 60
    if r.returncode != 0:
        sys.exit(f"[post] step '{step}' FAILED (exit {r.returncode}) after {mins:.1f} min")
    print(f"[post] step '{step}' done in {mins:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-leakage", action="store_true")
    ap.add_argument("--skip-ev", action="store_true")
    ap.add_argument("--universe", default="file")
    a = ap.parse_args()

    run("invariant selftest (fails fast before burning hours)",
        ["scripts/selftest.py"])
    if not a.skip_build:
        run("rebuild dataset (bad-bar sanitizer applies here)",
            ["cli.py", "build", "--universe", a.universe])
    if not a.skip_leakage:
        run("leakage re-audit (labels changed with sanitized bars)",
            ["scripts/leakage_test.py", "--sample-frac", "0.25"])
    run("train final ensemble with best params",
        ["cli.py", "train", "--best", "--universe", a.universe])
    run("evaluate --best (hit mode; saves last_report.json + oos_preds.parquet)",
        ["cli.py", "evaluate", "--best", "--universe", a.universe])
    run("intraday replay of OOS picks (stop-vs-target, pop timing, exit grid)",
        ["scripts/intraday_sim.py", "--k", "5"])
    run("meta-label take/skip second stage",
        ["scripts/meta_label.py", "--k", "5"])
    if not a.skip_ev:
        run("evaluate --best --target ev (rank by expected trade return)",
            ["cli.py", "evaluate", "--best", "--target", "ev",
             "--universe", a.universe])
    run("dashboard", ["cli.py", "report", "--universe", a.universe, "--top", "10"])
    print("\n[post] ALL DONE. See models/last_report.json, models/last_report_ev.json,")
    print("       models/intraday_sim.json, models/meta_curve.json, reports/report.html")


if __name__ == "__main__":
    main()
