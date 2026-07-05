"""gsp command line. Run any step:

  python cli.py download   [--universe sp500|file|fallback] [--limit N]
  python cli.py build                       # assemble feature/label dataset
  python cli.py evaluate   [--best] [--k 5] # walk-forward out-of-sample report
  python cli.py optimize   [--hours 12] [--quick]
  python cli.py train      [--best]         # train final model on all data
  python cli.py scan       [--top 20] [--universe ...]
  python cli.py all                         # download -> build -> evaluate -> train

Typical first run on a laptop:
  python cli.py download --universe sp500 --limit 120
  python cli.py build
  python cli.py evaluate
  python cli.py train
  python cli.py scan
"""
from __future__ import annotations
import argparse
import json

from gsp.config import MODEL_DIR, TOP_K, ENSEMBLE_N
from gsp import universe as U


def _resolve_universe(name: str, limit: int | None) -> list[str]:
    if name == "sp500":
        tickers = U.load_sp500() or U.FALLBACK
    elif name == "file":
        tickers = U.load_from_file() or U.FALLBACK
    elif name == "fallback":
        tickers = U.FALLBACK
    else:
        tickers = U.default_universe()
    if limit:
        tickers = tickers[:limit]
    return tickers


def _load_best_params() -> dict | None:
    p = MODEL_DIR / "best_params.json"
    if p.exists():
        return json.loads(p.read_text()).get("best_params")
    return None


def cmd_download(a):
    from gsp.data import update_universe, load_market
    load_market()
    tickers = _resolve_universe(a.universe, a.limit)
    print(f"[download] {len(tickers)} tickers (universe={a.universe})")
    have = update_universe(tickers)
    print(f"[download] done: {len(have)} tickers have data.")


def cmd_build(a):
    from gsp.dataset import build_dataset
    tickers = _resolve_universe(a.universe, a.limit)
    df = build_dataset(tickers)
    print(f"[build] dataset rows={len(df):,}  positives={int(df['y'].sum()):,}  "
          f"base_rate={df['y'].mean():.3%}")


def cmd_evaluate(a):
    from gsp.dataset import load_dataset
    from gsp.model import walk_forward
    from gsp.backtest import full_report, print_report
    params = _load_best_params() if a.best else None
    df = load_dataset()
    preds = walk_forward(df, params=params, n_models=a.ensemble, target=a.target)
    rep = full_report(preds, k=a.k, cost_bps=a.cost_bps)
    print_report(rep)
    # EV-mode runs get their own files so they never clobber the hit-mode report.
    suffix = "" if a.target == "hit" else f"_{a.target}"
    (MODEL_DIR / f"last_report{suffix}.json").write_text(
        json.dumps(rep, indent=2, default=float))
    # Keep the raw OOS predictions: scripts/intraday_sim.py replays the top-K
    # trades bar-by-bar against hourly data; scripts/meta_label.py trains the
    # take/skip second stage on them.
    preds.reset_index().to_parquet(MODEL_DIR / f"oos_preds{suffix}.parquet")
    print(f"[evaluate] saved OOS predictions -> {MODEL_DIR / f'oos_preds{suffix}.parquet'}")
    if a.target == "hit":
        from gsp.backtest import calibration_table
        (MODEL_DIR / "calibration.json").write_text(
            json.dumps(calibration_table(preds), indent=2))
        print("[evaluate] saved score->hit-rate calibration -> models/calibration.json")


def cmd_optimize(a):
    from gsp.dataset import load_dataset
    from gsp.optimize import optimize
    df = load_dataset()
    optimize(df, timeout_hours=a.hours, k=a.k, sample_frac=a.sample_frac,
             holdout_months=a.holdout_months)


def cmd_train(a):
    from gsp.dataset import load_dataset
    from gsp.model import train_final
    from gsp.scan import save_model
    params = _load_best_params() if a.best else None
    df = load_dataset()
    models, feat_cols = train_final(df, params=params, n_models=a.ensemble)
    save_model(models, feat_cols)
    print(f"[train] saved {len(models)}-model ensemble with {len(feat_cols)} "
          f"features -> {MODEL_DIR}")
    # Transparency: what is the model actually leaning on?
    import numpy as np
    imp = np.mean([m.feature_importance(importance_type="gain") for m in models],
                  axis=0)
    order = np.argsort(-imp)
    fi = {feat_cols[int(i)]: float(imp[int(i)]) for i in order}
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(fi, indent=2))
    total = max(imp.sum(), 1e-9)
    print("[train] top features by gain:")
    for i in order[:12]:
        print(f"    {feat_cols[int(i)]:24s} {imp[int(i)] / total:6.1%}")


def cmd_scan(a):
    from gsp.scan import scan
    tickers = _resolve_universe(a.universe, a.limit)
    out = scan(tickers, top=a.top)
    if getattr(a, "min_exp_hit", 0) and "exp_hit" in out.columns:
        kept = out[out["exp_hit"] >= a.min_exp_hit]
        if kept.empty:
            print(f"\n[scan] NO-TRADE DAY: no name clears calibrated hit rate "
                  f">= {a.min_exp_hit:.0%}. Best today: "
                  f"{out['ticker'].iloc[0]} at {out['exp_hit'].iloc[0]:.1%}.")
            return out.iloc[0:0]
        out = kept
    print("\n=========  TODAY'S CANDIDATES (P(>= +8% next session))  =========")
    with __import__("pandas").option_context("display.float_format", lambda x: f"{x:,.4f}"):
        print(out.to_string(index=False))
    print("\nReminder: these are probabilities, not promises. See README.\n")
    return out


def cmd_paper(a):
    from gsp.paper import settle, log_picks, summary
    n = settle()
    print(f"[paper] settled {n} pending trades")
    if not a.summary_only:
        picks = cmd_scan(a)
        if picks is not None and len(picks):
            added = log_picks(picks)
            print(f"[paper] logged {added} new picks")
    summary(cost_bps=a.cost_bps)


def cmd_daily(a):
    """One command each evening: refresh bars, settle yesterday's paper trades,
    scan, log today's picks, print the ledger stats, regenerate the dashboard,
    and publish it to the GitHub Pages site (unless --no-publish)."""
    cmd_download(a)
    cmd_paper(a)
    from gsp.report import generate
    tickers = _resolve_universe(a.universe, a.limit)
    out = generate(tickers, top=10)
    print(f"[report] wrote dashboard -> {out}")
    if not getattr(a, "no_publish", False):
        from gsp.publish import publish_dashboard
        publish_dashboard()


def cmd_report(a):
    from gsp.report import generate
    tickers = _resolve_universe(a.universe, a.limit)
    out = generate(tickers, top=a.top)
    print(f"[report] wrote dashboard -> {out}")
    print(f"[report] open it in a browser:  start {out}")
    if not getattr(a, "no_publish", False):
        from gsp.publish import publish_dashboard
        publish_dashboard()


def cmd_all(a):
    cmd_download(a); cmd_build(a); cmd_evaluate(a); cmd_train(a)


def main():
    p = argparse.ArgumentParser(description="gsp — ML next-day pop ranker")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--universe", default="sp500",
                        choices=["sp500", "file", "fallback", "default"])
        sp.add_argument("--limit", type=int, default=None)

    sp = sub.add_parser("download"); common(sp); sp.set_defaults(func=cmd_download)
    sp = sub.add_parser("build"); common(sp); sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("evaluate"); common(sp)
    sp.add_argument("--best", action="store_true")
    sp.add_argument("--k", type=int, default=TOP_K)
    sp.add_argument("--cost-bps", dest="cost_bps", type=float, default=25.0)
    sp.add_argument("--ensemble", type=int, default=ENSEMBLE_N,
                    help="seed-bagged models per fold (1 = fast single model)")
    sp.add_argument("--target", choices=["hit", "ev"], default="hit",
                    help="hit = P(+8%% print); ev = expected trade return (Huber regression)")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("optimize"); common(sp)
    sp.add_argument("--hours", type=float, default=12.0)
    sp.add_argument("--k", type=int, default=1,
                    help="precision@k to optimize; default 1 (the single best pick)")
    sp.add_argument("--sample-frac", dest="sample_frac", type=float, default=1.0,
                    help="fraction of tickers per trial (e.g. 0.3) to fit more trials in 12h")
    sp.add_argument("--holdout-months", dest="holdout_months", type=int, default=0,
                    help="keep the last N months OUT of the search entirely (a vault "
                         "for honest final validation); recommended 12 for fresh studies")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("train"); common(sp)
    sp.add_argument("--best", action="store_true")
    sp.add_argument("--ensemble", type=int, default=ENSEMBLE_N)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("scan"); common(sp)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--min-exp-hit", dest="min_exp_hit", type=float, default=0.0,
                    help="only show names whose calibrated hit rate clears this (e.g. 0.4)")
    sp.set_defaults(func=cmd_scan)

    def paper_args(sp):
        sp.add_argument("--top", type=int, default=5,
                        help="picks logged to the ledger per day")
        sp.add_argument("--min-exp-hit", dest="min_exp_hit", type=float, default=0.0)
        sp.add_argument("--cost-bps", dest="cost_bps", type=float, default=25.0)
        sp.add_argument("--summary-only", dest="summary_only", action="store_true")

    sp = sub.add_parser("paper", help="settle pending paper trades + log today's picks")
    common(sp); paper_args(sp); sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("daily", help="download refresh + paper settle/log + dashboard + publish")
    common(sp); paper_args(sp)
    sp.add_argument("--no-publish", dest="no_publish", action="store_true",
                    help="skip pushing the dashboard to the GitHub Pages site")
    sp.set_defaults(func=cmd_daily)

    sp = sub.add_parser("report"); common(sp)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--no-publish", dest="no_publish", action="store_true",
                    help="skip pushing the dashboard to the GitHub Pages site")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("all"); common(sp)
    sp.add_argument("--best", action="store_true")
    sp.add_argument("--k", type=int, default=TOP_K)
    sp.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--ensemble", type=int, default=ENSEMBLE_N)
    sp.add_argument("--target", choices=["hit", "ev"], default="hit")
    sp.set_defaults(func=cmd_all)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
