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

from gsp.config import MODEL_DIR, TOP_K
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
    preds = walk_forward(df, params=params)
    rep = full_report(preds, k=a.k, cost_bps=a.cost_bps)
    print_report(rep)
    (MODEL_DIR / "last_report.json").write_text(json.dumps(rep, indent=2, default=float))


def cmd_optimize(a):
    from gsp.dataset import load_dataset
    from gsp.optimize import optimize
    df = load_dataset()
    optimize(df, timeout_hours=a.hours, k=a.k, sample_frac=a.sample_frac)


def cmd_train(a):
    from gsp.dataset import load_dataset
    from gsp.model import train_final
    from gsp.scan import save_model
    params = _load_best_params() if a.best else None
    df = load_dataset()
    booster, feat_cols = train_final(df, params=params)
    save_model(booster, feat_cols)
    print(f"[train] saved model with {len(feat_cols)} features -> {MODEL_DIR}")


def cmd_scan(a):
    from gsp.scan import scan
    tickers = _resolve_universe(a.universe, a.limit)
    out = scan(tickers, top=a.top)
    print("\n=========  TODAY'S CANDIDATES (P(>= +8% next session))  =========")
    with __import__("pandas").option_context("display.float_format", lambda x: f"{x:,.4f}"):
        print(out.to_string(index=False))
    print("\nReminder: these are probabilities, not promises. See README.\n")


def cmd_report(a):
    from gsp.report import generate
    tickers = _resolve_universe(a.universe, a.limit)
    out = generate(tickers, top=a.top)
    print(f"[report] wrote dashboard -> {out}")
    print(f"[report] open it in a browser:  start {out}")


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
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("optimize"); common(sp)
    sp.add_argument("--hours", type=float, default=12.0)
    sp.add_argument("--k", type=int, default=1,
                    help="precision@k to optimize; default 1 (the single best pick)")
    sp.add_argument("--sample-frac", dest="sample_frac", type=float, default=1.0,
                    help="fraction of tickers per trial (e.g. 0.3) to fit more trials in 12h")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("train"); common(sp)
    sp.add_argument("--best", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("scan"); common(sp)
    sp.add_argument("--top", type=int, default=20)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("report"); common(sp)
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("all"); common(sp)
    sp.add_argument("--best", action="store_true")
    sp.add_argument("--k", type=int, default=TOP_K)
    sp.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    sp.add_argument("--top", type=int, default=20)
    sp.set_defaults(func=cmd_all)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
