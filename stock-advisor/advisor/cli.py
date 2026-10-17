"""コマンド。

  python -m advisor screen                    今日買う候補を出す
  python -m advisor backtest --start 2015-01-01   ルールを過去で検証する
  python -m advisor paper init --cash 1000000  紙上の口座を作る
  python -m advisor paper rebalance           候補に合わせて紙上で売買する
  python -m advisor paper status              紙上の口座の状態

共通: --synthetic（乱数データで動かす。ネットなしの動作確認用）, --universe FILE, --top N
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .broker import PaperBroker
from .data import load_prices
from .portfolio import DEFAULT_PORTFOLIO_FILE, Portfolio, plan_orders
from .strategy import StrategyConfig, compute_features, select_at
from .universe import DEFAULT_BENCHMARK, load_universe

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _cfg(args) -> StrategyConfig:
    return StrategyConfig(top_n=args.top, use_regime=not args.no_regime)


def _load(args, start: str):
    names = load_universe(args.universe)
    tickers = list(names)
    prices = load_prices(
        tickers, start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed,
    )
    bench = None
    if not args.no_regime:
        b = load_prices([DEFAULT_BENCHMARK], start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed + 1)
        bench = b[DEFAULT_BENCHMARK] if DEFAULT_BENCHMARK in b else None
        if bench is None:
            logging.warning("ベンチマーク %s を取れないので、地合いの判定なしで進めます", DEFAULT_BENCHMARK)
    return names, prices, bench


def _last_prices(prices: pd.DataFrame) -> dict[str, float]:
    last = prices.ffill().iloc[-1]
    return {t: float(v) for t, v in last.items() if pd.notna(v)}


# ---------- screen ----------
def cmd_screen(args) -> int:
    cfg = _cfg(args)
    names, prices, bench = _load(args, start=str(date.today().year - 3) + "-01-01")
    feats = compute_features(prices, bench, cfg)
    asof = pd.Timestamp(args.asof) if args.asof else prices.index[-1]
    sel = select_at(feats, prices, asof, cfg)

    lines = [f"# 買う候補（{sel.asof.date()} 時点）", ""]
    lines.append(f"- ルール: 12-1モメンタム上位 {cfg.top_n} 銘柄、200日線の上、年率変動率 {cfg.max_vol:.0%} 以下"
                 + ("、日経平均が200日線の上のときだけ" if cfg.use_regime and bench is not None else ""))
    lines.append(f"- 候補の母数: {len(prices.columns)} 銘柄、条件を満たしたのは {sel.eligible} 銘柄")
    lines.append("")
    if not sel.invested:
        lines.append(f"**今回は買わない。** {sel.reason}")
    else:
        lines.append("| 順位 | 銘柄 | 社名 | 終値 | 12-1モメンタム | 年率変動率 |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for i, p in enumerate(sel.picks, 1):
            lines.append(f"| {i} | {p.ticker} | {names.get(p.ticker, '')} | {p.price:,.0f} | {p.momentum:+.1%} | {p.vol:.0%} |")
    lines.append("")
    lines.append("※ 投資判断の参考情報で、利益を保証するものではない。買うかどうかは自分で決める。")
    text = "\n".join(lines)
    print(text)
    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"{sel.asof.date()}.md"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\n保存: {out}", file=sys.stderr)
    return 0


# ---------- backtest ----------
def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    names, prices, bench = _load(args, start=args.start)
    res = run_backtest(prices, bench, cfg, start=None, end=args.end, cost=args.cost, freq=args.freq)
    m = res.metrics
    print(f"# バックテスト {res.equity.index[0].date()} 〜 {res.equity.index[-1].date()}（{m['years']:.1f}年）")
    print(f"- 銘柄数 {len(prices.columns)}、上位 {cfg.top_n}、{'月' if args.freq == 'M' else '週'}1回入れ替え、片道コスト {args.cost:.2%}")
    print()
    print("| | ルール | ベンチマーク(買い持ち) |")
    print("|---|---:|---:|")
    rows = [("最終倍率", "final_multiple", "{:.2f}倍"), ("年率リターン", "cagr", "{:+.1%}"),
            ("年率変動率", "annual_vol", "{:.1%}"), ("シャープ", "sharpe", "{:.2f}"), ("最大下落", "max_drawdown", "{:.1%}")]
    for label, key, fmt in rows:
        b = fmt.format(res.bench_metrics[key]) if res.bench_metrics and key in res.bench_metrics else "-"
        print(f"| {label} | {fmt.format(m[key])} | {b} |")
    print(f"\n入れ替え {int(m['rebalances'])} 回、売買 {int(m['trades'])} 件")
    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / "backtest-equity.csv"
        df = pd.DataFrame({"strategy": res.equity})
        if res.bench_equity is not None:
            df["benchmark"] = res.bench_equity
        df.to_csv(out)
        print(f"保存: {out}", file=sys.stderr)
    return 0


# ---------- paper ----------
def cmd_paper(args) -> int:
    path = Path(args.file) if args.file else DEFAULT_PORTFOLIO_FILE
    if args.paper_cmd == "init":
        if path.exists() and not args.force:
            print(f"すでにあります: {path}（作り直すなら --force）", file=sys.stderr)
            return 1
        Portfolio(cash=float(args.cash)).save(path)
        print(f"紙上の口座を作りました: 現金 {args.cash:,.0f}円 → {path}")
        return 0

    if not path.exists():
        print(f"口座がありません。先に `python -m advisor paper init --cash 1000000`（{path}）", file=sys.stderr)
        return 1
    pf = Portfolio.load(path)
    cfg = _cfg(args)
    names, prices, bench = _load(args, start=str(date.today().year - 3) + "-01-01")
    last = _last_prices(prices)

    if args.paper_cmd == "status":
        eq = pf.equity(last)
        print(f"# 紙上の口座（{prices.index[-1].date()} の終値で評価）")
        print(f"- 現金 {pf.cash:,.0f}円 / 株 {pf.market_value(last):,.0f}円 / 合計 {eq:,.0f}円")
        if pf.positions:
            print("\n| 銘柄 | 社名 | 株数 | 取得単価 | 終値 | 損益 |")
            print("|---|---|---:|---:|---:|---:|")
            for t, n in pf.positions.items():
                px = last.get(t, pf.avg_cost.get(t, 0.0))
                pl = round((px - pf.avg_cost.get(t, px)) * n)
                print(f"| {t} | {names.get(t, '')} | {n:,} | {pf.avg_cost.get(t, 0):,.0f} | {px:,.0f} | {pl:+,.0f} |")
        print(f"\n売買の記録 {len(pf.history)} 件")
        return 0

    if args.paper_cmd == "rebalance":
        feats = compute_features(prices, bench, cfg)
        sel = select_at(feats, prices, prices.index[-1], cfg)
        targets = [p.ticker for p in sel.picks] if sel.invested else []
        orders, skipped = plan_orders(pf, targets, last, lot=args.lot, cash_reserve=args.reserve, names=names)
        print(f"# 紙上の入れ替え（{sel.asof.date()} の終値）")
        if not sel.invested:
            print(f"- {sel.reason}")
        print(f"- 目標: {', '.join(f'{t} {names.get(t, '')}' for t in targets) or '現金のみ'}")
        for t in skipped:
            print(f"- 資金が足りず外した: {t} {names.get(t, '')}（1単元 {args.lot}株 = {last[t] * args.lot:,.0f}円）")
        if not orders:
            print("- 注文なし（すでに目標どおり）")
            return 0
        print("\n| 売買 | 銘柄 | 株数 | 単価 | 金額 | 理由 |")
        print("|---|---|---:|---:|---:|---|")
        for o in orders:
            print(f"| {'買' if o.side == 'buy' else '売'} | {o.ticker} | {o.shares:,} | {o.price:,.0f} | {o.value:,.0f} | {o.reason} |")
        if args.dry_run:
            print("\n（--dry-run なので口座には反映していない）")
            return 0
        PaperBroker(pf, when=str(sel.asof.date())).execute(orders)
        pf.save(path)
        print(f"\n紙上で約定させて保存: {path}（現金 {pf.cash:,.0f}円）")
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="advisor", description="株の自動売買ツール（判断と紙上の売買）")
    p.add_argument("--universe", help="銘柄リストのファイル（既定: universe/jp-large.txt）")
    p.add_argument("--top", type=int, default=5, help="持つ銘柄数（既定 5）")
    p.add_argument("--no-regime", action="store_true", help="日経平均による地合いの判定を使わない")
    p.add_argument("--synthetic", action="store_true", help="乱数の株価で動かす（動作確認用）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh", action="store_true", help="キャッシュを使わず取り直す")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="今日の買う候補")
    s.add_argument("--asof", help="判断する日（既定: 最新）")
    s.add_argument("--save", action="store_true", help="reports/<日付>.md に保存")
    s.set_defaults(func=cmd_screen)

    b = sub.add_parser("backtest", help="過去の検証")
    b.add_argument("--start", default="2015-01-01")
    b.add_argument("--end")
    b.add_argument("--cost", type=float, default=0.001, help="片道の取引コスト（既定 0.1%%）")
    b.add_argument("--freq", choices=["M", "W"], default="M", help="入れ替えの頻度（M=月, W=週）")
    b.add_argument("--save", action="store_true", help="reports/backtest-equity.csv に保存")
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="紙上の口座")
    pp.add_argument("--file", help="口座ファイル（既定: data/paper-portfolio.json）")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)
    pi = ps.add_parser("init"); pi.add_argument("--cash", type=float, default=1_000_000); pi.add_argument("--force", action="store_true")
    pr = ps.add_parser("rebalance"); pr.add_argument("--dry-run", action="store_true"); pr.add_argument("--reserve", type=float, default=0.0, help="残しておく現金")
    pr.add_argument("--lot", type=int, default=100, help="売買の単位（既定 100株。単元未満株なら 1）")
    ps.add_parser("status")
    pp.set_defaults(func=cmd_paper)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (RuntimeError, ValueError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
