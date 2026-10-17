"""コマンド。

  python -m advisor screen                      今日買う候補を出す
  python -m advisor backtest --start 2015-01-01 ルールを過去で検証する（--sweep で隣の値も）
  python -m advisor paper init --cash 1000000   紙上の口座を作る
  python -m advisor paper check                 退出ルール（200日線割れ・−20%・遮断器）を今日の終値で確かめ、紙上で売る
  python -m advisor paper rebalance             候補に合わせて紙上で売買する
  python -m advisor paper auto                  check を毎日、月が変わっていれば rebalance（GitHub Actions 用）
  python -m advisor paper status                紙上の口座の状態
  python -m advisor paper resume                遮断器で止まった口座を再開する（理由を --why で残す）

共通: --synthetic（乱数データで動かす。ネットなしの動作確認用）, --universe FILE, --top N
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, sweep_variants
from .broker import PaperBroker
from .data import load_prices
from .portfolio import DEFAULT_PORTFOLIO_FILE, Order, Portfolio, plan_orders
from .strategy import StrategyConfig, compute_features, exit_signals, select_at
from .universe import DEFAULT_BENCHMARK, load_universe_info

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _cfg(args) -> StrategyConfig:
    return StrategyConfig(top_n=args.top, use_regime=not args.no_regime)


def _load(args, start: str):
    info = load_universe_info(args.universe)
    names = {t: n for t, (n, _) in info.items()}
    sectors = {t: s for t, (_, s) in info.items()}
    prices = load_prices(list(info), start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed)
    bench = None
    if not args.no_regime:
        b = load_prices([DEFAULT_BENCHMARK], start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed + 1)
        bench = b[DEFAULT_BENCHMARK] if DEFAULT_BENCHMARK in b else None
        if bench is None:
            logging.warning("ベンチマーク %s を取れないので、地合いの判定なしで進めます", DEFAULT_BENCHMARK)
    return names, sectors, prices, bench


def _recent_start() -> str:
    return f"{date.today().year - 3}-01-01"


def _last_prices(prices: pd.DataFrame) -> dict[str, float]:
    last = prices.ffill().iloc[-1]
    return {t: float(v) for t, v in last.items() if pd.notna(v)}


def _rule_line(cfg: StrategyConfig, regime: bool) -> str:
    s = (f"12-1モメンタム上位 {cfg.top_n} 銘柄、200日線の上、年率変動率 {cfg.max_vol:.0%} 以下、"
         f"同じ業種は {cfg.max_per_sector} 銘柄まで")
    if regime:
        s += "、日経平均が200日線の上のときだけ"
    return s


# ---------- screen ----------
def cmd_screen(args) -> int:
    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=_recent_start())
    feats = compute_features(prices, bench, cfg)
    asof = pd.Timestamp(args.asof) if args.asof else prices.index[-1]
    sel = select_at(feats, prices, asof, cfg, sectors)

    lines = [f"# 買う候補（{sel.asof.date()} 時点）", ""]
    lines.append(f"- ルール: {_rule_line(cfg, cfg.use_regime and bench is not None)}")
    lines.append(f"- 候補の母数: {len(prices.columns)} 銘柄、条件を満たしたのは {sel.eligible} 銘柄")
    if sel.sector_skipped:
        lines.append("- 業種の上限で外した: " + ", ".join(f"{t} {names.get(t, '')}" for t in sel.sector_skipped))
    lines.append("")
    if not sel.invested:
        lines.append(f"**今回は買わない。** {sel.reason}")
    else:
        lines.append("| 順位 | 銘柄 | 社名 | 業種 | 終値 | 12-1モメンタム | 年率変動率 |")
        lines.append("|---:|---|---|---|---:|---:|---:|")
        for i, p in enumerate(sel.picks, 1):
            lines.append(f"| {i} | {p.ticker} | {names.get(p.ticker, '')} | {p.sector} | {p.price:,.0f} | {p.momentum:+.1%} | {p.vol:.0%} |")
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
_ROWS = [("最終倍率", "final_multiple", "{:.2f}倍"), ("年率リターン", "cagr", "{:+.1%}"),
         ("年率変動率", "annual_vol", "{:.1%}"), ("シャープ", "sharpe", "{:.2f}"), ("最大下落", "max_drawdown", "{:.1%}")]


def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=args.start)
    res = run_backtest(prices, bench, cfg, end=args.end, cost=args.cost, freq=args.freq, sectors=sectors)
    m = res.metrics
    print(f"# バックテスト {res.equity.index[0].date()} 〜 {res.equity.index[-1].date()}（{m['years']:.1f}年）")
    print(f"- 銘柄数 {len(prices.columns)}、{_rule_line(cfg, cfg.use_regime and bench is not None)}")
    print(f"- {'月' if args.freq == 'M' else '週'}1回入れ替え、片道コスト {args.cost:.2%}、"
          f"退出: 200日線割れ / 取得値 −{cfg.stop_loss:.0%}、遮断器: 最高値から −{cfg.portfolio_dd_stop:.0%}")
    print()
    print("| | ルール | ベンチマーク(買い持ち) |")
    print("|---|---:|---:|")
    for label, key, fmt in _ROWS:
        b = fmt.format(res.bench_metrics[key]) if res.bench_metrics and key in res.bench_metrics else "-"
        print(f"| {label} | {fmt.format(m[key])} | {b} |")
    print(f"\n入れ替え {int(m['rebalances'])} 回、売買 {int(m['trades'])} 件、"
          f"退出: 200日線割れ {int(m['exit_trend'])} 回 / −{cfg.stop_loss:.0%} {int(m['exit_stop'])} 回、遮断器 {int(m['dd_stops'])} 回")

    if args.sweep:
        print("\n## 隣の値で結果がどう変わるか（大きく変わるなら、その値を信じない）\n")
        print("| 設定 | 年率リターン | 最大下落 | シャープ | 売買 |")
        print("|---|---:|---:|---:|---:|")
        for label, c in sweep_variants(cfg):
            r = run_backtest(prices, bench, c, end=args.end, cost=args.cost, freq=args.freq, sectors=sectors).metrics
            print(f"| {label} | {r['cagr']:+.1%} | {r['max_drawdown']:.1%} | {r['sharpe']:.2f} | {int(r['trades'])} |")

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
def _print_orders(orders: list[Order]) -> None:
    print("\n| 売買 | 銘柄 | 株数 | 単価 | 金額 | 理由 |")
    print("|---|---|---:|---:|---:|---|")
    for o in orders:
        print(f"| {'買' if o.side == 'buy' else '売'} | {o.ticker} | {o.shares:,} | {o.price:,.0f} | {o.value:,.0f} | {o.reason} |")


def _paper_check(pf: Portfolio, cfg, feats, prices, last, names, asof, dry_run: bool) -> list[Order]:
    """退出ルールと遮断器。売った注文を返す（dry_run なら反映しない）。"""
    orders: list[Order] = []
    if pf.halted:
        print(f"- 遮断器で停止中（{pf.halted_reason}）。`paper resume --why '理由'` で再開するまで何もしない")
        return orders
    sigs = exit_signals(feats, prices, asof, cfg, {t: pf.avg_cost.get(t, 0.0) for t in pf.positions})
    for s in sigs:
        why = "200日線を割った" if s.reason == "trend" else f"取得値から {s.change:+.1%}（−{cfg.stop_loss:.0%} の保険）"
        orders.append(Order(s.ticker, "sell", pf.positions[s.ticker], s.price, reason=f"退出: {why}"))
    eq = pf.equity(last)
    pf.peak_equity = max(pf.peak_equity, eq)
    dd = eq / pf.peak_equity - 1 if pf.peak_equity > 0 else 0.0
    print(f"- 評価額 {eq:,.0f}円（最高値 {pf.peak_equity:,.0f}円から {dd:+.1%}）")
    breaker = bool(cfg.portfolio_dd_stop and pf.positions and dd <= -cfg.portfolio_dd_stop)
    if breaker:
        for t, n in pf.positions.items():
            if t in last and all(o.ticker != t for o in orders):
                orders.append(Order(t, "sell", n, last[t], reason="遮断器: 全部売って止める"))
    if not orders:
        print("- 退出する銘柄なし")
        return orders
    _print_orders(orders)
    if dry_run:
        print("\n（--dry-run なので口座には反映していない）")
        return orders
    PaperBroker(pf, when=str(asof.date())).execute(orders)
    if breaker:
        pf.halted = True
        pf.halted_reason = f"{asof.date()} に評価額が最高値から {dd:.1%}"
        print(f"\n**遮断器が動いた。** 全部売って止めた。理由を調べて `paper resume --why '…'` で再開する")
    return orders


def _paper_rebalance(pf: Portfolio, cfg, feats, prices, last, names, sectors, asof, args) -> list[Order]:
    if pf.halted:
        print(f"- 遮断器で停止中（{pf.halted_reason}）。入れ替えしない")
        return []
    sel = select_at(feats, prices, asof, cfg, sectors)
    targets = [p.ticker for p in sel.picks] if sel.invested else []
    orders, skipped = plan_orders(pf, targets, last, lot=args.lot, cash_reserve=args.reserve, names=names)
    if not sel.invested:
        print(f"- {sel.reason}")
    print(f"- 目標: {', '.join(f'{t} {names.get(t, '')}' for t in targets) or '現金のみ'}")
    for t in skipped:
        print(f"- 資金が足りず外した: {t} {names.get(t, '')}（1単元 {args.lot}株 = {last[t] * args.lot:,.0f}円）")
    if not orders:
        print("- 注文なし（すでに目標どおり）")
    else:
        _print_orders(orders)
        if args.dry_run:
            print("\n（--dry-run なので口座には反映していない）")
            return orders
        PaperBroker(pf, when=str(asof.date())).execute(orders)
    if not args.dry_run:
        pf.last_rebalance = str(asof.date())
        pf.peak_equity = max(pf.peak_equity, pf.equity(last))
    return orders


def cmd_paper(args) -> int:
    path = Path(args.file) if args.file else DEFAULT_PORTFOLIO_FILE
    if args.paper_cmd == "init":
        if path.exists() and not args.force:
            print(f"すでにあります: {path}（作り直すなら --force）", file=sys.stderr)
            return 1
        Portfolio(cash=float(args.cash), peak_equity=float(args.cash)).save(path)
        print(f"紙上の口座を作りました: 現金 {args.cash:,.0f}円 → {path}")
        return 0

    if not path.exists():
        print(f"口座がありません。先に `python -m advisor paper init --cash 1000000`（{path}）", file=sys.stderr)
        return 1
    pf = Portfolio.load(path)

    if args.paper_cmd == "resume":
        if not pf.halted:
            print("止まっていません")
            return 0
        pf.halted = False
        pf.history.append({"when": datetime.now().isoformat(timespec="seconds"), "event": "resume",
                           "halted_reason": pf.halted_reason, "why": args.why})
        pf.halted_reason = ""
        pf.save(path)
        print(f"再開しました。理由を記録: {args.why}")
        return 0

    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=_recent_start())
    last = _last_prices(prices)
    asof = prices.index[-1]

    if args.paper_cmd == "status":
        eq = pf.equity(last)
        print(f"# 紙上の口座（{asof.date()} の終値で評価）")
        print(f"- 現金 {pf.cash:,.0f}円 / 株 {pf.market_value(last):,.0f}円 / 合計 {eq:,.0f}円"
              f"（最高値 {max(pf.peak_equity, eq):,.0f}円）")
        if pf.halted:
            print(f"- **遮断器で停止中**: {pf.halted_reason}")
        print(f"- 最後の入れ替え: {pf.last_rebalance or 'まだ'}")
        if pf.positions:
            print("\n| 銘柄 | 社名 | 株数 | 取得単価 | 終値 | 損益 |")
            print("|---|---|---:|---:|---:|---:|")
            for t, n in pf.positions.items():
                px = last.get(t, pf.avg_cost.get(t, 0.0))
                pl = round((px - pf.avg_cost.get(t, px)) * n)
                print(f"| {t} | {names.get(t, '')} | {n:,} | {pf.avg_cost.get(t, 0):,.0f} | {px:,.0f} | {pl:+,.0f} |")
        print(f"\n売買の記録 {len(pf.history)} 件")
        return 0

    feats = compute_features(prices, bench, cfg)
    if args.paper_cmd == "check":
        print(f"# 紙上の退出判定（{asof.date()} の終値）")
        _paper_check(pf, cfg, feats, prices, last, names, asof, args.dry_run)
    elif args.paper_cmd == "rebalance":
        print(f"# 紙上の入れ替え（{asof.date()} の終値）")
        _paper_rebalance(pf, cfg, feats, prices, last, names, sectors, asof, args)
    elif args.paper_cmd == "auto":
        print(f"# 紙上の自動運用（{asof.date()} の終値）")
        _paper_check(pf, cfg, feats, prices, last, names, asof, args.dry_run)
        due = (not pf.last_rebalance) or pf.last_rebalance[:7] != str(asof.date())[:7]
        if due:
            print("\n## 月が変わったので入れ替え")
            _paper_rebalance(pf, cfg, feats, prices, last, names, sectors, asof, args)
        else:
            print(f"\n- 入れ替えは月1回（最後: {pf.last_rebalance}）。今日はしない")
    if not args.dry_run:
        pf.save(path)
        print(f"\n保存: {path}（現金 {pf.cash:,.0f}円）")
    return 0


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
    b.add_argument("--sweep", action="store_true", help="隣の値でも検証して並べる（過学習の確認）")
    b.add_argument("--save", action="store_true", help="reports/backtest-equity.csv に保存")
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="紙上の口座")
    pp.add_argument("--file", help="口座ファイル（既定: data/paper-portfolio.json）")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)
    pi = ps.add_parser("init"); pi.add_argument("--cash", type=float, default=1_000_000); pi.add_argument("--force", action="store_true")
    for name in ("check", "rebalance", "auto"):
        q = ps.add_parser(name)
        q.add_argument("--dry-run", action="store_true")
        q.add_argument("--reserve", type=float, default=0.0, help="残しておく現金")
        q.add_argument("--lot", type=int, default=100, help="売買の単位（既定 100株。単元未満株なら 1）")
    ps.add_parser("status")
    pr = ps.add_parser("resume"); pr.add_argument("--why", required=True, help="再開する理由（記録に残す）")
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
