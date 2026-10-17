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
import contextlib
import io
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, sweep_costs, sweep_variants
from .data import load_prices, sanity_report
from .engine import StepReport, daily_step, last_prices
from .portfolio import DEFAULT_PORTFOLIO_FILE, Order, Portfolio
from .strategy import StrategyConfig, compute_features, select_at
from .universe import DEFAULT_BENCHMARK, load_universe_info

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _cfg(args) -> StrategyConfig:
    return StrategyConfig(top_n=args.top, use_regime=args.regime, scoring=args.scoring, weighting=args.weighting)


def _load(args, start: str, extra: list[str] | None = None):
    info = load_universe_info(args.universe)
    names = {t: n for t, (n, _) in info.items()}
    sectors = {t: s for t, (_, s) in info.items()}
    tickers = list(info) + [t for t in (extra or []) if t not in info]   # 保有している銘柄がリストから外れても株価は取る
    prices = load_prices(tickers, start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed)
    bench = None
    if True:  # 比較のため日経平均はいつも取る（地合いの判定に使うのは --regime のときだけ）
        b = load_prices([DEFAULT_BENCHMARK], start=start, synthetic=args.synthetic, refresh=args.refresh, seed=args.seed + 1)
        bench = b[DEFAULT_BENCHMARK] if DEFAULT_BENCHMARK in b else None
        if bench is None:
            logging.warning("ベンチマーク %s を取れないので、地合いの判定なしで進めます", DEFAULT_BENCHMARK)
    return names, sectors, prices, bench


def _recent_start() -> str:
    return f"{date.today().year - 3}-01-01"


def _sanity_lines(prices: pd.DataFrame) -> list[str]:
    issues = sanity_report(prices)
    if not issues:
        return []
    lines = ["", "## データの確認が必要", ""]
    for i in issues[:20]:
        lines.append(f"- {i.ticker}: {i.detail}")
    if len(issues) > 20:
        lines.append(f"- …ほか {len(issues) - 20} 件")
    return lines


def _rule_line(cfg: StrategyConfig, regime: bool) -> str:
    how = "合成スコア（6要素）" if cfg.scoring == "composite" else "12-1モメンタム"
    sizing = "逆ボラ配分" if cfg.weighting == "invvol" else "同額"
    s = (f"{how}上位 {cfg.top_n} 銘柄（{sizing}）、200日線の上、年率変動率 {cfg.max_vol:.0%} 以下、"
         f"同じ業種は {cfg.max_per_sector} 銘柄まで")
    if regime:
        s += "、日経平均が200日線の上のときだけ"
    return s


def _guard_line(cfg: StrategyConfig) -> str:
    parts = []
    if cfg.exit_below_trend:
        parts.append("200日線割れで退出")
    if cfg.stop_loss:
        parts.append(f"取得値 −{cfg.stop_loss:.0%} で退出")
    if cfg.portfolio_dd_stop:
        parts.append(f"遮断器: 最高値から −{cfg.portfolio_dd_stop:.0%}")
    return "、".join(parts) if parts else "守りの層なし（順張りだけ）"


# ---------- screen ----------
def cmd_screen(args) -> int:
    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=_recent_start())
    feats = compute_features(prices, bench, cfg)
    asof = pd.Timestamp(args.asof) if args.asof else prices.index[-1]
    sel = select_at(feats, prices, asof, cfg, sectors)

    lines = [f"# 買う候補（{sel.asof.date()} 時点）", ""]
    lines.append(f"- ルール: {_rule_line(cfg, cfg.use_regime and bench is not None)}（{_guard_line(cfg)}）")
    lines.append(f"- 候補の母数: {len(prices.columns)} 銘柄、条件を満たしたのは {sel.eligible} 銘柄")
    if sel.sector_skipped:
        lines.append("- 業種の上限で外した: " + ", ".join(f"{t} {names.get(t, '')}" for t in sel.sector_skipped))
    lines.append("")
    if not sel.invested:
        lines.append(f"**今回は買わない。** {sel.reason}")
    else:
        lines.append("| 順位 | 銘柄 | 社名 | 業種 | 終値 | 1単元(100株) | 12-1モメンタム | 年率変動率 |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
        for i, p in enumerate(sel.picks, 1):
            lines.append(f"| {i} | {p.ticker} | {names.get(p.ticker, '')} | {p.sector} | {p.price:,.0f} | {p.price * 100:,.0f} | {p.momentum:+.1%} | {p.vol:.0%} |")
    lines += _sanity_lines(prices)
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
         ("年率変動率", "annual_vol", "{:.1%}"), ("シャープ", "sharpe", "{:.2f}"), ("最大下落", "max_drawdown", "{:.1%}"),
         ("最悪の月", "worst_month", "{:.1%}")]


def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=args.start)
    res = run_backtest(prices, bench, cfg, end=args.end, cost=args.cost, freq=args.freq, sectors=sectors,
                       execution_lag=args.lag)
    m = res.metrics
    print(f"# バックテスト {res.equity.index[0].date()} 〜 {res.equity.index[-1].date()}（{m['years']:.1f}年）")
    print(f"- 銘柄数 {len(prices.columns)}、{_rule_line(cfg, cfg.use_regime and bench is not None)}")
    print(f"- {'月' if args.freq == 'M' else '週'}1回入れ替え、片道コスト {args.cost:.2%}、約定は判断の {args.lag} 営業日後の終値、"
          f"{_guard_line(cfg)}")
    print()
    print("| | ルール | 日経平均(買い持ち・配当なし) | 候補67銘柄を均等に買い持ち |".replace("67", str(len(prices.columns))))
    print("|---|---:|---:|---:|")
    for label, key, fmt in _ROWS:
        b = fmt.format(res.bench_metrics[key]) if res.bench_metrics and key in res.bench_metrics else "-"
        u = fmt.format(res.universe_metrics[key]) if res.universe_metrics and key in res.universe_metrics else "-"
        print(f"| {label} | {fmt.format(m[key])} | {b} | {u} |")
    print("\n※ 日経平均は株価指数（配当を含まない）。ルールと「均等に買い持ち」は配当込みの調整後終値。"
          "候補は今の大型株なので、生き残った銘柄だけを見ている（生存バイアス）。")
    print(f"\n入れ替え {int(m['rebalances'])} 回、売買 {int(m['trades'])} 件、株を持っていた日の割合 {m['time_in_market']:.0%}")
    print(f"退出: 200日線割れ {int(m['exit_trend'])} 回 / 取得値 −{cfg.stop_loss:.0%} {int(m['exit_stop'])} 回 / "
          f"株価なし {int(m['exit_missing'])} 回、遮断器 {int(m['dd_stops'])} 回")
    if m.get("exits_judged"):
        print(f"退出して正解だった割合（売った後21営業日で下がった）: {m['exit_correct_rate']:.0%}"
              f"（{int(m['exits_judged'])} 件、売った後の平均 {m['exit_avg_fwd_return']:+.1%}）")
    if args.by_year and res.yearly is not None:
        print("\n## 年ごと\n")
        print("| 年 | ルール | 日経平均 | 均等に買い持ち |")
        print("|---|---:|---:|---:|")
        for y, row in res.yearly.iterrows():
            b = f"{row['benchmark']:+.1%}" if "benchmark" in row and pd.notna(row["benchmark"]) else "-"
            u = f"{row['universe']:+.1%}" if "universe" in row and pd.notna(row["universe"]) else "-"
            print(f"| {y} | {row['strategy']:+.1%} | {b} | {u} |")

    if args.sweep:
        print("\n## 隣の値で結果がどう変わるか（大きく変わるなら、その値を信じない）\n")
        print("| 設定 | 年率リターン | 最大下落 | シャープ | 売買 |")
        print("|---|---:|---:|---:|---:|")
        for label, c in sweep_variants(cfg):
            r = run_backtest(prices, bench, c, end=args.end, cost=args.cost, freq=args.freq, sectors=sectors,
                             execution_lag=args.lag).metrics
            print(f"| {label} | {r['cagr']:+.1%} | {r['max_drawdown']:.1%} | {r['sharpe']:.2f} | {int(r['trades'])} |")
        for label, c in sweep_costs():
            if abs(c - args.cost) < 1e-12:
                continue
            r = run_backtest(prices, bench, cfg, end=args.end, cost=c, freq=args.freq, sectors=sectors,
                             execution_lag=args.lag).metrics
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


def _print_step(rep: StepReport, pf: Portfolio, names: dict[str, str], args) -> None:
    if rep.filled and not args.fill_now:
        print(f"## 約定待ちの注文を {rep.asof.date()} の終値で約定")
        _print_orders(rep.filled)
        print()
    for n in rep.notes:
        print(f"- {n}")
    if pf.halted and not rep.breaker:
        return
    print(f"- 評価額 {rep.equity:,.0f}円（最高値から {rep.drawdown:+.1%}）")
    if rep.exits:
        print("\n## 退出")
        _print_orders(rep.exits)
    else:
        print("- 退出する銘柄なし")
    if rep.breaker:
        print(f"\n**遮断器が動いた。** 全部売って止める。理由を調べて `paper resume --why '…'` で再開する")
    if rep.rebalanced:
        sel = rep.selection
        print("\n## 入れ替え")
        if sel is not None and not sel.invested:
            print(f"- {sel.reason}")
        print(f"- 目標: {', '.join(f'{t} {names.get(t, '')}' for t in rep.targets) or '現金のみ'}")
        if rep.dropped_exits:
            print("- 入れ替えの目標に残るので退出を取り消した: " + ", ".join(rep.dropped_exits))
        for t in rep.skipped:
            print(f"- 資金が足りず外した: {t} {names.get(t, '')}")
        if rep.rebalance_orders:
            _print_orders(rep.rebalance_orders)
        else:
            print("- 注文なし（すでに目標どおり）")
    elif args.paper_cmd == "auto":
        print(f"\n- 入れ替えは月1回（最後: {pf.last_rebalance or 'まだ'}）。今日はしない")
    if rep.orders:
        if args.dry_run:
            print("\n（--dry-run なので口座には反映していない）")
        elif args.fill_now:
            print("\n（--fill-now: 今日の終値で約定させた）")
        else:
            print("\n（約定待ちに入れた。次の営業日の `paper auto` / `check` でその日の終値で約定する）")


class _Tee(io.TextIOBase):
    """画面に出しつつ、文字列にも貯める（--save 用）。"""

    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.stream.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()


def cmd_paper(args) -> int:
    if getattr(args, "save", False) and args.paper_cmd in ("check", "rebalance", "auto"):
        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            code = _cmd_paper(args)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        asof = getattr(args, "_asof", None) or date.today()
        out = REPORT_DIR / f"{asof}-paper.md"
        with out.open("a", encoding="utf-8") as f:
            f.write("\n" + tee.buf.getvalue())
        print(f"追記: {out}", file=sys.stderr)
        return code
    return _cmd_paper(args)


def _cmd_paper(args) -> int:
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
        pf.halted_on = ""
        pf.peak_equity = 0.0  # 次の check で、いまの評価額が新しい基準になる
        pf.history.append({"when": datetime.now().isoformat(timespec="seconds"), "event": "resume",
                           "halted_reason": pf.halted_reason, "why": args.why})
        pf.halted_reason = ""
        pf.save(path)
        print(f"再開しました。理由を記録: {args.why}")
        return 0

    cfg = _cfg(args)
    names, sectors, prices, bench = _load(args, start=_recent_start(), extra=list(pf.positions) + [p["ticker"] for p in pf.pending])
    asof = prices.index[-1]
    args._asof = asof.date()
    last = last_prices(prices, asof)

    if args.paper_cmd == "status":
        eq = pf.equity(last)
        print(f"# 紙上の口座（{asof.date()} の終値で評価）")
        print(f"- 現金 {pf.cash:,.0f}円 / 株 {pf.market_value(last):,.0f}円 / 合計 {eq:,.0f}円"
              f"（最高値 {max(pf.peak_equity, eq):,.0f}円）")
        if pf.halted:
            print(f"- **遮断器で停止中**: {pf.halted_reason}")
        print(f"- 最後の入れ替え: {pf.last_rebalance or 'まだ'}")
        if pf.pending:
            print("- 約定待ち: " + ", ".join(f"{p['side']} {p['ticker']} {p['shares']}株" for p in pf.pending))
        for line in _sanity_lines(prices.loc[:, [t for t in pf.positions if t in prices.columns]] if pf.positions else prices.iloc[:, :0]):
            print(line)
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
    title = {"check": "退出判定", "rebalance": "入れ替え", "auto": "自動運用"}[args.paper_cmd]
    print(f"# 紙上の{title}（{asof.date()} の終値）")
    for line in _sanity_lines(prices):
        print(line)
    rebalance = {"check": False, "rebalance": True, "auto": None}[args.paper_cmd]
    rep = daily_step(pf, cfg, feats, prices, asof, sectors=sectors, names=names, lot=args.lot,
                     cash_reserve=args.reserve, fill_now=args.fill_now, rebalance=rebalance, dry_run=args.dry_run)
    _print_step(rep, pf, names, args)
    if not args.dry_run:
        pf.save(path)
        print(f"\n保存: {path}（現金 {pf.cash:,.0f}円）")
    return 0


# ---------- research ----------
def cmd_research(args) -> int:
    from . import research as R
    names, sectors, prices, bench = _load(args, start=args.start)
    if bench is None:
        print("ベンチマークが取れないので検証できません", file=sys.stderr)
        return 2
    out = R.run_all(prices, bench, sectors, bootstrap_runs=args.bootstrap, weight_runs=args.weights,
                    cost=args.cost, split=args.split, log=lambda m: print(m, file=sys.stderr))
    out["split"] = args.split
    text = R.report(out, title=f"1000回の検証（{prices.index[0].date()}〜{prices.index[-1].date()}、{len(prices.columns)} 銘柄）")
    print(text)
    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        outp = REPORT_DIR / f"research-{date.today()}.md"
        outp.write_text(text, encoding="utf-8")
        print(f"保存: {outp}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="advisor", description="株の自動売買ツール（判断と紙上の売買）")
    p.add_argument("--universe", help="銘柄リストのファイル（既定: universe/jp-large.txt）")
    p.add_argument("--top", type=int, default=8, help="持つ銘柄数（既定 8）")
    p.add_argument("--regime", action="store_true", help="日経平均が200日線の下なら全部現金にする（既定では使わない）")
    p.add_argument("--scoring", choices=["m12", "composite"], default="m12", help="並べ方（m12 = 12-1モメンタムだけ、composite = 合成スコア）")
    p.add_argument("--weighting", choices=["equal", "invvol"], default="equal", help="比率（equal = 同額、invvol = 逆ボラティリティ）")
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
    b.add_argument("--by-year", action="store_true", help="年ごとのリターンも出す")
    b.add_argument("--lag", type=int, default=1, help="判断から約定までの営業日数（既定 1。0 は判断した日の終値で約定＝楽観的）")
    b.add_argument("--save", action="store_true", help="reports/backtest-equity.csv に保存")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("research", help="1000回の検証（ブートストラップ・重みのランダム化・日取り・開始日・前半後半・格子）")
    r.add_argument("--start", default="2010-01-01")
    r.add_argument("--bootstrap", type=int, default=300, help="仮想の歴史の本数（既定 300。3 戦略を走らせるので 900 回）")
    r.add_argument("--weights", type=int, default=200, help="重みのランダム化の回数")
    r.add_argument("--split", default="2019-01-01", help="前半/後半の分かれ目")
    r.add_argument("--cost", type=float, default=0.001)
    r.add_argument("--save", action="store_true", help="reports/research-<日付>.md に保存")
    r.set_defaults(func=cmd_research)

    pp = sub.add_parser("paper", help="紙上の口座")
    pp.add_argument("--file", help="口座ファイル（既定: data/paper-portfolio.json）")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)
    pi = ps.add_parser("init"); pi.add_argument("--cash", type=float, default=1_000_000); pi.add_argument("--force", action="store_true")
    for name in ("check", "rebalance", "auto"):
        q = ps.add_parser(name)
        q.add_argument("--dry-run", action="store_true")
        q.add_argument("--reserve", type=float, default=0.0, help="残しておく現金")
        q.add_argument("--lot", type=int, default=100, help="売買の単位（既定 100株。単元未満株なら 1）")
        q.add_argument("--fill-now", action="store_true", help="今日の終値で約定させる（既定は翌営業日の終値）")
        q.add_argument("--save", action="store_true", help="出力を reports/<日付>.md に追記する")
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
