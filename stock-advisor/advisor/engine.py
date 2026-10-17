"""紙上の口座を1日ぶん進める共通の手順（CLI の paper check / rebalance / auto と、長期シミュレーションが同じものを使う）。

1日の順番:
1. 約定待ちの注文を今日の終値で約定させる（判断した日と同じ日は待つ）
2. 退出ルール（200日線割れ・取得値 −20%・株価なし）
3. 口座の遮断器（評価額が最高値から −20%）→ 全部売って止める
4. 入れ替え（月が変わっていれば、または明示されたとき）
5. 注文を出す（既定は約定待ち → 翌営業日の終値で約定。fill_now なら今日の終値）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .broker import PaperBroker
from .portfolio import LOT_SIZE, Order, Portfolio, fill_pending, plan_orders, queue_orders
from .strategy import Features, Selection, StrategyConfig, exit_signals, select_at

EXIT_REASON = {
    "trend": "200日線を割った",
    "stop": "取得値から {chg:+.1%}（−{stop:.0%} の保険）",
    "missing": "株価がとれない（上場廃止かデータ障害。売れるか確かめる）",
}


@dataclass
class StepReport:
    asof: pd.Timestamp
    filled: list[Order] = field(default_factory=list)      # 今日約定した注文
    exits: list[Order] = field(default_factory=list)       # 退出の売り（出した注文）
    breaker: bool = False                                   # 遮断器が動いた
    drawdown: float = 0.0
    equity: float = 0.0
    rebalanced: bool = False
    selection: Selection | None = None
    targets: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)        # 資金が足りず外した
    dropped_exits: list[str] = field(default_factory=list)  # 入れ替えの目標に残るので取り消した退出
    rebalance_orders: list[Order] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def orders(self) -> list[Order]:
        return self.exits + self.rebalance_orders


def last_prices(prices: pd.DataFrame, asof: pd.Timestamp) -> dict[str, float]:
    row = prices.loc[:asof].ffill().iloc[-1]
    return {t: float(v) for t, v in row.items() if pd.notna(v)}


def daily_step(
    pf: Portfolio,
    cfg: StrategyConfig,
    feats: Features,
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    sectors: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
    lot: int = LOT_SIZE,
    cash_reserve: float = 0.0,
    fill_now: bool = False,
    rebalance: bool | None = None,   # None = 月が変わっていれば
    dry_run: bool = False,
) -> StepReport:
    asof = pd.Timestamp(asof)
    when = str(asof.date())
    last = last_prices(prices, asof)
    rep = StepReport(asof=asof)

    # 1) 約定待ち
    if pf.pending and not dry_run:
        rep.filled = fill_pending(pf, last, when, lot=lot)
        if pf.pending:
            rep.notes.append("まだ待ち: " + ", ".join(p["ticker"] for p in pf.pending) + "（株価がとれない、または判断した日と同じ日）")

    rep.equity = pf.equity(last)
    if pf.halted:
        rep.notes.append(f"遮断器で停止中（{pf.halted_reason}）。`paper resume --why '理由'` で再開するまで何もしない")
        return rep

    # 2) 退出
    pending_exit = {p["ticker"] for p in pf.pending if p["side"] == "sell"}
    for s in exit_signals(feats, prices, asof, cfg, {t: pf.avg_cost.get(t, 0.0) for t in pf.positions}):
        if s.ticker in pending_exit:
            continue
        why = EXIT_REASON[s.reason].format(chg=s.change, stop=cfg.stop_loss)
        rep.exits.append(Order(s.ticker, "sell", pf.positions[s.ticker], s.price, reason=f"退出: {why}"))

    # 3) 遮断器
    peak = max(pf.peak_equity, rep.equity)
    rep.drawdown = rep.equity / peak - 1 if peak > 0 else 0.0
    if not dry_run:
        pf.peak_equity = peak
    if cfg.portfolio_dd_stop and pf.positions and rep.drawdown <= -cfg.portfolio_dd_stop:
        rep.breaker = True
        done = {o.ticker for o in rep.exits}
        for t, n in pf.positions.items():
            if t in last and t not in done and t not in pending_exit:
                rep.exits.append(Order(t, "sell", n, last[t], reason="遮断器: 全部売って止める"))

    # 4) 入れ替え
    month_changed = (not pf.last_rebalance) or pf.last_rebalance[:7] != when[:7]
    do_rebalance = (rebalance if rebalance is not None else month_changed) and not rep.breaker
    if do_rebalance:
        sel = select_at(feats, prices, asof, cfg, sectors)
        rep.selection = sel
        rep.targets = [p.ticker for p in sel.picks] if sel.invested else []
        # 退出の予定と入れ替えが同じ日なら、入れ替えの目標が優先（バックテストと同じ扱い）
        rep.dropped_exits = [o.ticker for o in rep.exits if o.ticker in rep.targets]
        rep.exits = [o for o in rep.exits if o.ticker not in rep.targets]
        old_pending = [p for p in pf.pending if p["side"] == "sell" and p["ticker"] in rep.targets and p["reason"].startswith("退出")]
        if old_pending and not dry_run:
            pf.pending = [p for p in pf.pending if p not in old_pending]
            rep.dropped_exits += [p["ticker"] for p in old_pending]
        # 退出の売りで戻る現金を、入れ替えの計算に入れる
        view = Portfolio(cash=pf.cash + sum(o.value for o in rep.exits),
                         positions={t: n for t, n in pf.positions.items() if all(o.ticker != t for o in rep.exits)},
                         avg_cost=dict(pf.avg_cost))
        rep.rebalance_orders, rep.skipped = plan_orders(view, rep.targets, last, lot=lot, cash_reserve=cash_reserve,
                                                        names=names, band=cfg.rebalance_band)
        rep.rebalanced = True

    # 5) 注文を出す
    if dry_run:
        return rep
    orders = rep.orders
    if orders:
        if fill_now:
            PaperBroker(pf, when=when).execute(orders)
            rep.filled += orders
        else:
            queue_orders(pf, orders, when)
    if rep.breaker:
        pf.halted = True
        pf.halted_on = when
        pf.halted_reason = f"{when} に評価額が最高値から {rep.drawdown:.1%}"
    if rep.rebalanced:
        pf.last_rebalance = when
    return rep


def simulate_paper(
    prices: pd.DataFrame,
    feats: Features,
    cfg: StrategyConfig,
    *,
    cash: float = 10_000_000,
    lot: int = LOT_SIZE,
    sectors: dict[str, str] | None = None,
    start: pd.Timestamp | None = None,
    auto_resume: bool = True,
) -> tuple[pd.Series, Portfolio]:
    """毎日 daily_step を回す長期シミュレーション。戻り値は（評価額の推移, 最後の口座）。
    auto_resume: 遮断器のあと、次の月初に自動で再開する（バックテストと同じ仮定）。"""
    pf = Portfolio(cash=cash, peak_equity=cash)
    days = prices.index[prices.index >= start] if start is not None else prices.index[cfg.min_history:]
    eq = []
    for d in days:
        when = str(d.date())
        if pf.halted and auto_resume and pf.halted_on[:7] != when[:7]:
            pf.halted, pf.halted_reason, pf.peak_equity = False, "", 0.0  # 翌月の最初の営業日に再開
        daily_step(pf, cfg, feats, prices, d, sectors=sectors, lot=lot)
        eq.append(pf.equity(last_prices(prices, d)))
    return pd.Series(eq, index=days), pf
