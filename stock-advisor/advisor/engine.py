"""紙上の口座を1日ぶん進める共通の手順（CLI の paper check / rebalance / auto と、長期シミュレーションが同じものを使う）。

1日の順番:
1. 約定待ちの注文を今日の終値で約定させる（判断した日と同じ日は待つ）
2. 退出ルール（200日線割れ・取得値 −20%・株価なし）
3. 口座の遮断器（評価額が最高値から −20%）→ 全部売って止める
4. 入れ替え（月が変わっていれば、または明示されたとき）
5. 注文を出す（既定は約定待ち → 翌営業日の終値で約定。fill_now なら今日の終値）
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from .broker import PaperBroker
from .portfolio import LOT_SIZE, Order, Portfolio, fill_pending, plan_orders, queue_orders
from .strategy import (Features, Selection, StrategyConfig, combine_sleeves, exit_signals, select_at, sleeve_due_day)

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
    sleeves_refreshed: list[int] = field(default_factory=list)
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
    if dry_run:
        pf = copy.deepcopy(pf)   # 本番と同じ手順を複製の口座で回す（保存しないので何も変わらない）

    # 1) 約定待ち
    if pf.pending:
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
    pf.peak_equity = peak
    for t in pf.positions:
        if t not in last:
            rep.notes.append(f"{t}: 株価がとれない。取得値で評価している。手で確かめる")
    if cfg.portfolio_dd_stop and pf.positions and rep.drawdown <= -cfg.portfolio_dd_stop:
        rep.breaker = True
        done = {o.ticker for o in rep.exits}
        for t, n in pf.positions.items():
            if t in last and t not in done and t not in pending_exit:
                rep.exits.append(Order(t, "sell", n, last[t], reason="遮断器: 全部売って止める"))

    # 4) 入れ替え（列ごと。列 k は月の k×(20/列数) 営業日目以降、その月にまだなら入れ替える。回し忘れた日があっても次の日に拾う）
    n_sleeves = max(1, cfg.sleeves)
    month = when[:7]
    in_month = prices.index[(prices.index <= asof) & (prices.index.to_period("M") == asof.to_period("M"))]
    tdm = len(in_month) - 1
    if not pf.sleeve_month and pf.last_rebalance[:7] == month:
        # 古い口座ファイル（列の記録がない）で今月すでに入れ替えていたら、全部の列を入れ替え済みとみなす（二重にしない）
        for k in range(n_sleeves):
            pf.sleeve_month[str(k)] = month
    if rebalance is True:
        # 手で入れ替え: いちばん最近の入れ替え日を過ぎた列だけを入れ替え直す（他の列はそのまま。列のずらしを壊さない）
        passed = [k for k in range(n_sleeves) if tdm >= sleeve_due_day(k, n_sleeves)]
        due = [max(passed)] if passed else [0]
    elif rebalance is False:
        due = []
    else:
        due = [k for k in range(n_sleeves)
               if tdm >= sleeve_due_day(k, n_sleeves) and pf.sleeve_month.get(str(k)) != month]
    do_rebalance = bool(due) and not rep.breaker
    if do_rebalance:
        sel = select_at(feats, prices, asof, cfg, sectors)
        rep.selection = sel
        rep.sleeves_refreshed = due
        new_w = {p.ticker: p.weight for p in sel.picks} if sel.invested else {}
        for k in due:
            pf.sleeves[str(k)] = new_w
            pf.sleeve_month[str(k)] = month
        # 列の数が変わっていたら、古い列は捨てる
        for key in list(pf.sleeves):
            if int(key) >= n_sleeves:
                del pf.sleeves[key]; pf.sleeve_month.pop(key, None)
        if n_sleeves > 1:
            target_w = combine_sleeves({int(k): w for k, w in pf.sleeves.items()}, n_sleeves)
        else:
            target_w = new_w
        rep.targets = list(target_w)
        # 退出の予定と入れ替えが同じ日なら、入れ替えの目標が優先（バックテストと同じ扱い）
        rep.dropped_exits = [o.ticker for o in rep.exits if o.ticker in rep.targets]
        rep.exits = [o for o in rep.exits if o.ticker not in rep.targets]
        old_pending = [p for p in pf.pending if p["side"] == "sell" and p["ticker"] in rep.targets and p["reason"].startswith("退出")]
        if old_pending:
            pf.pending = [p for p in pf.pending if p not in old_pending]
            rep.dropped_exits += [p["ticker"] for p in old_pending]
        # 退出の売りで戻る現金を、入れ替えの計算に入れる
        view = Portfolio(cash=pf.cash + sum(o.value for o in rep.exits),
                         positions={t: n for t, n in pf.positions.items() if all(o.ticker != t for o in rep.exits)},
                         avg_cost=dict(pf.avg_cost))
        rep.rebalance_orders, rep.skipped = plan_orders(view, target_w, last, lot=lot, cash_reserve=cash_reserve,
                                                        names=names, band=cfg.rebalance_band)
        rep.rebalanced = True

    # 5) 注文を出す（dry_run のときは複製の口座に）
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
