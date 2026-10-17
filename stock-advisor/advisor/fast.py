"""高速なバックテスト（検証を何百回も回すため）。

通常の run_backtest と同じ判断（select_at を月初に呼ぶ）だが、退出ルール・遮断器・バンドは扱わず、
「判断の翌営業日の終値で目標比率に入れ替え、次の入れ替えまで持ちっぱなし」を行列計算で評価する。
守りの層なし・バンド 0 のときは run_backtest とほぼ一致する（テストで確認）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import metrics_from_equity, rebalance_dates
from .strategy import StrategyConfig, compute_features, select_at


def fast_backtest(
    prices: pd.DataFrame,
    bench: pd.Series | None,
    cfg: StrategyConfig,
    *,
    cost: float = 0.001,
    execution_lag: int = 1,
    sectors: dict[str, str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    rebalance_offset: int = 0,     # 月の何営業日目に判断するか（0 = 最初の営業日）
    feats=None,
) -> dict:
    prices = prices.sort_index()
    if feats is None:
        feats = compute_features(prices, bench, cfg)
    filled = prices.ffill()
    rets = filled.pct_change().fillna(0.0)
    idx = prices.index
    first_ok = idx[min(cfg.min_history, len(idx) - 1)]
    s = max(pd.Timestamp(start), first_ok) if start is not None else first_ok
    e = pd.Timestamp(end) if end is not None else idx[-1]
    days = idx[(idx >= s) & (idx <= e)]
    if len(days) < 30:
        raise ValueError("期間が短すぎます")

    # 判断日: 各月の rebalance_offset 営業日目
    month_first = rebalance_dates(days, "M")
    decision = []
    for d in month_first:
        pos = days.get_loc(d) + rebalance_offset
        if pos < len(days):
            decision.append(days[pos])
    # 目標比率（判断日）→ 約定日から次の約定日の前日まで保持
    W = pd.DataFrame(0.0, index=days, columns=prices.columns)
    exec_days = []
    for d in decision:
        sel = select_at(feats, prices, d, cfg, sectors)
        pos = days.get_loc(d) + execution_lag
        if pos >= len(days):
            break
        exec_days.append(days[pos])
        if sel.invested:
            for p in sel.picks:
                W.iloc[pos:, W.columns.get_loc(p.ticker)] = p.weight
            others = [c for c in W.columns if c not in {p.ticker for p in sel.picks}]
            W.iloc[pos:, [W.columns.get_loc(c) for c in others]] = 0.0
        else:
            W.iloc[pos:, :] = 0.0
    # 約定日の比率は翌日の値動きから効く。その日の値動きは前の比率（ドリフトつき）で受ける
    R = rets.loc[days]
    held = W.shift(1).fillna(0.0)
    # ドリフト: 入れ替えの間は比率が値動きで変わる。行列で厳密に追うのは重いので、
    # 入れ替え区間ごとに「初期比率 × 累積リターン」で保有比率を再現する
    eq = np.ones(len(days))
    port_r = np.zeros(len(days))
    turnover_cost = np.zeros(len(days))
    exec_set = set(exec_days)
    cur = np.zeros(W.shape[1])          # 現在の比率（ドリフトつき）
    Rv = R.values
    Wv = W.values
    for i in range(len(days)):
        r = Rv[i]
        pr = float(np.dot(cur, r))
        port_r[i] = pr
        if cur.sum() > 0:
            cur = cur * (1 + r) / (1 + pr)
        if days[i] in exec_set:
            new = Wv[i]
            turnover_cost[i] = cost * float(np.abs(new - cur).sum())
            cur = new.copy()
    growth = (1 + port_r) * (1 - turnover_cost)
    equity = pd.Series(np.cumprod(growth), index=days)
    m = metrics_from_equity(equity)
    m["rebalances"] = float(len(exec_days))
    return {"equity": equity, "metrics": m, "weights": W}


def equal_weight_equity(prices: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    r = prices.ffill().pct_change().loc[days].mean(axis=1).fillna(0.0)
    return (1 + r).cumprod()


def tranche_backtest(prices, bench, cfg, *, sleeves: int = 4, base_offset: int = 0, feats=None, **kw) -> dict:
    """分割入れ替え: 月を sleeves 等分し、各列を違う週に入れ替える。評価額は各列（同額）の平均。"""
    if feats is None:
        feats = compute_features(prices, bench, cfg)
    step = 20 // sleeves
    curves = []
    for k in range(sleeves):
        r = fast_backtest(prices, bench, cfg, feats=feats, rebalance_offset=base_offset + k * step, **kw)
        curves.append(r["equity"])
    eq = pd.concat(curves, axis=1).mean(axis=1)
    return {"equity": eq, "metrics": metrics_from_equity(eq), "sleeves": curves}
