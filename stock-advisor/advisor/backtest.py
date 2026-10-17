"""ルールを過去のデータで検証する（月初に入れ替え、終値で売買、取引コストあり）。

先読みはしない: ある日の終値までのデータで判断し、翌営業日以降の値動きに反映する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, compute_features, select_at


@dataclass
class BacktestResult:
    equity: pd.Series
    bench_equity: pd.Series | None
    trades: pd.DataFrame
    metrics: dict[str, float]
    bench_metrics: dict[str, float] | None


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> list[pd.Timestamp]:
    """各期間の最初の営業日。freq は 'M'（月）か 'W'（週）。"""
    s = pd.Series(index, index=index)
    if freq == "W":
        key = index.to_period("W")
    else:
        key = index.to_period("M")
    return list(s.groupby(key).first())


def metrics_from_equity(eq: pd.Series) -> dict[str, float]:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    return {
        "final_multiple": float(eq.iloc[-1] / eq.iloc[0]),
        "cagr": float(cagr),
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "years": float(years),
    }


def run_backtest(
    prices: pd.DataFrame,
    bench: pd.Series | None,
    cfg: StrategyConfig,
    start: str | None = None,
    end: str | None = None,
    cost: float = 0.001,
    freq: str = "M",
) -> BacktestResult:
    prices = prices.sort_index()
    feats = compute_features(prices, bench, cfg)
    rets = prices.pct_change()

    idx = prices.index
    first_ok = idx[min(cfg.min_history, len(idx) - 1)]
    s = max(pd.Timestamp(start), first_ok) if start else first_ok
    e = pd.Timestamp(end) if end else idx[-1]
    days = idx[(idx >= s) & (idx <= e)]
    if len(days) < 2:
        raise ValueError("検証できる期間が短すぎます。データの開始日を早くするか、期間を広げてください。")

    rb = set(rebalance_dates(days, freq))
    weights: dict[str, float] = {}
    eq = 1.0
    equity = pd.Series(np.nan, index=days)
    trades = []

    for d in days:
        # 1) 前日までの保有で今日の値動きを受ける
        if weights:
            day_r = {t: float(rets.at[d, t]) if not np.isnan(rets.at[d, t]) else 0.0 for t in weights}
            port_r = sum(w * day_r[t] for t, w in weights.items())
            eq *= 1 + port_r
            # 保有比率のドリフト
            weights = {t: w * (1 + day_r[t]) / (1 + port_r) for t, w in weights.items()}
        # 2) 入れ替え日なら、今日の終値までの情報で判断し、終値で売買
        if d in rb:
            sel = select_at(feats, prices, d, cfg)
            new = {p.ticker: 1.0 / len(sel.picks) for p in sel.picks} if sel.invested else {}
            tickers = set(weights) | set(new)
            turnover = sum(abs(new.get(t, 0.0) - weights.get(t, 0.0)) for t in tickers)
            eq *= 1 - cost * turnover
            for t in tickers:
                if abs(new.get(t, 0.0) - weights.get(t, 0.0)) > 1e-9:
                    trades.append({"date": d, "ticker": t, "from": weights.get(t, 0.0), "to": new.get(t, 0.0)})
            weights = new
        equity.at[d] = eq

    bench_eq = None
    bench_m = None
    if bench is not None:
        b = bench.reindex(days).ffill().dropna()
        if len(b) > 1:
            bench_eq = b / b.iloc[0]
            bench_m = metrics_from_equity(bench_eq)

    m = metrics_from_equity(equity)
    m["rebalances"] = float(len(rb))
    m["trades"] = float(len(trades))
    return BacktestResult(
        equity=equity, bench_equity=bench_eq,
        trades=pd.DataFrame(trades, columns=["date", "ticker", "from", "to"]),
        metrics=m, bench_metrics=bench_m,
    )
