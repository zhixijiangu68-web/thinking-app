"""ルールを過去のデータで検証する（月初に入れ替え、毎日退出を判定、終値で売買、取引コストあり）。

先読みはしない: ある日の終値までのデータで判断し、翌営業日以降の値動きに反映する。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, compute_features, exit_signals, select_at


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
    key = index.to_period("W") if freq == "W" else index.to_period("M")
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
    sectors: dict[str, str] | None = None,
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
    entry: dict[str, float] = {}     # ticker -> 取得値
    eq, peak = 1.0, 1.0
    halted = False
    equity = pd.Series(np.nan, index=days)
    trades: list[dict] = []
    n_exit_trend = n_exit_stop = n_dd_stop = 0

    def sell(t: str, d: pd.Timestamp, why: str) -> None:
        nonlocal eq
        eq *= 1 - cost * weights[t]
        trades.append({"date": d, "ticker": t, "from": weights[t], "to": 0.0, "why": why})
        del weights[t]
        entry.pop(t, None)

    for d in days:
        # 1) 前日までの保有で今日の値動きを受ける
        if weights:
            day_r = {t: float(rets.at[d, t]) if not np.isnan(rets.at[d, t]) else 0.0 for t in weights}
            port_r = sum(w * day_r[t] for t, w in weights.items())
            eq *= 1 + port_r
            weights = {t: w * (1 + day_r[t]) / (1 + port_r) for t, w in weights.items()}

        # 2) 毎日の退出判定（今日の終値で売る）
        if weights:
            for sig in exit_signals(feats, prices, d, cfg, {t: entry.get(t, 0.0) for t in weights}):
                sell(sig.ticker, d, sig.reason)
                if sig.reason == "trend":
                    n_exit_trend += 1
                else:
                    n_exit_stop += 1

        # 3) 口座の遮断器
        peak = max(peak, eq)
        if cfg.portfolio_dd_stop and weights and eq / peak - 1 <= -cfg.portfolio_dd_stop:
            for t in list(weights):
                sell(t, d, "dd_stop")
            n_dd_stop += 1
            halted = True

        # 4) 入れ替え日なら、今日の終値までの情報で判断し、終値で売買
        if d in rb:
            if halted:
                halted = False  # 実運用では人が理由を見て再開する。検証では次の月初に再開したとみなす
                peak = eq
            sel = select_at(feats, prices, d, cfg, sectors)
            new = {p.ticker: 1.0 / len(sel.picks) for p in sel.picks} if sel.invested else {}
            tickers = set(weights) | set(new)
            turnover = sum(abs(new.get(t, 0.0) - weights.get(t, 0.0)) for t in tickers)
            eq *= 1 - cost * turnover
            for t in tickers:
                if abs(new.get(t, 0.0) - weights.get(t, 0.0)) > 1e-9:
                    trades.append({"date": d, "ticker": t, "from": weights.get(t, 0.0), "to": new.get(t, 0.0), "why": "rebalance"})
                if t in new and t not in weights:
                    entry[t] = float(prices.at[d, t])
            for t in list(entry):
                if t not in new:
                    del entry[t]
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
    m.update({
        "rebalances": float(len(rb)), "trades": float(len(trades)),
        "exit_trend": float(n_exit_trend), "exit_stop": float(n_exit_stop), "dd_stops": float(n_dd_stop),
    })
    return BacktestResult(
        equity=equity, bench_equity=bench_eq,
        trades=pd.DataFrame(trades, columns=["date", "ticker", "from", "to", "why"]),
        metrics=m, bench_metrics=bench_m,
    )


def sweep_variants(base: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
    """過学習の確認用: 基準から1つずつ値を変えた設定。結果が値で大きく変わるなら疑う。"""
    v: list[tuple[str, StrategyConfig]] = [("基準", base)]
    v += [(f"上位{n}", replace(base, top_n=n)) for n in (3, 8) if n != base.top_n]
    v += [("6-1モメンタム", replace(base, mom_lookback=126))]
    v += [(f"トレンド{w}日", replace(base, trend_window=w, regime_window=w)) for w in (150, 250)]
    v += [("地合いなし", replace(base, use_regime=False))]
    v += [("業種の上限なし", replace(base, max_per_sector=0))]
    v += [("退出ルールなし", replace(base, exit_below_trend=False, stop_loss=0.0))]
    v += [("遮断器なし", replace(base, portfolio_dd_stop=0.0))]
    return v
