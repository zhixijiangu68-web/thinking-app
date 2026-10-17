"""ルールを過去のデータで検証する。

- 月初に入れ替え、毎日退出を判定、終値で売買、取引コストあり
- 先読みはしない: ある日の終値までのデータで判断する
- 約定は翌営業日の終値（execution_lag=1）。実際には判断した日の引け後に注文し、翌日に約定するため。
  0 にすると判断した日の終値で約定（楽観的）
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, compute_features, exit_signals, select_at, target_weights


@dataclass
class BacktestResult:
    equity: pd.Series
    bench_equity: pd.Series | None
    trades: pd.DataFrame
    metrics: dict[str, float]
    bench_metrics: dict[str, float] | None
    yearly: pd.DataFrame | None = None   # 年ごとのリターン（strategy / benchmark / universe）
    universe_equity: pd.Series | None = None   # 候補の全銘柄を均等に買い持ち（同じ銘柄・同じ配当込みの比較）
    universe_metrics: dict[str, float] | None = None


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
    monthly = eq.resample("ME").last().pct_change().dropna() if len(eq) > 40 else pd.Series(dtype=float)
    return {
        "final_multiple": float(eq.iloc[-1] / eq.iloc[0]),
        "cagr": float(cagr),
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "years": float(years),
    }


def yearly_table(equity: pd.Series, bench_eq: pd.Series | None, uni_eq: pd.Series | None = None) -> pd.DataFrame:
    def yr(s: pd.Series) -> pd.Series:
        e = s.dropna()
        first = e.groupby(e.index.year).first()
        last = e.groupby(e.index.year).last()
        prev_last = last.shift(1)
        base = prev_last.fillna(first)
        return last / base - 1
    out = pd.DataFrame({"strategy": yr(equity)})
    if bench_eq is not None:
        out["benchmark"] = yr(bench_eq)
    if uni_eq is not None:
        out["universe"] = yr(uni_eq)
    return out


def equal_weight_equity(prices: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """候補の全銘柄を毎日均等に持ったときの評価額（株価のない銘柄はその日は除く）。"""
    r = prices.pct_change().loc[days].mean(axis=1).fillna(0.0)
    return (1 + r).cumprod()


def _planned_to_weights(plan: dict, weights: dict[str, float], filled: pd.DataFrame, d: pd.Timestamp, eq: float) -> dict[str, float]:
    """判断した日の計画（株数に相当する比率）を、約定日の比率に直す。紙上の口座と同じ扱い:
    - 残す銘柄は約定日の比率のまま触らない
    - 買う銘柄は、判断した日の株価で決めた株数を約定日の株価で評価した比率（現金が足りなければ縮める）"""
    new = {t: weights[t] for t in plan["kept"] if t in weights}
    rest = {}
    for t, w in plan["rest"].items():
        p0, p1 = plan["px"].get(t), float(filled.at[d, t])
        if p0 and p0 > 0 and not np.isnan(p1):
            rest[t] = w * (p1 / p0) * (plan["eq"] / eq)
        else:
            rest[t] = w
    room = max(0.0, 1.0 - sum(new.values()))
    total = sum(rest.values())
    if total > room + 1e-12 and total > 0:
        rest = {t: w * room / total for t, w in rest.items()}
    new.update(rest)
    return new


def run_backtest(
    prices: pd.DataFrame,
    bench: pd.Series | None,
    cfg: StrategyConfig,
    start: str | None = None,
    end: str | None = None,
    cost: float = 0.001,
    freq: str = "M",
    sectors: dict[str, str] | None = None,
    execution_lag: int = 1,
) -> BacktestResult:
    prices = prices.sort_index()
    feats = compute_features(prices, bench, cfg)
    filled = prices.ffill()  # 株価がとれない日は最後の値で評価する
    rets = filled.pct_change()  # pandas 3 の pct_change は欠損を埋めないので、埋めた株価から計算する

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
    counts = {"trend": 0, "stop": 0, "missing": 0, "dd_stop": 0}   # 約定した退出の回数（銘柄ごと）
    days_in_market = 0
    # 約定待ち: (実行日, 種類, 中身)
    pending: list[tuple[int, str, object]] = []

    def sell(t: str, d: pd.Timestamp, why: str) -> None:
        nonlocal eq
        eq *= 1 - cost * weights[t]
        trades.append({"date": d, "ticker": t, "from": weights[t], "to": 0.0, "why": why})
        if why in counts:
            counts[why] += 1
        del weights[t]
        entry.pop(t, None)

    def execute_target(new: dict[str, float], d: pd.Timestamp, why: str) -> None:
        nonlocal eq, weights
        tickers = set(weights) | set(new)
        turnover = sum(abs(new.get(t, 0.0) - weights.get(t, 0.0)) for t in tickers)
        eq *= 1 - cost * turnover
        for t in tickers:
            if abs(new.get(t, 0.0) - weights.get(t, 0.0)) > 1e-6:
                trades.append({"date": d, "ticker": t, "from": weights.get(t, 0.0), "to": new.get(t, 0.0), "why": why})
            if t in new and t not in weights:
                entry[t] = float(filled.at[d, t])
            elif t in new and new[t] > weights[t] + 1e-9:
                # 買い増し: 取得値は比率で加重した平均（紙上の口座の平均取得単価に近づける）
                p_now = float(filled.at[d, t])
                entry[t] = (weights[t] * entry.get(t, p_now) + (new[t] - weights[t]) * p_now) / new[t]
        for t in list(entry):
            if t not in new:
                del entry[t]
        weights = new

    for i, d in enumerate(days):
        # 1) 前日までの保有で今日の値動きを受ける
        if weights:
            day_r = {t: float(rets.at[d, t]) if not np.isnan(rets.at[d, t]) else 0.0 for t in weights}
            port_r = sum(w * day_r[t] for t, w in weights.items())
            eq *= 1 + port_r
            weights = {t: w * (1 + day_r[t]) / (1 + port_r) for t, w in weights.items()}
            days_in_market += 1

        # 2) 前に出した注文を今日の終値で約定させる
        due = [p for p in pending if p[0] <= i]
        pending = [p for p in pending if p[0] > i]
        for _, kind, payload in due:
            if kind == "exit":
                t, why = payload
                if t in weights:
                    sell(t, d, why)
            elif kind == "target":
                execute_target(_planned_to_weights(payload, weights, filled, d, eq), d, "rebalance")

        # 3) 今日の終値で判断する → 注文（lag 日後に約定）
        todays: list[tuple[int, str, object]] = []
        if weights:
            already = {p[2][0] for p in pending if p[1] == "exit"}
            for sig in exit_signals(feats, prices, d, cfg, {t: entry.get(t, 0.0) for t in weights}):
                if sig.ticker not in already:
                    todays.append((i + execution_lag, "exit", (sig.ticker, sig.reason)))

        peak = max(peak, eq)
        fired_today = False
        if cfg.portfolio_dd_stop and weights and eq / peak - 1 <= -cfg.portfolio_dd_stop and not halted:
            todays = [(i + execution_lag, "exit", (t, "dd_stop")) for t in weights]
            halted = True
            fired_today = True

        if d in rb and not fired_today:
            if halted:
                halted = False  # 実運用では人が理由を見て再開する。検証では次の月初に再開したとみなす
                peak = eq
            sel = select_at(feats, prices, d, cfg, sectors)
            targets = [p.ticker for p in sel.picks] if sel.invested else []
            # 退出の予定と入れ替えが同じ日なら、入れ替えの目標が優先（退出銘柄は目標に入らない）
            todays = [p for p in todays if p[1] != "exit" or p[2][0] not in targets]
            # 比率は判断した日の保有比率から決める（紙上の口座が判断した日の株価で株数を決めるのと同じ）
            exiting = {p[2][0] for p in pending + todays if p[1] == "exit"}
            current = {t: w for t, w in weights.items() if t not in exiting}
            planned = target_weights(current, targets, cfg.rebalance_band)
            kept = [t for t in targets if t in current and planned.get(t) == current[t]]
            rest = {t: w for t, w in planned.items() if t not in kept}
            plan = {"kept": kept, "rest": rest, "eq": eq, "px": {t: float(filled.at[d, t]) for t in rest}}
            todays.append((i + execution_lag, "target", plan))

        if execution_lag == 0:
            for _, kind, payload in todays:
                if kind == "exit":
                    t, why = payload
                    if t in weights:
                        sell(t, d, why)
                else:
                    execute_target(_planned_to_weights(payload, weights, filled, d, eq), d, "rebalance")
        else:
            pending.extend(todays)
        equity.at[d] = eq

    bench_eq = None
    bench_m = None
    if bench is not None:
        b = bench.reindex(days).ffill().dropna()
        if len(b) > 1:
            bench_eq = b / b.iloc[0]
            bench_m = metrics_from_equity(bench_eq)

    uni_eq = equal_weight_equity(prices, days)
    uni_m = metrics_from_equity(uni_eq)

    trades_df = pd.DataFrame(trades, columns=["date", "ticker", "from", "to", "why"])
    m = metrics_from_equity(equity)
    m.update({
        "rebalances": float(len(rb)), "trades": float(len(trades)),
        "exit_trend": float(counts["trend"]), "exit_stop": float(counts["stop"]),
        "exit_missing": float(counts["missing"]),
        "dd_stops": float(len({r["date"] for r in trades if r["why"] == "dd_stop"})),
        "time_in_market": days_in_market / len(days),
    })
    m.update(exit_quality(trades_df, filled))
    return BacktestResult(
        equity=equity, bench_equity=bench_eq, trades=trades_df,
        metrics=m, bench_metrics=bench_m, yearly=yearly_table(equity, bench_eq, uni_eq),
        universe_equity=uni_eq, universe_metrics=uni_m,
    )


def exit_quality(trades: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21) -> dict[str, float]:
    """退出して正解だった割合: 売った後 horizon 営業日でその銘柄が下がっていれば正解。"""
    ex = trades[trades["why"].isin(["trend", "stop", "dd_stop"])]
    if ex.empty:
        return {"exits_judged": 0.0, "exit_correct_rate": np.nan, "exit_avg_fwd_return": np.nan}
    fwd = []
    for r in ex.itertuples():
        pos = prices.index.get_loc(r.date)
        if pos + horizon >= len(prices):
            continue
        p0, p1 = prices.iat[pos, prices.columns.get_loc(r.ticker)], prices.iat[pos + horizon, prices.columns.get_loc(r.ticker)]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            fwd.append(float(p1 / p0 - 1))
    if not fwd:
        return {"exits_judged": 0.0, "exit_correct_rate": np.nan, "exit_avg_fwd_return": np.nan}
    a = np.array(fwd)
    return {"exits_judged": float(len(a)), "exit_correct_rate": float((a < 0).mean()), "exit_avg_fwd_return": float(a.mean())}


def sweep_variants(base: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
    """過学習の確認用: 基準から1つずつ値を変えた設定。結果が値で大きく変わるなら疑う。"""
    v: list[tuple[str, StrategyConfig]] = [("基準", base)]
    v += [(f"上位{n}", replace(base, top_n=n)) for n in (5, 10, 12) if n != base.top_n]
    v += [("6-1モメンタム", replace(base, mom_lookback=126))]
    v += [(f"トレンド{w}日", replace(base, trend_window=w, regime_window=w)) for w in (150, 250)]
    v += [("業種の上限なし", replace(base, max_per_sector=0))]
    v += [("毎回均等に戻す", replace(base, rebalance_band=0.0))]
    v += [("地合いの判定をつける", replace(base, use_regime=True))]
    v += [("退出ルールをつける（200日線割れ・−20%）", replace(base, exit_below_trend=True, stop_loss=0.20))]
    v += [("遮断器をつける（−20%）", replace(base, portfolio_dd_stop=0.20))]
    v += [("守りを全部つける（上位5・旧ルール）",
           replace(base, top_n=5, use_regime=True, exit_below_trend=True, stop_loss=0.20, portfolio_dd_stop=0.20))]
    return v


def sweep_costs() -> list[tuple[str, float]]:
    return [("コスト0.1%", 0.001), ("コスト0.3%", 0.003), ("コスト0.5%", 0.005)]
