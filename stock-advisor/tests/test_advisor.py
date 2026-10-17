import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.backtest import run_backtest, rebalance_dates
from advisor.data import synthetic_prices
from advisor.portfolio import Order, Portfolio, plan_orders
from advisor.strategy import StrategyConfig, compute_features, select_at
from advisor.universe import load_universe

TICKERS = [f"{1000 + i}.T" for i in range(12)]


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(TICKERS, start="2018-01-01", end="2022-12-30", seed=1)


@pytest.fixture(scope="module")
def bench():
    return synthetic_prices(["^B"], start="2018-01-01", end="2022-12-30", seed=2)["^B"]


def test_universe_loads_default():
    u = load_universe()
    assert "7203.T" in u and u["7203.T"].startswith("トヨタ")
    assert all(t.endswith(".T") for t in u)


def test_select_picks_top_n_and_only_eligible(prices, bench):
    cfg = StrategyConfig(top_n=3)
    feats = compute_features(prices, bench, cfg)
    sel = select_at(feats, prices, prices.index[-1], cfg)
    assert len(sel.picks) <= 3
    if sel.invested:
        scores = [p.score for p in sel.picks]
        assert scores == sorted(scores, reverse=True)
        for p in sel.picks:
            assert p.momentum > 0 and p.vol <= cfg.max_vol
            assert bool(feats.trend_ok.loc[prices.index[-1], p.ticker])


def test_no_lookahead(prices, bench):
    """ある日の判断は、その日より後の株価を変えても変わらない。"""
    cfg = StrategyConfig(top_n=3)
    asof = prices.index[-300]
    base = select_at(compute_features(prices, bench, cfg), prices, asof, cfg)
    tweaked = prices.copy()
    tweaked.loc[tweaked.index > asof] *= 3.0
    after = select_at(compute_features(tweaked, bench, cfg), tweaked, asof, cfg)
    assert [p.ticker for p in base.picks] == [p.ticker for p in after.picks]
    assert base.invested == after.invested


def test_regime_off_means_cash(prices):
    cfg = StrategyConfig(top_n=3)
    falling = pd.Series(np.linspace(100, 10, len(prices)), index=prices.index)  # ずっと下げ相場
    sel = select_at(compute_features(prices, falling, cfg), prices, prices.index[-1], cfg)
    assert not sel.invested and sel.picks == []


def test_rebalance_dates_first_bday_of_month(prices):
    d = rebalance_dates(prices.index, "M")
    assert all(x.day <= 5 for x in d)
    assert len({(x.year, x.month) for x in d}) == len(d)


def test_backtest_runs_and_metrics_finite(prices, bench):
    res = run_backtest(prices, bench, StrategyConfig(top_n=3), cost=0.001)
    assert res.equity.notna().all() and (res.equity > 0).all()
    for k in ("cagr", "annual_vol", "max_drawdown", "final_multiple"):
        assert np.isfinite(res.metrics[k])
    assert -1 <= res.metrics["max_drawdown"] <= 0
    assert res.bench_metrics is not None


def test_backtest_cost_reduces_return(prices, bench):
    cfg = StrategyConfig(top_n=3)
    a = run_backtest(prices, bench, cfg, cost=0.0).metrics["final_multiple"]
    b = run_backtest(prices, bench, cfg, cost=0.01).metrics["final_multiple"]
    assert b < a


def test_plan_orders_respects_lots_and_cash():
    pf = Portfolio(cash=1_000_000)
    px = {"A": 1000.0, "B": 2500.0, "C": 50_000.0}
    orders, skipped = plan_orders(pf, ["A", "B", "C"], px, lot=100)
    assert skipped == ["C"]  # 1単元 500万円で買えない
    assert all(o.shares % 100 == 0 and o.side == "buy" for o in orders)
    assert sum(o.value for o in orders) <= pf.cash
    assert {o.ticker for o in orders} == {"A", "B"}


def test_plan_orders_sells_dropped_and_nets_same_ticker():
    pf = Portfolio(cash=0, positions={"A": 300, "B": 200}, avg_cost={"A": 900.0, "B": 2000.0})
    px = {"A": 1000.0, "B": 2500.0, "C": 1200.0}
    orders, _ = plan_orders(pf, ["A", "C"], px, lot=100)
    by = {o.ticker: o for o in orders}
    assert by["B"].side == "sell" and by["B"].shares == 200
    assert by["C"].side == "buy"
    assert len([o for o in orders if o.ticker == "A"]) <= 1  # 同じ銘柄に売りと買いを両方出さない
    sells = sum(o.value for o in orders if o.side == "sell")
    buys = sum(o.value for o in orders if o.side == "buy")
    assert buys <= sells + 1e-6


def test_portfolio_apply_and_roundtrip(tmp_path):
    pf = Portfolio(cash=100_000)
    pf.apply(Order("A", "buy", 100, 500.0))
    assert pf.cash == 50_000 and pf.positions["A"] == 100
    with pytest.raises(ValueError):
        pf.apply(Order("A", "buy", 1000, 500.0))  # 現金不足
    with pytest.raises(ValueError):
        pf.apply(Order("A", "sell", 200, 500.0))  # 保有以上
    pf.apply(Order("A", "sell", 100, 600.0))
    assert pf.cash == 110_000 and "A" not in pf.positions
    f = tmp_path / "pf.json"
    pf.save(f)
    back = Portfolio.load(f)
    assert back.cash == pf.cash and len(back.history) == 2
    assert json.loads(f.read_text())["positions"] == {}


# ---------- 設計書（docs/design.md）のルール ----------
from advisor.strategy import exit_signals
from advisor.universe import load_sectors


def test_universe_has_sectors():
    s = load_sectors()
    assert s["7203.T"] == "自動車" and "不明" not in s.values()


def test_sector_cap(prices, bench):
    cfg = StrategyConfig(top_n=5, max_per_sector=2)
    feats = compute_features(prices, bench, cfg)
    sectors = {t: ("A" if i < 8 else "B") for i, t in enumerate(TICKERS)}
    sel = select_at(feats, prices, prices.index[-1], cfg, sectors)
    from collections import Counter
    assert all(n <= 2 for n in Counter(p.sector for p in sel.picks).values())
    nocap = select_at(feats, prices, prices.index[-1], StrategyConfig(top_n=5, max_per_sector=0), sectors)
    assert len(nocap.picks) >= len(sel.picks)


def test_exit_signals_trend_and_stop(prices, bench):
    cfg = StrategyConfig(stop_loss=0.20)
    feats = compute_features(prices, bench, cfg)
    d = prices.index[-1]
    t_ok = next(t for t in TICKERS if bool(feats.trend_ok.at[d, t]))
    t_bad = next((t for t in TICKERS if not bool(feats.trend_ok.at[d, t])), None)
    px = float(prices.at[d, t_ok])
    # 200日線の上でも、取得値から −20% なら売る
    sigs = exit_signals(feats, prices, d, cfg, {t_ok: px * 1.5})
    assert [s.reason for s in sigs] == ["stop"]
    # 含み益なら売らない
    assert exit_signals(feats, prices, d, cfg, {t_ok: px * 0.9}) == []
    if t_bad:
        assert [s.reason for s in exit_signals(feats, prices, d, cfg, {t_bad: 1.0})] == ["trend"]


def test_backtest_circuit_breaker_fires_on_crash():
    """全銘柄が一斉に −40% になる相場では、遮断器が動いて下落が −20% 付近で止まる。"""
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    n = len(idx)
    up = np.linspace(100, 200, n)
    px = pd.DataFrame({t: up * (1 + 0.01 * i) for i, t in enumerate(TICKERS)}, index=idx)
    crash_at = idx[-120]
    px.loc[idx >= crash_at] *= 0.6                      # 一日で −40%
    bench = pd.Series(np.linspace(100, 200, n), index=idx)  # 地合いは良いまま
    cfg = StrategyConfig(top_n=3, exit_below_trend=False, stop_loss=0.0, portfolio_dd_stop=0.20, max_per_sector=0)
    res = run_backtest(px, bench, cfg, cost=0.0)
    assert res.metrics["dd_stops"] >= 1
    after = res.equity.loc[crash_at:]
    assert after.iloc[0] / res.equity.loc[:crash_at].max() - 1 <= -0.20   # 暴落の日に −20% を超えた
    assert after.iloc[1:].nunique() == 1                                   # 全部売ったので、その後は横ばい
    no_breaker = run_backtest(px, bench, StrategyConfig(top_n=3, exit_below_trend=False, stop_loss=0.0,
                                                        portfolio_dd_stop=0.0, max_per_sector=0), cost=0.0)
    assert no_breaker.metrics["dd_stops"] == 0


def test_backtest_stop_loss_counts(prices, bench):
    cfg = StrategyConfig(top_n=3, stop_loss=0.05)   # きつい保険なら、何回かは動く
    res = run_backtest(prices, bench, cfg, cost=0.001)
    assert res.metrics["exit_stop"] + res.metrics["exit_trend"] > 0
    assert set(res.trades["why"]) <= {"rebalance", "trend", "stop", "dd_stop"}


def test_portfolio_new_fields_roundtrip(tmp_path):
    pf = Portfolio(cash=1.0, halted=True, halted_reason="x", last_rebalance="2026-10-01", peak_equity=2.0)
    f = tmp_path / "pf.json"; pf.save(f)
    back = Portfolio.load(f)
    assert back.halted and back.halted_reason == "x" and back.last_rebalance == "2026-10-01" and back.peak_equity == 2.0


def test_portfolio_old_file_without_new_fields(tmp_path):
    f = tmp_path / "pf.json"
    f.write_text(json.dumps({"cash": 5.0, "positions": {}, "avg_cost": {}, "history": [], "created": "x"}))
    back = Portfolio.load(f)
    assert not back.halted and back.last_rebalance == "" and back.peak_equity == 0.0
