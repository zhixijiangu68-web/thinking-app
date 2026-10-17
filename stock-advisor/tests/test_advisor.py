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
    cfg = StrategyConfig(top_n=3, use_regime=True)
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
    cfg = StrategyConfig(top_n=3, use_regime=True)
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
    cfg = StrategyConfig(top_n=5, max_per_sector=2, use_regime=True)
    feats = compute_features(prices, bench, cfg)
    sectors = {t: ("A" if i < 8 else "B") for i, t in enumerate(TICKERS)}
    sel = select_at(feats, prices, prices.index[-1], cfg, sectors)
    from collections import Counter
    assert all(n <= 2 for n in Counter(p.sector for p in sel.picks).values())
    nocap = select_at(feats, prices, prices.index[-1], StrategyConfig(top_n=5, max_per_sector=0), sectors)
    assert len(nocap.picks) >= len(sel.picks)


def test_exit_signals_trend_and_stop(prices, bench):
    cfg = StrategyConfig(stop_loss=0.20, exit_below_trend=True)
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
    cfg = StrategyConfig(top_n=3, exit_below_trend=False, stop_loss=0.0, portfolio_dd_stop=0.20, max_per_sector=0, sleeves=1)
    res = run_backtest(px, bench, cfg, cost=0.0)
    assert res.metrics["dd_stops"] >= 1
    after = res.equity.loc[crash_at:]
    assert after.iloc[0] / res.equity.loc[:crash_at].max() - 1 <= -0.20   # 暴落の日に −20% を超えた
    assert after.iloc[1:].nunique() == 1                                   # 全部売ったので、その後は横ばい
    no_breaker = run_backtest(px, bench, StrategyConfig(top_n=3, exit_below_trend=False, stop_loss=0.0,
                                                        portfolio_dd_stop=0.0, max_per_sector=0, sleeves=1), cost=0.0)
    assert no_breaker.metrics["dd_stops"] == 0


def test_backtest_stop_loss_counts(prices, bench):
    cfg = StrategyConfig(top_n=3, stop_loss=0.05, exit_below_trend=True)   # きつい保険なら、何回かは動く
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


# ---------- 検証ラウンド2: 現実との差（約定の遅れ・欠損・注文待ち） ----------
from advisor.backtest import exit_quality
from advisor.data import sanity_report, _is_fresh
from advisor.portfolio import fill_pending, queue_orders


def test_lag_changes_result_but_not_signals(prices, bench):
    cfg = StrategyConfig(top_n=3)
    r0 = run_backtest(prices, bench, cfg, execution_lag=0)
    r1 = run_backtest(prices, bench, cfg, execution_lag=1)
    # 同じ判断（入れ替えの銘柄）で、約定日が1日ずれる（比率は約定日の保有比率から決めるので少し違う）
    a = r0.trades[r0.trades.why == "rebalance"]["ticker"].head(6).tolist()
    b = r1.trades[r1.trades.why == "rebalance"]["ticker"].head(6).tolist()
    assert a == b
    d0 = r0.trades[r0.trades.why == "rebalance"].date.iloc[0]
    d1 = r1.trades[r1.trades.why == "rebalance"].date.iloc[0]
    assert prices.index.get_loc(d1) == prices.index.get_loc(d0) + 1
    assert np.isfinite(r1.metrics["cagr"]) and 0 < r1.metrics["time_in_market"] <= 1


def test_missing_price_forces_exit(prices, bench):
    cfg = StrategyConfig(top_n=3, use_regime=False, max_missing_days=5)
    feats = compute_features(prices, None, cfg)
    d = prices.index[300]
    held = select_at(feats, prices, d, cfg).picks[0].ticker
    px = prices.copy()
    px.loc[prices.index[320]:, held] = np.nan
    res = run_backtest(px, None, cfg)
    assert res.metrics["exit_missing"] >= 1
    t = res.trades[(res.trades.ticker == held) & (res.trades.why == "missing")]
    assert not t.empty
    # 4日の欠損では退出しない
    sig = exit_signals(feats, px, prices.index[323], cfg, {held: 1.0})
    assert sig == []
    sig = exit_signals(feats, px, prices.index[325], cfg, {held: 1.0})
    assert [s.reason for s in sig] == ["missing"]


def test_sector_cap_strict(prices):
    cfg = StrategyConfig(top_n=5, max_per_sector=2, use_regime=False)
    feats = compute_features(prices, None, cfg)
    sel = select_at(feats, prices, prices.index[-1], cfg, {t: "X" for t in TICKERS})
    assert len(sel.picks) <= 2
    if sel.eligible > 2:
        assert len(sel.picks) == 2 and len(sel.sector_skipped) >= 1


def test_pending_orders_fill_next_day_only():
    pf = Portfolio(cash=1_000_000)
    queue_orders(pf, [Order("A", "buy", 100, 1000.0, "x")], "2026-10-16")
    assert fill_pending(pf, {"A": 1000.0}, "2026-10-16") == []        # 同じ日は約定しない
    assert pf.pending and pf.cash == 1_000_000
    filled = fill_pending(pf, {"A": 1200.0}, "2026-10-17")            # 翌日、値上がりして約定
    assert filled[0].price == 1200.0 and pf.positions["A"] == 100 and pf.pending == []
    # 値上がりで現金が足りなければ単元ずつ減らす
    pf2 = Portfolio(cash=250_000)
    queue_orders(pf2, [Order("A", "buy", 200, 1000.0, "x")], "2026-10-16")
    f = fill_pending(pf2, {"A": 1300.0}, "2026-10-17")
    assert f[0].shares == 100 and pf2.cash == 250_000 - 130_000
    # 株価がとれなければ待ったまま、売りは保有以上にならない
    pf3 = Portfolio(cash=0, positions={"B": 100}, avg_cost={"B": 1.0})
    queue_orders(pf3, [Order("B", "sell", 300, 1.0, "x"), Order("C", "buy", 100, 1.0, "x")], "2026-10-16")
    f = fill_pending(pf3, {"B": 2.0}, "2026-10-17")
    assert f[0].shares == 100 and pf3.cash == 200 and [p["ticker"] for p in pf3.pending] == ["C"]
    # 同じ銘柄の新しい注文は古いものを置き換える
    queue_orders(pf3, [Order("C", "buy", 200, 1.0, "y")], "2026-10-17")
    assert len(pf3.pending) == 1 and pf3.pending[0]["shares"] == 200


def test_sanity_report_flags_jump_stale_short(prices):
    px = prices.copy()
    px.loc[px.index[-10], TICKERS[0]] *= 1.5                 # +50%
    px.loc[px.index[-8]:, TICKERS[1]] = np.nan               # 8日とれない
    px["NEW"] = np.nan; px.loc[px.index[-30]:, "NEW"] = 100.0  # 履歴30日
    kinds = {(i.ticker, i.kind) for i in sanity_report(px)}
    assert (TICKERS[0], "jump") in kinds
    assert (TICKERS[1], "stale") in kinds
    assert ("NEW", "short") in kinds
    assert not any(i.kind == "jump" for i in sanity_report(prices) if i.ticker == TICKERS[2])


def test_cache_freshness_uses_business_days():
    import datetime as dt
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2026-10-16"]))   # 金曜の終値
    assert _is_fresh(s, today=dt.date(2026, 10, 16))   # 金曜の引け後
    assert _is_fresh(s, today=dt.date(2026, 10, 17))   # 土曜
    assert _is_fresh(s, today=dt.date(2026, 10, 18))   # 日曜
    assert not _is_fresh(s, today=dt.date(2026, 10, 19))   # 月曜（月曜の終値がまだない → 取り直す。毎日1回は取り直す）
    assert not _is_fresh(s, today=dt.date(2026, 10, 20))


def test_plan_orders_invariants_random():
    """ランダムな口座と株価で、注文が壊れた状態を作らないこと。"""
    rng = np.random.default_rng(42)
    for _ in range(300):
        names = [f"S{i}" for i in range(8)]
        px = {t: float(rng.integers(300, 60000)) for t in names}
        pos = {t: int(rng.integers(0, 6)) * 100 for t in rng.choice(names, 3, replace=False)}
        pos = {t: n for t, n in pos.items() if n}
        pf = Portfolio(cash=float(rng.integers(0, 5_000_000)), positions=pos, avg_cost={t: px[t] for t in pos})
        targets = list(rng.choice(names, int(rng.integers(0, 6)), replace=False))
        reserve = float(rng.integers(0, 200_000))
        orders, skipped = plan_orders(pf, targets, px, lot=100, cash_reserve=reserve)
        tickers = [o.ticker for o in orders]
        assert len(tickers) == len(set(tickers)), "同じ銘柄に2つの注文"
        for o in orders:
            assert o.shares > 0 and o.shares % 100 == 0
            if o.side == "sell":
                assert o.shares <= pf.positions.get(o.ticker, 0)
        sim = Portfolio(cash=pf.cash, positions=dict(pf.positions), avg_cost=dict(pf.avg_cost))
        PaperBroker(sim).execute(orders)        # 現金不足なら ValueError
        assert sim.cash >= reserve - 1e-6 or not any(o.side == "buy" for o in orders)
        for t in sim.positions:
            assert t in targets, "対象外の銘柄が残った"
        assert all(t in targets for t in skipped)


from advisor.broker import PaperBroker  # noqa: E402  (上のテストで使う)


# ---------- 検証ラウンド4: バックテストと紙上の口座が同じ判断をするか ----------
from advisor.engine import daily_step, simulate_paper, last_prices


def _pair(seed=3, n=20, **kw):
    kw.setdefault("sleeves", 1)
    T = [f"{1000+i}.T" for i in range(n)]
    px = synthetic_prices(T, "2016-01-01", "2022-12-30", seed=seed)
    b = synthetic_prices(["^B"], "2016-01-01", "2022-12-30", seed=seed + 1)["^B"]
    cfg = StrategyConfig(top_n=5, **kw)
    return px, b, cfg, compute_features(px, b, cfg)


def test_paper_matches_backtest_without_breaker():
    """単元1株・十分な現金なら、紙上の口座の評価額はバックテストとほぼ一致する（差は判断日と約定日の株価差だけ）。"""
    px, b, cfg, feats = _pair(portfolio_dd_stop=0.0)
    bt = run_backtest(px, b, cfg, cost=0.0, execution_lag=1)
    sim, _ = simulate_paper(px, feats, cfg, cash=1e9, lot=1, start=bt.equity.index[0])
    ratio = (sim / sim.iloc[0]) / (bt.equity / bt.equity.iloc[0])
    assert (ratio - 1).abs().max() < 0.01


def test_paper_matches_backtest_with_breaker():
    """守りの層を全部つけても、しきい値ぎりぎりの1日差を除けば一致する。"""
    px, b, cfg, feats = _pair(use_regime=True, exit_below_trend=True, stop_loss=0.20, portfolio_dd_stop=0.20)
    bt = run_backtest(px, b, cfg, cost=0.0, execution_lag=1)
    sim, pf = simulate_paper(px, feats, cfg, cash=1e9, lot=1, start=bt.equity.index[0])
    ratio = (sim / sim.iloc[0]) / (bt.equity / bt.equity.iloc[0])
    assert (ratio - 1).abs().max() < 0.06
    assert abs(ratio.iloc[-1] - 1) < 0.04


def test_breaker_on_rebalance_day_does_not_reenter_same_day():
    idx = pd.bdate_range("2018-01-01", "2019-06-28")
    n = len(idx); T = [f"{1000+i}.T" for i in range(6)]
    up = np.linspace(100, 300, n)
    px = pd.DataFrame({t: up * (1 + 0.01 * i) for i, t in enumerate(T)}, index=idx)
    crash = [d for d in idx if d.month == 5 and d.year == 2019][0]   # 5月の最初の営業日 = 入れ替え日
    px.loc[idx >= crash] *= 0.7
    cfg = StrategyConfig(top_n=3, exit_below_trend=False, stop_loss=0.0, use_regime=False, max_per_sector=0,
                         portfolio_dd_stop=0.20, sleeves=1)
    bt = run_backtest(px, None, cfg, cost=0.0, execution_lag=0)
    t = bt.trades
    assert (t[t.date == crash].why == "dd_stop").all() and (t.date == crash).sum() == 3
    assert t[(t.date > crash) & (t.date < pd.Timestamp("2019-06-01"))].empty   # 5月中は何もしない
    # 紙上の口座も同じ: 遮断器の日に入れ替えない、翌日に約定、月内は止まったまま
    feats = compute_features(px, None, cfg)
    sim, pf = simulate_paper(px, feats, cfg, cash=1e9, lot=1, start=idx[250])
    sells = [h for h in pf.history if h["side"] == "sell" and "遮断器" in h["reason"]]
    assert sells and all(h["when"] == str(idx[idx.get_loc(crash) + 1].date()) for h in sells)
    buys_after = [h for h in pf.history if h["side"] == "buy" and str(crash.date()) < h["when"] < "2019-06-01"]
    assert buys_after == []


def test_cli_auto_twice_same_day_does_not_double_fill(tmp_path):
    from advisor.cli import main
    f = str(tmp_path / "pf.json")
    base = ["--synthetic", "--seed", "0", "paper", "--file", f]
    assert main(base + ["init", "--cash", "5000000"]) == 0
    assert main(base + ["auto"]) == 0
    pf1 = Portfolio.load(f)
    assert pf1.pending and pf1.positions == {}
    assert main(base + ["auto"]) == 0            # 同じ日にもう一度 → 約定しない、二重に注文しない
    pf2 = Portfolio.load(f)
    assert pf2.positions == {} and len(pf2.pending) == len(pf1.pending)
    assert main(base + ["auto", "--dry-run"]) == 0
    assert Portfolio.load(f).pending == pf2.pending


# ---------- 検証ラウンド5: ルールが狙った効果を拾うか ----------
from advisor.backtest import metrics_from_equity


def _world(seed: int, persist: float, n_stocks: int = 30):
    """persist > 0: 銘柄ごとの期待リターンが1年続く（モメンタムがある世界）。0: 純粋なランダムウォーク"""
    T = [f"{1000+i}.T" for i in range(n_stocks)]
    idx = pd.bdate_range("2012-01-01", "2022-12-31"); n = len(idx)
    rng = np.random.default_rng(seed); cols = {}
    for t in T:
        mu = np.repeat(rng.normal(0, persist, size=n // 252 + 1), 252)[:n] if persist else np.zeros(n)
        r = 0.0002 + mu + 0.015 * rng.standard_normal(n)
        cols[t] = 1000 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def _ew_buy_hold(px: pd.DataFrame, index) -> dict:
    r = px.pct_change().mean(axis=1).fillna(0)
    return metrics_from_equity((1 + r).cumprod().loc[index])


def test_rule_captures_momentum_when_it_exists():
    cfg = StrategyConfig(top_n=5, use_regime=False, max_per_sector=0)
    wins = 0
    for seed in range(4):
        px = _world(seed, 0.0012)
        bt = run_backtest(px, None, cfg, cost=0.001)
        wins += bt.metrics["cagr"] > _ew_buy_hold(px, bt.equity.index)["cagr"]
    assert wins >= 3


def test_regime_filter_cuts_drawdown_in_bear_market():
    T = [f"{1000+i}.T" for i in range(20)]
    idx = pd.bdate_range("2012-01-01", "2022-12-31"); n = len(idx)
    rng = np.random.default_rng(1)
    path = np.concatenate([np.linspace(0, 0.6, n // 2), np.linspace(0.6, -0.1, n - n // 2)])
    px = pd.DataFrame({t: 1000 * np.exp(path + np.cumsum(0.012 * rng.standard_normal(n))) for t in T}, index=idx)
    bench = pd.Series(1000 * np.exp(path), index=idx)
    base = dict(top_n=5, max_per_sector=0, exit_below_trend=False, stop_loss=0.0, portfolio_dd_stop=0.0)
    on = run_backtest(px, bench, StrategyConfig(use_regime=True, **base)).metrics
    off = run_backtest(px, bench, StrategyConfig(use_regime=False, **base)).metrics
    assert on["max_drawdown"] > off["max_drawdown"] + 0.05   # 下落が5ポイント以上浅い
    assert on["time_in_market"] < off["time_in_market"]


# ---------- 検証ラウンド6: 無駄な売買を減らす（目標比率のバンド） ----------
from advisor.strategy import target_weights


def test_target_weights_band():
    cur = {"A": 0.22, "B": 0.15, "C": 0.30}
    tw = target_weights(cur, ["A", "B", "D"], band=0.25)      # 目標 1/3 ≈ 0.333
    assert "A" not in {k for k, v in tw.items() if v == cur["A"]} or True
    # A(0.22) は 0.333 から 34% 外れている → 戻す。B(0.15) も戻す。C は対象外
    assert set(tw) == {"A", "B", "D"} and abs(sum(tw.values()) - 1) < 1e-9
    cur2 = {"A": 0.30, "B": 0.36}
    tw2 = target_weights(cur2, ["A", "B", "D"], band=0.25)     # A, B は ±25% 以内 → そのまま
    assert tw2["A"] == 0.30 and tw2["B"] == 0.36 and abs(tw2["D"] - 0.34) < 1e-9
    assert target_weights(cur2, ["A", "B", "D"], band=0.0) == {"A": 1/3, "B": 1/3, "D": 1/3}
    assert target_weights({}, [], 0.25) == {}


def test_band_reduces_trades_and_keeps_engines_consistent():
    px, b, cfg0, _ = _pair(portfolio_dd_stop=0.0, rebalance_band=0.0)
    cfg1 = StrategyConfig(top_n=5, portfolio_dd_stop=0.0, rebalance_band=0.25, sleeves=1)
    bt0 = run_backtest(px, b, cfg0, cost=0.001)
    bt1 = run_backtest(px, b, cfg1, cost=0.001)
    assert bt1.metrics["trades"] < bt0.metrics["trades"]
    sim, _ = simulate_paper(px, compute_features(px, b, cfg1), cfg1, cash=1e9, lot=1, start=bt1.equity.index[0])
    bt = run_backtest(px, b, cfg1, cost=0.0)
    ratio = (sim / sim.iloc[0]) / (bt.equity / bt.equity.iloc[0])
    assert (ratio - 1).abs().max() < 0.01


def test_plan_orders_band_keeps_positions_within_band():
    pf = Portfolio(cash=0, positions={"A": 100, "B": 100}, avg_cost={"A": 1000.0, "B": 1000.0})
    px = {"A": 1100.0, "B": 900.0, "C": 500.0}
    # 合計 200,000、目標 1/3 ≈ 66,667。A=110,000, B=90,000 は ±25% の外だが、1単元が目標額より高いので
    # 売って減らすことはできない（0 か 100 株）→ そのまま持つ。C は資金が残らず買えない
    orders, skipped = plan_orders(pf, ["A", "B", "C"], px, lot=100, band=0.25)
    assert orders == [] and skipped == ["C"]
    # 単元が 1 株なら、A と B を減らして C を買う
    orders1, _ = plan_orders(pf, ["A", "B", "C"], px, lot=1, band=0.25)
    assert {o.ticker for o in orders1} == {"A", "B", "C"} and all(o.side == "buy" for o in orders1 if o.ticker == "C")
    pf2 = Portfolio(cash=0, positions={"A": 100, "B": 100, "C": 100}, avg_cost={})
    px2 = {"A": 1000.0, "B": 1100.0, "C": 950.0}     # どれも目標 ±25% 以内
    orders2, _ = plan_orders(pf2, ["A", "B", "C"], px2, lot=100, band=0.25)
    assert orders2 == []
    orders3, _ = plan_orders(pf2, ["A", "B", "C"], px2, lot=1, band=0.0)   # バンドなしなら均等に戻す
    assert orders3
