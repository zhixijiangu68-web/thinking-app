"""合成スコア・逆ボラ配分・高速バックテストの検証"""
import numpy as np
import pandas as pd
import pytest

from advisor import signals as S
from advisor.backtest import run_backtest
from advisor.data import synthetic_prices
from advisor.engine import simulate_paper
from advisor.fast import fast_backtest, equal_weight_equity
from advisor.portfolio import Portfolio, plan_orders
from advisor.research import block_bootstrap_prices, exp_start_dates, exp_rolling_windows
from advisor.strategy import StrategyConfig, compute_features, select_at, target_weights

T = [f"{1000+i}.T" for i in range(20)]


@pytest.fixture(scope="module")
def data():
    px = synthetic_prices(T, "2015-01-01", "2022-12-30", seed=11)
    b = synthetic_prices(["^B"], "2015-01-01", "2022-12-30", seed=12)["^B"]
    return px, b


def test_components_and_score_shapes(data):
    px, b = data
    comps = S.compute_components(px, b)
    assert set(comps) == set(S.COMPONENTS)
    score = S.composite_score(comps)
    assert score.shape == px.shape
    last = score.iloc[-1].dropna()
    assert len(last) == len(T) and (last >= 0).all() and (last <= 1).all()
    # 重みを M12 だけにすると、順位は 12-1 モメンタムの順位と一致する
    only = S.composite_score(comps, {"M12": 1.0})
    assert (only.iloc[-1].rank() == comps["M12"].iloc[-1].rank()).all()


def test_fip_and_h52_direction(data):
    px, b = data
    comps = S.compute_components(px, b)
    # 52週高値の比率は 1 以下
    assert (comps["H52"].dropna() <= 1.0 + 1e-9).all().all()
    # FIP: 上昇日の割合が高い上昇銘柄ほど ID は負（連続）
    rets = px.pct_change()
    m12 = comps["M12"].iloc[-1]
    winners = m12[m12 > 0].index
    if len(winners):
        pos = (rets > 0).iloc[-252:].mean()
        fip = comps["FIP"].iloc[-1]
        # 上昇日の割合と FIP は負の相関（連続して上がる株ほど小さい）
        assert np.corrcoef(pos[winners], fip[winners])[0, 1] < 0


def test_inverse_vol_weights_caps():
    v = pd.Series({"A": 0.10, "B": 0.20, "C": 0.40, "D": 0.80, "E": 1.60, "F": 0.30, "G": 0.25, "H": 0.50})
    w = S.inverse_vol_weights(v, max_weight=0.20, min_weight=0.05)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert max(w.values()) <= 0.20 + 1e-9 and min(w.values()) >= 0.05 - 1e-9
    assert w["A"] >= w["E"]
    assert S.inverse_vol_weights(pd.Series(dtype=float)) == {}
    # 上限・下限が銘柄数と両立しないときは同額
    w2 = S.inverse_vol_weights(pd.Series({"A": 0.1, "B": 0.2}), max_weight=0.20, min_weight=0.05)
    assert w2 == {"A": 0.5, "B": 0.5}


def test_select_with_composite_and_invvol(data):
    px, b = data
    cfg = StrategyConfig(scoring="composite", weighting="invvol", top_n=5)
    feats = compute_features(px, b, cfg)
    assert feats.score is not None
    sel = select_at(feats, px, px.index[-1], cfg)
    if sel.invested:
        assert abs(sum(p.weight for p in sel.picks) - 1) < 1e-9
        assert all(0.05 - 1e-9 <= p.weight <= 0.20 + 1e-9 for p in sel.picks) or len(sel.picks) < 5
        scores = [p.score for p in sel.picks]
        assert scores == sorted(scores, reverse=True)


def test_target_weights_with_dict_targets():
    # A=0.30 は目標 0.5 の ±25%（0.375〜0.625）の外、B=0.36 も目標 0.25 の ±25% の外 → 全部を目標どおりに
    tw = target_weights({"A": 0.30, "B": 0.36}, {"A": 0.5, "B": 0.25, "D": 0.25}, band=0.25)
    assert tw == {"A": 0.5, "B": 0.25, "D": 0.25}
    # バンド内の銘柄はそのまま、残り（D）が余りを受け取る
    tw = target_weights({"A": 0.45, "B": 0.26}, {"A": 0.5, "B": 0.25, "D": 0.25}, band=0.25)
    assert tw["A"] == 0.45 and tw["B"] == 0.26 and abs(tw["D"] - 0.29) < 1e-9
    tw0 = target_weights({}, {"A": 0.5, "B": 0.25, "D": 0.25}, band=0.25)
    assert tw0 == {"A": 0.5, "B": 0.25, "D": 0.25}


def test_plan_orders_proportional_targets():
    pf = Portfolio(cash=1_000_000)
    px = {"A": 1000.0, "B": 1000.0, "C": 1000.0}
    orders, skipped = plan_orders(pf, {"A": 0.5, "B": 0.3, "C": 0.2}, px, lot=100, band=0.0)
    got = {o.ticker: o.shares for o in orders}
    assert got["A"] > got["B"] > got["C"] and all(n % 100 == 0 for n in got.values())
    assert sum(n * px[t] for t, n in got.items()) <= 1_000_000


def test_fast_backtest_matches_full(data):
    px, b = data
    for cfg in (StrategyConfig(scoring="m12", weighting="equal", rebalance_band=0.0, top_n=5),
                StrategyConfig(scoring="composite", weighting="invvol", rebalance_band=0.0, top_n=5)):
        fb = fast_backtest(px, b, cfg, cost=0.001)
        bt = run_backtest(px, b, cfg, cost=0.001)
        a = fb["equity"]; c = bt.equity.reindex(a.index)
        # 高速版は判断日の比率をそのまま約定日に使う（通常版は判断日の株数を約定日の株価で評価）ので、数%の差は出る
        assert ((a / a.iloc[0]) / (c / c.iloc[0]) - 1).abs().max() < 0.04


def test_paper_matches_backtest_composite_invvol(data):
    px, b = data
    cfg = StrategyConfig(scoring="composite", weighting="invvol", top_n=5)
    feats = compute_features(px, b, cfg)
    bt = run_backtest(px, b, cfg, cost=0.0)
    sim, _ = simulate_paper(px, feats, cfg, cash=1e9, lot=1, start=bt.equity.index[0])
    ratio = (sim / sim.iloc[0]) / (bt.equity / bt.equity.iloc[0])
    assert (ratio - 1).abs().max() < 0.01


def test_block_bootstrap_preserves_shape_and_listing(data):
    px, b = data
    px2 = px.copy(); px2.iloc[:300, 0] = np.nan      # 上場前
    new_px, new_b = block_bootstrap_prices(px2, b, np.random.default_rng(0), block=63)
    assert new_px.shape == px2.shape and new_px.iloc[:300, 0].isna().all() and new_px.iloc[300:, 0].notna().all()
    assert (new_px.dropna() > 0).all().all() and len(new_b) == len(px2)
    # 乱数が違えば違う歴史
    other, _ = block_bootstrap_prices(px2, b, np.random.default_rng(1), block=63)
    assert not np.allclose(new_px.iloc[-1].dropna().values, other.iloc[-1].dropna().values)


def test_start_dates_and_rolling_windows(data):
    px, b = data
    cfg = StrategyConfig(top_n=5, rebalance_band=0.0)
    r = fast_backtest(px, b, cfg)
    ew = equal_weight_equity(px, r["equity"].index)
    d = exp_start_dates(r["equity"], ew, min_years=2.0)
    assert len(d) > 12 and d["years"].is_monotonic_decreasing
    w = exp_rolling_windows({"x": r["equity"]}, ew, years=2)
    assert len(w) >= 3 and {"window", "ew", "x"} <= set(w.columns)
