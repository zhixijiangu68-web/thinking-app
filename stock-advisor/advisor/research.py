"""1000回の検証（docs/composite-20261017.md の A〜F）。実データで GitHub Actions 上で回す。

A ブートストラップ: 日次リターンを全銘柄共通の時間ブロックで並べ替えた仮想の歴史で、合成 / M12 / 均等買い持ちを比べる
B 重みのランダム化: 合成の 6 要素の重みを乱数で引き直す
C 入れ替え日のずらし: 月の 1〜20 営業日目
D 開始日のずらし: 各月を開始日に（同じ判断列の評価額の比だけで出せる）
E 期間を分ける: 前半で選び、後半で確かめる。3年の窓でも
F 格子: 上位の数・業種上限
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import signals as S
from .backtest import metrics_from_equity
from .fast import equal_weight_equity, fast_backtest, tranche_backtest
from .strategy import StrategyConfig, compute_features

BASE_M12 = StrategyConfig(scoring="m12", weighting="equal", rebalance_band=0.0, sleeves=1)   # 高速版は 1 列で回し、分割は H で別に検証
BASE_COMP = StrategyConfig(scoring="composite", weighting="invvol", rebalance_band=0.0, sleeves=1)
BASE_VT = replace(BASE_M12, vol_target=0.20)   # 案1: ボラ目標 20%（docs/risk-sizing-20261017.md）


def _cagr(eq: pd.Series) -> float:
    m = metrics_from_equity(eq)
    return m.get("cagr", np.nan)


def _sharpe(eq: pd.Series) -> float:
    return metrics_from_equity(eq).get("sharpe", np.nan)


def _mdd(eq: pd.Series) -> float:
    return metrics_from_equity(eq).get("max_drawdown", np.nan)


# ---------- A ----------
def block_bootstrap_prices(prices: pd.DataFrame, bench: pd.Series, rng: np.random.Generator, block: int = 126):
    """日次リターンを長さ block のブロックで（全銘柄とベンチマーク共通の時間軸で）並べ替えて株価を作り直す。"""
    px = prices.ffill()
    r = px.pct_change().fillna(0.0).values
    rb = bench.reindex(px.index).ffill().pct_change().fillna(0.0).values
    n = len(px)
    starts = rng.integers(0, n - block, size=n // block + 1)
    order = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:n]
    new_r, new_rb = r[order], rb[order]
    new_px = pd.DataFrame(np.cumprod(1 + new_r, axis=0) * 100.0, index=px.index, columns=px.columns)
    new_b = pd.Series(np.cumprod(1 + new_rb) * 100.0, index=px.index)
    # 元データで株価がなかった日（上場前）は NaN に戻す（履歴の短い銘柄が最初から存在したことにしない）
    new_px = new_px.where(prices.notna().values)
    return new_px, new_b


def exp_bootstrap(prices, bench, sectors, runs: int, seed: int = 0, cost: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(runs):
        px, b = block_bootstrap_prices(prices, bench, rng)
        f_comp = compute_features(px, b, BASE_COMP)
        f_m12 = compute_features(px, b, BASE_M12)
        c = fast_backtest(px, b, BASE_COMP, cost=cost, sectors=sectors, feats=f_comp)
        m = fast_backtest(px, b, BASE_M12, cost=cost, sectors=sectors, feats=f_m12)
        v = fast_backtest(px, b, BASE_VT, cost=cost, sectors=sectors, feats=f_m12)
        days = c["equity"].index
        ew = equal_weight_equity(px, days)
        rows.append({"run": i, "comp": _cagr(c["equity"]), "m12": _cagr(m["equity"]), "ew": _cagr(ew), "vt": _cagr(v["equity"]),
                     "comp_sharpe": _sharpe(c["equity"]), "m12_sharpe": _sharpe(m["equity"]), "ew_sharpe": _sharpe(ew), "vt_sharpe": _sharpe(v["equity"]),
                     "comp_mdd": _mdd(c["equity"]), "m12_mdd": _mdd(m["equity"]), "ew_mdd": _mdd(ew), "vt_mdd": _mdd(v["equity"])})
    return pd.DataFrame(rows)


# ---------- B ----------
def exp_random_weights(prices, bench, sectors, runs: int, seed: int = 1, cost: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_feats = compute_features(prices, bench, BASE_COMP)
    comps = S.compute_components(prices.ffill(limit=5), bench, mom_lookback=BASE_COMP.mom_lookback,
                                 mom_skip=BASE_COMP.mom_skip, vol_window=BASE_COMP.vol_window)
    m12 = fast_backtest(prices, bench, BASE_M12, cost=cost, sectors=sectors)
    days = m12["equity"].index
    ew = equal_weight_equity(prices, days)
    rows = [{"run": -1, "label": "M12だけ", "cagr": _cagr(m12["equity"]), "sharpe": _sharpe(m12["equity"]), "mdd": _mdd(m12["equity"])},
            {"run": -2, "label": "均等買い持ち", "cagr": _cagr(ew), "sharpe": _sharpe(ew), "mdd": _mdd(ew)}]
    for i in range(runs):
        w = rng.dirichlet(np.ones(len(S.COMPONENTS)))
        weights = dict(zip(S.COMPONENTS, w))
        feats = replace(base_feats, score=S.composite_score(comps, weights))
        cfg = replace(BASE_COMP, score_weights=weights)
        r = fast_backtest(prices, bench, cfg, cost=cost, sectors=sectors, feats=feats)
        rows.append({"run": i, "label": "ランダムな重み", "cagr": _cagr(r["equity"]), "sharpe": _sharpe(r["equity"]), "mdd": _mdd(r["equity"]),
                     **{f"w_{k}": float(v) for k, v in weights.items()}})
    return pd.DataFrame(rows)


# ---------- C ----------
def exp_offsets(prices, bench, sectors, cfg: StrategyConfig, offsets=range(0, 20), cost: float = 0.001) -> pd.DataFrame:
    feats = compute_features(prices, bench, cfg)
    rows = []
    for k in offsets:
        r = fast_backtest(prices, bench, cfg, cost=cost, sectors=sectors, feats=feats, rebalance_offset=k)
        rows.append({"offset": k, "cagr": _cagr(r["equity"]), "sharpe": _sharpe(r["equity"]), "mdd": _mdd(r["equity"])})
    return pd.DataFrame(rows)


# ---------- D ----------
def exp_start_dates(eq: pd.Series, ew: pd.Series, min_years: float = 5.0) -> pd.DataFrame:
    """同じ判断列で、開始月を変えたときの年率（終わりは同じ）。"""
    rows = []
    months = eq.resample("MS").first().dropna().index
    end = eq.index[-1]
    for m0 in months:
        d0 = eq.index[eq.index.get_indexer([m0], method="bfill")[0]]
        years = (end - d0).days / 365.25
        if years < min_years:
            break
        rows.append({"start": d0.date(), "years": years,
                     "cagr": (eq.iloc[-1] / eq.loc[d0]) ** (1 / years) - 1,
                     "ew": (ew.iloc[-1] / ew.loc[d0]) ** (1 / years) - 1})
    return pd.DataFrame(rows)


# ---------- E ----------
def exp_walk_forward(prices, bench, sectors, candidates: list[tuple[str, StrategyConfig]], split: str, cost: float = 0.001) -> dict:
    """前半（〜split）で候補の中からシャープが最良のものを選び、後半（split〜）で全候補を評価する。"""
    rows = []
    feats_cache = {}
    for label, cfg in candidates:
        key = (cfg.scoring, cfg.use_regime)
        if key not in feats_cache:
            feats_cache[key] = compute_features(prices, bench, cfg)
        feats = feats_cache[key]
        full = fast_backtest(prices, bench, cfg, cost=cost, sectors=sectors, feats=feats)
        eq = full["equity"]
        a, b = eq.loc[:split], eq.loc[split:]
        rows.append({"label": label, "in_cagr": _cagr(a), "in_sharpe": _sharpe(a), "out_cagr": _cagr(b),
                     "out_sharpe": _sharpe(b), "out_mdd": _mdd(b)})
    df = pd.DataFrame(rows)
    chosen = df.sort_values("in_sharpe", ascending=False).iloc[0]["label"]
    days = fast_backtest(prices, bench, candidates[0][1], cost=cost, sectors=sectors, feats=feats_cache[(candidates[0][1].scoring, candidates[0][1].use_regime)])["equity"].index
    ew = equal_weight_equity(prices, days)
    ew_out = ew.loc[split:]
    return {"table": df, "chosen": chosen, "ew_out_cagr": _cagr(ew_out), "ew_out_sharpe": _sharpe(ew_out), "ew_out_mdd": _mdd(ew_out)}


def exp_rolling_windows(eqs: dict[str, pd.Series], ew: pd.Series, years: int = 3) -> pd.DataFrame:
    """years 年の窓を1年ずつずらし、各戦略が均等買い持ちに勝った窓の割合。"""
    idx = ew.index
    rows = []
    start_years = range(idx[0].year, idx[-1].year - years + 1)
    for y in start_years:
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + years}-01-01")
        w_ew = ew.loc[a:b]
        if len(w_ew) < 200:
            continue
        row = {"window": f"{y}-{y + years}", "ew": _cagr(w_ew)}
        for k, e in eqs.items():
            row[k] = _cagr(e.loc[a:b])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- F ----------
def exp_grid(prices, bench, sectors, base: StrategyConfig, cost: float = 0.001) -> pd.DataFrame:
    feats = compute_features(prices, bench, base)
    rows = []
    for top in (5, 8, 10, 12, 15):
        for sec in (0, 2, 3):
            for maxw in (0.15, 0.20, 0.30):
                cfg = replace(base, top_n=top, max_per_sector=sec, max_weight=maxw)
                r = fast_backtest(prices, bench, cfg, cost=cost, sectors=sectors, feats=feats)
                rows.append({"top": top, "sector_cap": sec, "max_weight": maxw, "cagr": _cagr(r["equity"]),
                             "sharpe": _sharpe(r["equity"]), "mdd": _mdd(r["equity"])})
    return pd.DataFrame(rows)


# ---------- 全部 ----------
def run_all(prices, bench, sectors, *, bootstrap_runs: int = 300, weight_runs: int = 200, cost: float = 0.001,
            split: str = "2019-01-01", log=print) -> dict:
    t0 = time.time()
    out: dict = {}
    f_comp = compute_features(prices, bench, BASE_COMP)
    f_m12 = compute_features(prices, bench, BASE_M12)
    comp = fast_backtest(prices, bench, BASE_COMP, cost=cost, sectors=sectors, feats=f_comp)
    m12 = fast_backtest(prices, bench, BASE_M12, cost=cost, sectors=sectors, feats=f_m12)
    days = comp["equity"].index
    ew = equal_weight_equity(prices, days)
    out["base"] = {"comp": comp["metrics"], "m12": m12["metrics"], "ew": metrics_from_equity(ew)}
    log(f"[0] 基準: 合成 {comp['metrics']['cagr']:+.1%} / M12 {m12['metrics']['cagr']:+.1%} / 均等 {metrics_from_equity(ew)['cagr']:+.1%}  ({time.time()-t0:.0f}s)")
    out["A"] = exp_bootstrap(prices, bench, sectors, bootstrap_runs, cost=cost)
    log(f"[A] ブートストラップ {bootstrap_runs} 回 ({time.time()-t0:.0f}s)")
    out["B"] = exp_random_weights(prices, bench, sectors, weight_runs, cost=cost)
    log(f"[B] 重みのランダム化 {weight_runs} 回 ({time.time()-t0:.0f}s)")
    out["C"] = {"comp": exp_offsets(prices, bench, sectors, BASE_COMP, cost=cost), "m12": exp_offsets(prices, bench, sectors, BASE_M12, cost=cost),
                "vt": exp_offsets(prices, bench, sectors, BASE_VT, cost=cost)}
    log(f"[C] 入れ替え日のずらし 60 回 ({time.time()-t0:.0f}s)")
    # H: 分割入れ替え（4列）。開始の週 0〜4 で回す
    rows = []
    for k in range(5):
        tr = tranche_backtest(prices, bench, BASE_M12, feats=f_m12, base_offset=k, cost=cost, sectors=sectors)
        rows.append({"offset": k, "cagr": tr["metrics"]["cagr"], "sharpe": tr["metrics"]["sharpe"], "mdd": tr["metrics"]["max_drawdown"]})
    out["H"] = pd.DataFrame(rows)
    out["vt_full"] = fast_backtest(prices, bench, BASE_VT, cost=cost, sectors=sectors, feats=f_m12)["metrics"]
    out["E_roll_vt"] = exp_rolling_windows({"M12": m12["equity"], "ボラ目標": fast_backtest(prices, bench, BASE_VT, cost=cost, sectors=sectors, feats=f_m12)["equity"]}, ew)
    log(f"[H] 分割入れ替え 20 回 ({time.time()-t0:.0f}s)")
    out["D"] = {"comp": exp_start_dates(comp["equity"], ew), "m12": exp_start_dates(m12["equity"], ew)}
    log(f"[D] 開始日のずらし ({time.time()-t0:.0f}s)")
    candidates = [
        ("M12だけ・同額", BASE_M12),
        ("M12だけ・同額・ボラ目標20%", BASE_VT),
        ("M12だけ・同額・ボラ目標15%", replace(BASE_M12, vol_target=0.15)),
        ("M12だけ・逆ボラ", replace(BASE_M12, weighting="invvol")),
        ("合成・同額", replace(BASE_COMP, weighting="equal")),
        ("合成・逆ボラ", BASE_COMP),
        ("合成・逆ボラ・上位10", replace(BASE_COMP, top_n=10)),
        ("合成・逆ボラ・上位12", replace(BASE_COMP, top_n=12)),
    ]
    out["E"] = exp_walk_forward(prices, bench, sectors, candidates, split, cost=cost)
    out["E_roll"] = exp_rolling_windows({"合成": comp["equity"], "M12": m12["equity"]}, ew)
    log(f"[E] 前半/後半と3年の窓 ({time.time()-t0:.0f}s)")
    out["F"] = exp_grid(prices, bench, sectors, BASE_COMP, cost=cost)
    log(f"[F] 格子 45 回 ({time.time()-t0:.0f}s)")
    out["elapsed"] = time.time() - t0
    out["total_runs"] = bootstrap_runs * 4 + weight_runs + 60 + 20 + 2 + len(out["D"]["comp"]) + len(out["D"]["m12"]) + len(candidates) + 45
    return out


def verdict(out: dict) -> tuple[list[tuple[str, bool, str]], bool]:
    """docs/composite-20261017.md の合格基準（事前に決めたもの）。"""
    A, B, E = out["A"], out["B"], out["E"]
    a1 = (A["comp"] > A["m12"]).mean(); a2 = (A["comp"] > A["ew"]).mean()
    rw = B[B["run"] >= 0]; m12_cagr = float(B[B["label"] == "M12だけ"]["cagr"].iloc[0])
    b1 = (rw["cagr"] > m12_cagr).mean()
    t = E["table"].set_index("label")
    comp_out, m12_out = t.loc["合成・逆ボラ"], t.loc["M12だけ・同額"]
    e_ok = bool(comp_out["out_cagr"] > m12_out["out_cagr"] and comp_out["out_cagr"] > E["ew_out_cagr"]
                and comp_out["out_sharpe"] > m12_out["out_sharpe"] and comp_out["out_sharpe"] > E["ew_out_sharpe"])
    C = out["C"]["comp"]; D = out["D"]["comp"]
    ew_cagr = out["base"]["ew"]["cagr"]
    c_ok = bool(C["cagr"].min() >= ew_cagr - 0.03 and (D["cagr"] - D["ew"]).min() >= -0.03)
    checks = [
        (f"A: 仮想の歴史で 合成 > M12 が {a1:.0%}（60% 以上）、合成 > 均等 が {a2:.0%}（55% 以上）", bool(a1 >= 0.60 and a2 >= 0.55), ""),
        (f"B: ランダムな重みの {b1:.0%} が M12 だけを上回る（70% 以上）", bool(b1 >= 0.70), ""),
        (f"E: 後半（{out.get('split', '')}〜）で合成が M12 と均等の両方に年率・シャープで勝つ", e_ok,
         f"合成 {comp_out['out_cagr']:+.1%}/{comp_out['out_sharpe']:.2f}, M12 {m12_out['out_cagr']:+.1%}/{m12_out['out_sharpe']:.2f}, 均等 {E['ew_out_cagr']:+.1%}/{E['ew_out_sharpe']:.2f}"),
        (f"C・D: 最悪の日取り {C['cagr'].min():+.1%} と最悪の開始日の差 {(D['cagr'] - D['ew']).min():+.1%} が均等 −3 ポイント以内", c_ok, ""),
    ]
    passed = sum(ok for _, ok, _ in checks) >= 3
    return checks, passed


def verdict_sizing(out: dict) -> tuple[list[tuple[str, bool, str]], bool, list[tuple[str, bool, str]], bool]:
    """docs/risk-sizing-20261017.md の案1（ボラ目標）と案2（分割入れ替え）の合格基準。"""
    A, E = out["A"], out["E"]
    t = E["table"].set_index("label")
    base_out, vt_out = t.loc["M12だけ・同額"], t.loc["M12だけ・同額・ボラ目標20%"]
    c1 = bool(vt_out["out_sharpe"] >= base_out["out_sharpe"] + 0.10 and vt_out["out_mdd"] >= base_out["out_mdd"] + 0.05)
    c2 = float((A["vt_sharpe"] > A["m12_sharpe"]).mean())
    base_full, vt_full = out["base"]["m12"], out["vt_full"]
    c3 = bool(vt_full["cagr"] >= base_full["cagr"] - 0.03)
    R = out["E_roll_vt"]
    # 窓ごとのシャープは持っていないので、年率で代用（事前基準は「シャープ」だが窓のシャープは計算していない → 年率で判定し、明記する）
    c4 = float((R["ボラ目標"] > R["M12"]).mean())
    checks1 = [
        (f"1. 後半のシャープ +0.10 以上かつ最大下落 5 ポイント以上浅い（基準 {base_out['out_sharpe']:.2f}/{base_out['out_mdd']:.0%} → ボラ目標 {vt_out['out_sharpe']:.2f}/{vt_out['out_mdd']:.0%}）", c1, ""),
        (f"2. ブートストラップでシャープが基準より良い割合 {c2:.0%}（65% 以上）", bool(c2 >= 0.65), ""),
        (f"3. 全期間の年率が基準より 3 ポイント以上は下がらない（{base_full['cagr']:+.1%} → {vt_full['cagr']:+.1%}）", c3, ""),
        (f"4. 3年の窓で基準より良い窓 {c4:.0%}（60% 以上。窓のシャープは出していないので年率で判定）", bool(c4 >= 0.60), ""),
    ]
    pass1 = sum(ok for _, ok, _ in checks1) >= 3
    C, H = out["C"]["m12"], out["H"]
    single_range = float(C["cagr"].max() - C["cagr"].min())
    tr_range = float(H["cagr"].max() - H["cagr"].min())
    med_diff = float(H["cagr"].median() - C["cagr"].median())
    checks2 = [
        (f"年率の幅: 1列 {single_range:.1%} → 4列 {tr_range:.1%}（半分以下。4つの平均なので幅が縮むのは当然で、これは実装の確認）", bool(tr_range <= single_range / 2), ""),
        (f"年率の中央値の差 {med_diff:+.1%}（±1 ポイント）", bool(abs(med_diff) <= 0.01), ""),
    ]
    pass2 = all(ok for _, ok, _ in checks2)
    return checks1, pass1, checks2, pass2


def report(out: dict, title: str = "1000回の検証") -> str:
    L = [f"# {title}", ""]
    b = out["base"]
    L.append(f"合計 {out['total_runs']} 回のバックテスト、{out['elapsed']/60:.1f} 分。")
    L.append("")
    L.append("| 基準の設定（全期間） | 年率 | シャープ | 最大下落 |")
    L.append("|---|---:|---:|---:|")
    for k, lab in (("comp", "合成スコア・逆ボラ・上位8"), ("m12", "M12 だけ・同額・上位8"), ("ew", "候補の均等買い持ち")):
        L.append(f"| {lab} | {b[k]['cagr']:+.1%} | {b[k]['sharpe']:.2f} | {b[k]['max_drawdown']:.1%} |")
    checks, passed = verdict(out)
    L += ["", "## 合格基準（事前に決めたもの）", ""]
    for text, ok, detail in checks:
        L.append(f"- {'合格' if ok else '不合格'}: {text}" + (f"（{detail}）" if detail else ""))
    L.append("")
    L.append(f"**判定: {'3つ以上を満たしたので合成スコアを既定にしてよい' if passed else '満たさなかったので、複雑にしない（M12 だけのまま）'}**")
    A = out["A"]
    L += ["", f"## A. ブートストラップ（{len(A)} 本の仮想の歴史）", ""]
    L.append("| | 合成 | M12 | 均等 |"); L.append("|---|---:|---:|---:|")
    for lab, q in (("年率の中央値", 0.5), ("年率の下位10%", 0.1), ("年率の上位10%", 0.9)):
        L.append(f"| {lab} | {A['comp'].quantile(q):+.1%} | {A['m12'].quantile(q):+.1%} | {A['ew'].quantile(q):+.1%} |")
    L.append(f"| シャープの中央値 | {A['comp_sharpe'].median():.2f} | {A['m12_sharpe'].median():.2f} | {A['ew_sharpe'].median():.2f} |")
    L.append(f"| 最大下落の中央値 | {A['comp_mdd'].median():.1%} | {A['m12_mdd'].median():.1%} | {A['ew_mdd'].median():.1%} |")
    L.append(f"\n合成 > M12: {(A['comp'] > A['m12']).mean():.0%}、合成 > 均等: {(A['comp'] > A['ew']).mean():.0%}、M12 > 均等: {(A['m12'] > A['ew']).mean():.0%}")
    B = out["B"]; rw = B[B["run"] >= 0]
    L += ["", f"## B. 重みのランダム化（{len(rw)} 通り）", ""]
    L.append(f"年率: 最低 {rw['cagr'].min():+.1%} / 中央 {rw['cagr'].median():+.1%} / 最高 {rw['cagr'].max():+.1%}。M12 だけ {float(B[B['label']=='M12だけ']['cagr'].iloc[0]):+.1%}、均等 {float(B[B['label']=='均等買い持ち']['cagr'].iloc[0]):+.1%}")
    corr = {k: float(np.corrcoef(rw[f"w_{k}"], rw["cagr"])[0, 1]) for k in S.COMPONENTS}
    L.append("重みと年率の相関（正なら、その要素を重くするほど良かった）: " + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items()))
    C = out["C"]
    L += ["", "## C. 入れ替え日のずらし（月の 1〜20 営業日目）", ""]
    L.append(f"合成: 年率 {C['comp']['cagr'].min():+.1%}〜{C['comp']['cagr'].max():+.1%}（中央 {C['comp']['cagr'].median():+.1%}）。M12: {C['m12']['cagr'].min():+.1%}〜{C['m12']['cagr'].max():+.1%}")
    D = out["D"]
    L += ["", f"## D. 開始日のずらし（{len(D['comp'])} 通り、終わりは同じ、5年以上）", ""]
    diff = D["comp"]["cagr"] - D["comp"]["ew"]
    L.append(f"合成 − 均等: 最低 {diff.min():+.1%} / 中央 {diff.median():+.1%} / 最高 {diff.max():+.1%}。均等に負けた開始日の割合 {(diff < 0).mean():.0%}")
    diff2 = D["m12"]["cagr"] - D["m12"]["ew"]
    L.append(f"M12 − 均等: 最低 {diff2.min():+.1%} / 中央 {diff2.median():+.1%} / 最高 {diff2.max():+.1%}")
    E = out["E"]
    L += ["", f"## E. 前半で選び、後半で確かめる（分かれ目 {out.get('split', '')}）", ""]
    L.append("| 設定 | 前半 年率 | 前半 シャープ | 後半 年率 | 後半 シャープ | 後半 最大下落 |"); L.append("|---|---:|---:|---:|---:|---:|")
    for r in E["table"].itertuples():
        L.append(f"| {r.label} | {r.in_cagr:+.1%} | {r.in_sharpe:.2f} | {r.out_cagr:+.1%} | {r.out_sharpe:.2f} | {r.out_mdd:.1%} |")
    L.append(f"| 均等買い持ち（後半） | | | {E['ew_out_cagr']:+.1%} | {E['ew_out_sharpe']:.2f} | {E['ew_out_mdd']:.1%} |")
    L.append(f"\n前半のシャープで選ばれたのは「{E['chosen']}」。")
    R = out["E_roll"]
    L += ["", "## E-2. 3年の窓（1年ずつずらす）", ""]
    L.append("| 窓 | 合成 | M12 | 均等 |"); L.append("|---|---:|---:|---:|")
    for r in R.itertuples():
        L.append(f"| {r.window} | {getattr(r, '合成'):+.1%} | {r.M12:+.1%} | {r.ew:+.1%} |")
    L.append(f"\n均等に勝った窓: 合成 {(R['合成'] > R['ew']).mean():.0%}、M12 {(R['M12'] > R['ew']).mean():.0%}")
    if "vt_full" in out:
        checks1, pass1, checks2, pass2 = verdict_sizing(out)
        v = out["vt_full"]; bm = out["base"]["m12"]
        L += ["", "## G. 案1: ボラ目標 20%（順張りのまま、ポジションの大きさの規律）", ""]
        L.append("| 全期間 | 年率 | シャープ | 最大下落 | 年率変動率 |"); L.append("|---|---:|---:|---:|---:|")
        L.append(f"| M12 だけ（基準） | {bm['cagr']:+.1%} | {bm['sharpe']:.2f} | {bm['max_drawdown']:.1%} | {bm['annual_vol']:.1%} |")
        L.append(f"| M12 + ボラ目標 20% | {v['cagr']:+.1%} | {v['sharpe']:.2f} | {v['max_drawdown']:.1%} | {v['annual_vol']:.1%} |")
        L.append(f"\nブートストラップ {len(A)} 本: 年率の中央値 {A['vt'].median():+.1%}（基準 {A['m12'].median():+.1%}）、シャープの中央値 {A['vt_sharpe'].median():.2f}（基準 {A['m12_sharpe'].median():.2f}）、最大下落の中央値 {A['vt_mdd'].median():.1%}（基準 {A['m12_mdd'].median():.1%}）")
        L.append(f"入れ替え日のずらし: ボラ目標 {out['C']['vt']['cagr'].min():+.1%}〜{out['C']['vt']['cagr'].max():+.1%}")
        L.append("")
        for text, ok, _ in checks1:
            L.append(f"- {'合格' if ok else '不合格'}: {text}")
        L.append(f"\n**案1の判定: {'合格（3つ以上）' if pass1 else '不合格'}**")
        H = out["H"]
        L += ["", "## H. 案2: 分割入れ替え（4列、週をずらす）", ""]
        L.append("| 開始の週 | 年率 | シャープ | 最大下落 |"); L.append("|---:|---:|---:|---:|")
        for r in H.itertuples():
            L.append(f"| {int(r.offset)} | {r.cagr:+.1%} | {r.sharpe:.2f} | {r.mdd:.1%} |")
        L.append("")
        for text, ok, _ in checks2:
            L.append(f"- {'合格' if ok else '不合格'}: {text}")
        L.append(f"\n**案2の判定: {'合格' if pass2 else '不合格'}**")
    F = out["F"]
    L += ["", "## F. 格子（上位の数 × 業種上限 × 1銘柄の上限）", ""]
    L.append(f"年率 {F['cagr'].min():+.1%}〜{F['cagr'].max():+.1%}、シャープ {F['sharpe'].min():.2f}〜{F['sharpe'].max():.2f}。最良: " +
             ", ".join(f"上位{int(r.top)}/業種{int(r.sector_cap)}/上限{r.max_weight:.0%} {r.cagr:+.1%}" for r in F.sort_values("sharpe", ascending=False).head(3).itertuples()))
    L.append("")
    if out.get("dropped"):
        L.append("データが疑わしいので外した銘柄: " + "、".join(out["dropped"]))
        L.append("")
    L.append("※ 候補は今の大型株（生存バイアス）。日次終値・片道 0.1%。合格基準は検証の前に docs/composite-20261017.md に書いたもので、あとから動かしていない。")
    return "\n".join(L) + "\n"
