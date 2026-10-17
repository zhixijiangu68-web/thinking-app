"""合成スコアの要素（docs/composite-20261017.md）。全部、日付ごとの全候補の中での順位（0〜1、高いほど良い）に直す。

- M12 / M6: 12か月・6か月の値上がり率（直近1か月除く）
- RES: 残差モメンタム（日経平均に対するベータを除いた残りの累積、直近1か月除く）
- FIP: 情報の連続性（sign(M12) × (下落日の割合 − 上昇日の割合)。小さいほど良い）
- H52: 株価 ÷ 52週高値
- LOWVOL: 60日変動率が低いほど良い
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: dict[str, float] = {"M12": 0.30, "M6": 0.15, "RES": 0.20, "FIP": 0.15, "H52": 0.10, "LOWVOL": 0.10}
COMPONENTS = tuple(DEFAULT_WEIGHTS)


def momentum(px: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return px.shift(skip) / px.shift(lookback + skip) - 1.0


def residual_momentum(rets: pd.DataFrame, mkt: pd.Series, lookback: int = 252, skip: int = 21, beta_window: int = 252) -> pd.DataFrame:
    """r_i − β_i·r_m の累積（β は過去 beta_window 日の共分散 ÷ 分散）。"""
    m = mkt.reindex(rets.index)
    cov = rets.rolling(beta_window, min_periods=beta_window // 2).cov(m)
    var = m.rolling(beta_window, min_periods=beta_window // 2).var()
    beta = cov.div(var, axis=0)
    resid = rets.sub(beta.mul(m, axis=0))
    return resid.rolling(lookback, min_periods=lookback // 2).sum().shift(skip)


def frog_in_pan(rets: pd.DataFrame, m12: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """情報の連続性 ID。小さい（負に大きい）ほど「じわじわ」。"""
    pos = (rets > 0).rolling(window, min_periods=window // 2).mean()
    neg = (rets < 0).rolling(window, min_periods=window // 2).mean()
    return np.sign(m12) * (neg - pos)


def high52(px: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    return px / px.rolling(window, min_periods=window // 2).max()


def annual_vol(px: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return np.log(px).diff().rolling(window, min_periods=window).std() * np.sqrt(252)


def pct_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """日付ごとの順位（0〜1）。ascending=True なら大きいほど 1 に近い。"""
    return df.rank(axis=1, pct=True, ascending=ascending)


def compute_components(px: pd.DataFrame, bench: pd.Series | None, *, mom_lookback: int = 252, mom_skip: int = 21,
                       vol_window: int = 60) -> dict[str, pd.DataFrame]:
    """各要素の生の値。px は短い欠損を埋めた株価。"""
    rets = px.pct_change()
    m12 = momentum(px, mom_lookback, mom_skip)
    out = {
        "M12": m12,
        "M6": momentum(px, mom_lookback // 2, mom_skip),
        "FIP": frog_in_pan(rets, m12),
        "H52": high52(px),
        "LOWVOL": annual_vol(px, vol_window),
    }
    if bench is not None:
        b = bench.reindex(px.index).ffill().pct_change()
        out["RES"] = residual_momentum(rets, b, mom_lookback, mom_skip)
    else:
        out["RES"] = m12.copy()   # ベンチマークがなければ素のモメンタムで代用
    return out


def composite_score(components: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """順位に直して重みつきで合成（0〜1）。重みは合計 1 に正規化。"""
    w = dict(weights or DEFAULT_WEIGHTS)
    total = sum(w.values())
    w = {k: v / total for k, v in w.items() if v > 0}
    ranks = {
        "M12": pct_rank(components["M12"]),
        "M6": pct_rank(components["M6"]),
        "RES": pct_rank(components["RES"]),
        "FIP": pct_rank(components["FIP"], ascending=False),      # 小さいほど良い
        "H52": pct_rank(components["H52"]),
        "LOWVOL": pct_rank(components["LOWVOL"], ascending=False),  # 変動率が低いほど良い
    }
    score = None
    for k, wk in w.items():
        part = ranks[k] * wk
        score = part if score is None else score.add(part, fill_value=0.0)
    # 要素が1つでも NaN の銘柄はスコアも NaN（判断に使えない）
    mask = None
    for k in w:
        m = ranks[k].notna()
        mask = m if mask is None else (mask & m)
    return score.where(mask)


def inverse_vol_weights(vols: pd.Series, max_weight: float = 0.20, min_weight: float = 0.05) -> dict[str, float]:
    """逆ボラティリティの比率（上限・下限つき、合計 1）。vols は年率変動率。"""
    v = vols.dropna()
    v = v[v > 0]
    if v.empty:
        return {}
    n = len(v)
    if n * min_weight > 1.0 or n * max_weight < 1.0:
        # 上限・下限が銘柄数と両立しないときは同額
        return {t: 1.0 / n for t in v.index}
    w = (1.0 / v.values)
    w = w / w.sum()
    for _ in range(20):
        w = np.clip(w, min_weight, max_weight)
        w = w / w.sum()
        if w.min() >= min_weight - 1e-12 and w.max() <= max_weight + 1e-12:
            break
    return {t: float(x) for t, x in zip(v.index, w)}
