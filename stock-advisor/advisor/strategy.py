"""「どの株を買うか」を決めるルール。

ルール（モメンタム + トレンド + 地合い）:
1. 地合い: ベンチマーク（日経平均）が 200日移動平均より上のときだけ株を持つ。下なら全部現金。
2. 候補の条件: 株価が自分の 200日移動平均より上、かつ 60日の変動率（年率）が max_vol 以下。
3. 並べ方: 12か月の値上がり率（直近1か月を除く＝「12-1 モメンタム」）が高い順。
4. 上位 top_n 銘柄を同じ金額ずつ持つ。月に1回入れ替える。

根拠: モメンタム（Jegadeesh & Titman 1993）と移動平均によるトレンド追随（Faber 2007）は、
長期・多市場で確認されている数少ない規則性。ただし将来も効く保証はない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    top_n: int = 5
    mom_lookback: int = 252   # 約12か月
    mom_skip: int = 21        # 直近約1か月は除く
    trend_window: int = 200
    vol_window: int = 60
    max_vol: float = 0.60     # 年率60%を超える荒い銘柄は外す
    regime_window: int = 200
    use_regime: bool = True

    @property
    def min_history(self) -> int:
        return max(self.mom_lookback + self.mom_skip, self.trend_window, self.regime_window) + 5


@dataclass
class Features:
    momentum: pd.DataFrame
    trend_ok: pd.DataFrame
    vol: pd.DataFrame
    regime_ok: pd.Series  # 日付ごとに True/False


@dataclass
class Pick:
    ticker: str
    score: float
    momentum: float
    vol: float
    price: float


@dataclass
class Selection:
    asof: pd.Timestamp
    invested: bool
    picks: list[Pick] = field(default_factory=list)
    eligible: int = 0
    reason: str = ""


def compute_features(prices: pd.DataFrame, bench: pd.Series | None, cfg: StrategyConfig) -> Features:
    px = prices.sort_index()
    mom = px.shift(cfg.mom_skip) / px.shift(cfg.mom_lookback + cfg.mom_skip) - 1.0
    sma = px.rolling(cfg.trend_window, min_periods=cfg.trend_window).mean()
    trend_ok = (px > sma) & sma.notna()
    vol = np.log(px).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window).std() * np.sqrt(252)

    if cfg.use_regime and bench is not None:
        b = bench.reindex(px.index).ffill()
        bsma = b.rolling(cfg.regime_window, min_periods=cfg.regime_window).mean()
        regime_ok = (b > bsma) & bsma.notna()
    else:
        regime_ok = pd.Series(True, index=px.index)
    return Features(momentum=mom, trend_ok=trend_ok, vol=vol, regime_ok=regime_ok)


def rank_at(feats: Features, prices: pd.DataFrame, asof: pd.Timestamp, cfg: StrategyConfig) -> pd.DataFrame:
    """ある日の全銘柄の表（スコア順）。eligible 列が True のものが候補。"""
    df = pd.DataFrame({
        "price": prices.loc[asof],
        "momentum": feats.momentum.loc[asof],
        "trend_ok": feats.trend_ok.loc[asof],
        "vol": feats.vol.loc[asof],
    })
    df["eligible"] = (
        df["momentum"].notna() & df["vol"].notna() & df["price"].notna()
        & df["trend_ok"].astype(bool)
        & (df["vol"] <= cfg.max_vol)
        & (df["momentum"] > 0)
    )
    df["score"] = df["momentum"]
    return df.sort_values("score", ascending=False)


def select_at(feats: Features, prices: pd.DataFrame, asof: pd.Timestamp, cfg: StrategyConfig) -> Selection:
    asof = pd.Timestamp(asof)
    if asof not in prices.index:
        # 指定日が休場なら、その直前の営業日
        asof = prices.index[prices.index.get_indexer([asof], method="pad")[0]]
    regime_ok = bool(feats.regime_ok.loc[asof]) if asof in feats.regime_ok.index else True
    if not regime_ok:
        return Selection(asof=asof, invested=False, reason="地合いが弱い（ベンチマークが200日線の下）ので全部現金")
    table = rank_at(feats, prices, asof, cfg)
    elig = table[table["eligible"]]
    picks = [
        Pick(ticker=t, score=float(r.score), momentum=float(r.momentum), vol=float(r.vol), price=float(r.price))
        for t, r in elig.head(cfg.top_n).iterrows()
    ]
    reason = "" if picks else "条件を満たす銘柄がない（上昇トレンドかつ12-1モメンタムがプラスの銘柄なし）ので現金"
    return Selection(asof=asof, invested=bool(picks), picks=picks, eligible=int(len(elig)), reason=reason)
