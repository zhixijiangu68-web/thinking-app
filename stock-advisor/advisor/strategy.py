"""「どの株を買うか」を決めるルール（順張りだけ。2026-10-17 オーナーの判断）。設計の理由は docs/design.md。

買い（月1回の入れ替え日）:
1. 候補の条件: 株価が自分の 200日移動平均より上、年率の変動率が max_vol 以下、12-1モメンタムがプラス。
2. 並べ方: 12か月の値上がり率（直近1か月を除く）が高い順。同じ業種は max_per_sector まで。
3. 上位 top_n 銘柄を同じ金額ずつ持つ。残る銘柄は目標比率 ±rebalance_band 以内なら触らない。
4. 条件を満たす銘柄が足りなければ、その分は現金。

守りの層（既定では使わない。実データで年率 10 ポイントを削っていた。フラグで戻せる）:
- use_regime: 日経平均が 200日線の下なら全部現金
- exit_below_trend / stop_loss: 毎日の退出（200日線割れ・取得値から −N%）
- portfolio_dd_stop: 口座の遮断器（評価額が最高値から −N% で全部売って止める）
株価がとれない銘柄の退出（max_missing_days）はデータの安全装置なので残す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import signals as S


@dataclass
class StrategyConfig:
    top_n: int = 8
    mom_lookback: int = 252   # 約12か月
    mom_skip: int = 21        # 直近約1か月は除く
    trend_window: int = 200
    vol_window: int = 60
    max_vol: float = 0.60     # 年率60%を超える荒い銘柄は外す
    regime_window: int = 200
    use_regime: bool = False        # 順張りだけ（既定）。True で地合いの判定
    max_per_sector: int = 2   # 同じ業種はここまで（0 = 制限なし）
    exit_below_trend: bool = False  # 保有銘柄が200日線を割ったら売る（既定では使わない）
    stop_loss: float = 0.0          # 取得値から −N% で売る（0 = 使わない）
    portfolio_dd_stop: float = 0.0  # 評価額が最高値から −N% で全部売って止める（0 = 使わない）
    max_missing_days: int = 5       # 株価がこれ以上の営業日とれない銘柄は、最後の値で退出（上場廃止・データ障害）
    rebalance_band: float = 0.25    # 残る銘柄は、目標比率から ±25% 以内なら触らない（無駄な売買を減らす。0 = 毎回均等に戻す）
    scoring: str = "m12"            # 並べ方: "m12"（12-1モメンタムだけ）か "composite"（合成スコア。docs/composite-20261017.md）
    score_weights: dict | None = None   # composite の重み（None = signals.DEFAULT_WEIGHTS）
    weighting: str = "equal"        # 比率: "equal"（同額）か "invvol"（逆ボラティリティ、上限 max_weight・下限 min_weight）
    max_weight: float = 0.20
    min_weight: float = 0.05
    vol_target: float = 0.0         # ボラティリティ目標（年率）。0 = 使わない。選んだ銘柄の直近 vol_window 日の共分散から
                                    # 全部持ったときの変動率を見積もり、目標を超えるぶんだけ株の比率を下げて現金にする

    @property
    def min_history(self) -> int:
        return max(self.mom_lookback + self.mom_skip, self.trend_window, self.regime_window) + 5


@dataclass
class Features:
    momentum: pd.DataFrame
    trend_ok: pd.DataFrame
    vol: pd.DataFrame
    regime_ok: pd.Series  # 日付ごとに True/False
    score: pd.DataFrame | None = None   # composite のとき


@dataclass
class Pick:
    ticker: str
    score: float
    momentum: float
    vol: float
    price: float
    sector: str = ""
    weight: float = 0.0   # 目標比率


@dataclass
class Selection:
    asof: pd.Timestamp
    invested: bool
    picks: list[Pick] = field(default_factory=list)
    eligible: int = 0
    reason: str = ""
    sector_skipped: list[str] = field(default_factory=list)  # 業種の上限で外した銘柄
    exposure: float = 1.0        # 株に充てる比率の合計（ボラ目標で 1 未満になる）
    est_vol: float = float("nan")  # 選んだ組み合わせの見積もり変動率（年率）


def compute_features(prices: pd.DataFrame, bench: pd.Series | None, cfg: StrategyConfig) -> Features:
    raw = prices.sort_index()
    # 短い欠損（数日の取得失敗）は前の値で埋めて計算する。埋めないと rolling が 200 日間 NaN になり、退出扱いになる
    px = raw.ffill(limit=max(cfg.max_missing_days, 1))
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
    score = None
    if cfg.scoring == "composite":
        comps = S.compute_components(px, bench, mom_lookback=cfg.mom_lookback, mom_skip=cfg.mom_skip, vol_window=cfg.vol_window)
        score = S.composite_score(comps, cfg.score_weights)
    return Features(momentum=mom, trend_ok=trend_ok, vol=vol, regime_ok=regime_ok, score=score)


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
    if feats.score is not None:
        df["score"] = feats.score.loc[asof]
        df["eligible"] &= df["score"].notna()
    else:
        df["score"] = df["momentum"]
    return df.sort_values("score", ascending=False)


def _nearest_index(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Timestamp:
    asof = pd.Timestamp(asof)
    if asof in prices.index:
        return asof
    pos = prices.index.get_indexer([asof], method="pad")[0]
    if pos < 0:
        raise ValueError(f"{asof.date()} より前の株価がありません（データの開始は {prices.index[0].date()}）")
    return prices.index[pos]


def select_at(
    feats: Features, prices: pd.DataFrame, asof: pd.Timestamp, cfg: StrategyConfig,
    sectors: dict[str, str] | None = None,
) -> Selection:
    asof = _nearest_index(prices, asof)
    regime_ok = bool(feats.regime_ok.loc[asof]) if asof in feats.regime_ok.index else True
    if not regime_ok:
        return Selection(asof=asof, invested=False, reason="地合いが弱い（ベンチマークが200日線の下）ので全部現金")
    table = rank_at(feats, prices, asof, cfg)
    elig = table[table["eligible"]]
    sectors = sectors or {}
    count: dict[str, int] = {}
    picks: list[Pick] = []
    skipped: list[str] = []
    for t, r in elig.iterrows():
        if len(picks) >= cfg.top_n:
            break
        sec = sectors.get(t, "不明")
        if cfg.max_per_sector and count.get(sec, 0) >= cfg.max_per_sector:
            skipped.append(t)
            continue
        count[sec] = count.get(sec, 0) + 1
        picks.append(Pick(ticker=t, score=float(r.score), momentum=float(r.momentum),
                          vol=float(r.vol), price=float(r.price), sector=sec))
    reason = "" if picks else "条件を満たす銘柄がない（上昇トレンドかつ12-1モメンタムがプラスの銘柄なし）ので現金"
    exposure, est_vol = 1.0, float("nan")
    if picks:
        if cfg.weighting == "invvol":
            w = S.inverse_vol_weights(pd.Series({p.ticker: p.vol for p in picks}), cfg.max_weight, cfg.min_weight)
        else:
            w = {p.ticker: 1.0 / len(picks) for p in picks}
        if cfg.vol_target and cfg.vol_target > 0:
            est_vol = estimate_portfolio_vol(prices, asof, w, cfg.vol_window)
            if est_vol and est_vol > 0 and not np.isnan(est_vol):
                exposure = float(min(1.0, cfg.vol_target / est_vol))
        for p in picks:
            p.weight = w.get(p.ticker, 0.0) * exposure
    return Selection(asof=asof, invested=bool(picks), picks=picks, eligible=int(len(elig)),
                     reason=reason, sector_skipped=skipped, exposure=exposure, est_vol=est_vol)


def estimate_portfolio_vol(prices: pd.DataFrame, asof: pd.Timestamp, weights: dict[str, float], window: int = 60) -> float:
    """比率 weights で持ったときの年率変動率の見積もり（直近 window 日の共分散から）。"""
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        return float("nan")
    r = prices[tickers].loc[:asof].ffill().tail(window + 1).pct_change().dropna(how="all")
    if len(r) < max(10, window // 2):
        return float("nan")
    cov = r.cov().values * 252.0
    w = np.array([weights[t] for t in tickers])
    v = float(w @ np.nan_to_num(cov) @ w)
    return float(np.sqrt(v)) if v > 0 else float("nan")


@dataclass
class ExitSignal:
    ticker: str
    reason: str   # "trend" / "stop" / "missing"
    price: float
    change: float  # 取得値からの変化


def exit_signals(
    feats: Features, prices: pd.DataFrame, asof: pd.Timestamp, cfg: StrategyConfig,
    positions: dict[str, float],  # ticker -> 取得値（平均取得単価）
) -> list[ExitSignal]:
    """毎日の退出判定。positions のうち売るべき銘柄を返す。"""
    asof = _nearest_index(prices, asof)
    out: list[ExitSignal] = []
    pos = prices.index.get_loc(asof)
    for t, entry in positions.items():
        if t not in prices.columns:
            # 株価の表にない（銘柄リストから外れた、取得できない）。取得値のまま退出扱いにして人に知らせる
            out.append(ExitSignal(t, "missing", float(entry or 0.0), 0.0))
            continue
        px = prices.at[asof, t]
        if pd.isna(px):
            # 株価がとれない。しばらく続くなら最後の値で退出（売れるかは別問題なので、人が確かめる）
            col = prices[t].iloc[: pos + 1]
            last_valid = col.last_valid_index()
            missing_days = pos - prices.index.get_loc(last_valid) if last_valid is not None else pos + 1
            if cfg.max_missing_days and missing_days >= cfg.max_missing_days:
                last_px = float(col.loc[last_valid]) if last_valid is not None else float(entry or 0.0)
                chg = float(last_px / entry - 1.0) if entry and entry > 0 else 0.0
                out.append(ExitSignal(t, "missing", last_px, chg))
            continue
        chg = float(px / entry - 1.0) if entry and entry > 0 else 0.0
        if cfg.exit_below_trend and not bool(feats.trend_ok.at[asof, t]):
            out.append(ExitSignal(t, "trend", float(px), chg))
        elif cfg.stop_loss and chg <= -cfg.stop_loss:
            out.append(ExitSignal(t, "stop", float(px), chg))
    return out


def target_weights(current: dict[str, float], targets: dict[str, float] | list[str], band: float) -> dict[str, float]:
    """入れ替え日の目標比率。targets は {銘柄: 目標比率}（list なら同額）。
    残る銘柄は自分の目標から ±band 以内なら今の比率のまま、それ以外（新しい銘柄・はみ出た銘柄）で
    残りを目標比率に比例して分ける。current は評価額に対する比率。"""
    if not targets:
        return {}
    if isinstance(targets, (list, tuple)):
        targets = {t: 1.0 / len(targets) for t in targets}
    exposure = sum(targets.values())   # 1 未満ならその分は現金（ボラ目標）
    kept = {t: current[t] for t, tw in targets.items() if t in current and band > 0 and abs(current[t] - tw) <= band * tw}
    rest = {t: tw for t, tw in targets.items() if t not in kept}
    out = dict(kept)
    if rest:
        room = max(0.0, exposure - sum(kept.values()))
        rs = sum(rest.values())
        out.update({t: room * tw / rs for t, tw in rest.items()})
    return out
