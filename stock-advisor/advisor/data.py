"""株価（調整後終値）の取得。

- 本番: yfinance で Yahoo Finance から取り、data/cache/<ticker>.csv に保存する
- 検証: --synthetic で乱数の株価を作る（ネットワークなしで動作を確かめるため）
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_idx_").replace("/", "_")
    return cache_dir / f"{safe}.csv"


def _read_cache(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        s = df["Close"].astype(float)
        s.index = pd.DatetimeIndex(s.index)
        return s.dropna()
    except Exception as e:  # 壊れたキャッシュは捨てる
        log.warning("キャッシュを読めないので取り直します: %s (%s)", path, e)
        return None


def _is_fresh(series: pd.Series, max_age_days: int) -> bool:
    if series.empty:
        return False
    last = series.index.max().date()
    return (date.today() - last).days <= max_age_days


def _download(tickers: list[str], start: str) -> dict[str, pd.Series]:
    import yfinance as yf  # 遅延 import（合成データのときは不要）

    out: dict[str, pd.Series] = {}
    if not tickers:
        return out
    raw = yf.download(
        tickers, start=start, auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    s = raw[t]["Close"]
                elif "Close" in raw.columns.get_level_values(0):
                    s = raw["Close"][t]
                else:
                    continue
            else:
                s = raw["Close"]
        except KeyError:
            continue
        s = pd.Series(np.asarray(s, dtype=float), index=pd.DatetimeIndex(raw.index), name="Close").dropna()
        if not s.empty:
            out[t] = s
    return out


def load_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    max_age_days: int = 1,
    synthetic: bool = False,
    seed: int = 0,
    end: str | None = None,
) -> pd.DataFrame:
    """調整後終値の表（行: 日付、列: ティッカー）を返す。取れなかった銘柄は列に含めない。"""
    if synthetic:
        return synthetic_prices(tickers, start=start, end=end, seed=seed)

    cdir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for t in tickers:
        s = None if refresh else _read_cache(_cache_path(cdir, t))
        if s is not None and _is_fresh(s, max_age_days) and s.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10):
            series[t] = s
        else:
            missing.append(t)

    if missing:
        log.info("%d 銘柄をダウンロードします", len(missing))
        fetched = _download(missing, start)
        for t, s in fetched.items():
            series[t] = s
            s.rename("Close").to_frame().rename_axis("Date").to_csv(_cache_path(cdir, t))
        for t in missing:
            if t not in fetched:
                log.warning("取得できませんでした: %s", t)

    if not series:
        raise RuntimeError("株価を1つも取得できませんでした。ネットワークとティッカーを確認してください。")
    df = pd.DataFrame(series).sort_index()
    df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[: pd.Timestamp(end)]
    return df


def synthetic_prices(
    tickers: list[str], start: str = "2015-01-01", end: str | None = None, seed: int = 0
) -> pd.DataFrame:
    """検証用の合成株価。幾何ブラウン運動 + 銘柄ごとに違うトレンド。"""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(date.today())
    idx = pd.bdate_range(pd.Timestamp(start), end_ts)
    n = len(idx)
    cols = {}
    for i, t in enumerate(tickers):
        drift = rng.normal(0.00025, 0.0004)       # 年率 +6% くらいを中心にばらつく
        vol = rng.uniform(0.012, 0.025)            # 日次ボラ 1.2〜2.5%
        # ときどきトレンドが切り替わる（モメンタムが意味を持つように）
        regime = np.repeat(rng.normal(0, 0.0008, size=max(1, n // 120)), 120)[:n]
        if len(regime) < n:
            regime = np.concatenate([regime, np.zeros(n - len(regime))])
        r = drift + regime + vol * rng.standard_normal(n)
        price = 1000.0 * (1 + i % 7) * np.exp(np.cumsum(r))
        cols[t] = price
    return pd.DataFrame(cols, index=idx)
