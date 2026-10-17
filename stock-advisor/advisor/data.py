"""株価（調整後終値）の取得。

- 本番: yfinance で Yahoo Finance から取り、data/cache/<ticker>.csv に保存する（取り直しは3回まで再試行）
- 検証: --synthetic で乱数の株価を作る（ネットワークなしで動作を確かめるため）
- sanity_report: 1日で ±30% 以上動いた・最近の株価がとれない、など人が確かめるべき点を出す
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CHUNK = 25          # 一度にダウンロードする銘柄数
RETRIES = 3


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_idx_").replace("/", "_")
    return cache_dir / f"{safe}.csv"


def _meta_path(cache_dir: Path, ticker: str) -> Path:
    return _cache_path(cache_dir, ticker).with_suffix(".meta.json")


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


def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_business_day(today: date | None = None) -> pd.Timestamp:
    """今日が営業日なら今日、そうでなければ直前の営業日。"""
    t = pd.Timestamp(today or date.today())
    return pd.bdate_range(end=t, periods=1)[0]


def _is_fresh(series: pd.Series, today: date | None = None) -> bool:
    """最新の営業日の終値まで入っていれば新しいとみなす。
    引け後に回すと毎日1回は取り直すが、土日は取り直さない。引け前に回すと取り直しても前日までしか入らない（それでよい）。"""
    if series.empty:
        return False
    return series.index.max() >= _latest_business_day(today)


def _download_once(tickers: list[str], start: str) -> dict[str, pd.Series]:
    import yfinance as yf  # 遅延 import（合成データのときは不要）

    out: dict[str, pd.Series] = {}
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
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


def _download(tickers: list[str], start: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        for attempt in range(1, RETRIES + 1):
            try:
                got = _download_once(chunk, start)
            except Exception as e:
                got = {}
                log.warning("ダウンロード失敗（%d/%d 回目）: %s", attempt, RETRIES, e)
            out.update(got)
            missing = [t for t in chunk if t not in out]
            if not missing:
                break
            chunk = missing
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
    return out


def load_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
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
        meta = _read_meta(_meta_path(cdir, t))
        covers_start = bool(meta.get("start")) and pd.Timestamp(meta["start"]) <= pd.Timestamp(start)
        if s is not None and _is_fresh(s) and covers_start:
            series[t] = s
        else:
            missing.append(t)

    if missing:
        log.info("%d 銘柄をダウンロードします", len(missing))
        fetched = _download(missing, start)
        for t, s in fetched.items():
            series[t] = s
            s.rename("Close").to_frame().rename_axis("Date").to_csv(_cache_path(cdir, t))
            _meta_path(cdir, t).write_text(
                json.dumps({"start": start, "fetched": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
        for t in missing:
            if t not in fetched:
                stale = _read_cache(_cache_path(cdir, t))
                if stale is not None and not stale.empty:
                    log.warning("取得できないので古いキャッシュを使います: %s（最終 %s）", t, stale.index.max().date())
                    series[t] = stale
                else:
                    log.warning("取得できませんでした: %s", t)

    if not series:
        raise RuntimeError("株価を1つも取得できませんでした。ネットワークとティッカーを確認してください。")
    df = pd.DataFrame(series).sort_index()
    df.index.name = "Date"
    df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[: pd.Timestamp(end)]
    return df


@dataclass
class SanityIssue:
    ticker: str
    kind: str      # "jump" / "stale" / "short"
    detail: str


def sanity_report(prices: pd.DataFrame, *, jump: float = 0.30, stale_days: int = 5, min_days: int = 260,
                  lookback_days: int = 60) -> list[SanityIssue]:
    """人が確かめるべき異常。直近 lookback_days 日の ±jump 以上の値動き、株価がとれていない銘柄、履歴が短い銘柄。"""
    issues: list[SanityIssue] = []
    if prices.empty:
        return issues
    last_day = prices.index[-1]
    recent = prices.iloc[-lookback_days:]
    chg = recent.pct_change()
    for t in prices.columns:
        col = prices[t]
        big = chg[t][chg[t].abs() >= jump].dropna()
        for d, v in big.items():
            issues.append(SanityIssue(t, "jump", f"{d.date()} に {v:+.0%}（分割の未調整や誤データかもしれない）"))
        lv = col.last_valid_index()
        if lv is None:
            issues.append(SanityIssue(t, "stale", "株価が1日もない"))
        else:
            age = prices.index.get_loc(last_day) - prices.index.get_loc(lv)
            if age >= stale_days:
                issues.append(SanityIssue(t, "stale", f"最後の株価が {lv.date()}（{age} 営業日前）"))
        if col.notna().sum() < min_days:
            issues.append(SanityIssue(t, "short", f"履歴が {int(col.notna().sum())} 日しかない（判断に使えない）"))
    return issues


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
        regime = np.repeat(rng.normal(0, 0.0008, size=max(1, n // 120)), 120)[:n]
        if len(regime) < n:
            regime = np.concatenate([regime, np.zeros(n - len(regime))])
        r = drift + regime + vol * rng.standard_normal(n)
        price = 1000.0 * (1 + i % 7) * np.exp(np.cumsum(r))
        cols[t] = price
    return pd.DataFrame(cols, index=idx)
