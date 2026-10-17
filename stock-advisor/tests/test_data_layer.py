"""データ層の検証（yfinance の応答を模倣して、取得・キャッシュ・新しさ・失敗時の扱いを確かめる）"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import yfinance

from advisor import data as D


def _series_dict(tickers, start="2024-01-01", days=400):
    """_download_once の戻り値（解析後）を模倣"""
    idx = pd.bdate_range(start, periods=days)
    return {t: pd.Series(np.linspace(100, 200, days), index=idx, name="Close") for t in tickers}


def _yf_frame(tickers, start="2024-01-01", days=400, ticker_first=True):
    idx = pd.bdate_range(start, periods=days)
    cols, vals = [], []
    for t in tickers:
        for f in ("Open", "High", "Low", "Close", "Volume"):
            cols.append((t, f) if ticker_first else (f, t))
            vals.append(np.linspace(100, 200, days) if f != "Volume" else np.ones(days))
    df = pd.DataFrame(np.array(vals).T, index=idx, columns=pd.MultiIndex.from_tuples(cols))
    return df


@pytest.mark.parametrize("ticker_first", [True, False])
def test_download_parses_both_column_orders(monkeypatch, ticker_first):
    calls = []

    def fake_yf_download(tickers, **kw):
        calls.append((list(tickers), kw.get("auto_adjust"), kw.get("group_by")))
        return _yf_frame(tickers, ticker_first=ticker_first)

    monkeypatch.setattr(yfinance, "download", fake_yf_download)
    got = D._download(["7203.T", "6758.T"], "2024-01-01")
    assert set(got) == {"7203.T", "6758.T"}
    assert got["7203.T"].name == "Close" and len(got["7203.T"]) == 400
    assert float(got["7203.T"].iloc[0]) == 100.0
    assert calls == [(["7203.T", "6758.T"], True, "ticker")]   # 調整後終値・銘柄ごとの列


def test_download_single_ticker_flat_columns(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=10)
    flat = pd.DataFrame({"Open": 1.0, "Close": np.arange(10.0), "Volume": 1.0}, index=idx)
    monkeypatch.setattr(yfinance, "download", lambda tickers, **kw: flat)
    got = D._download(["^N225"], "2024-01-01")
    assert list(got) == ["^N225"] and float(got["^N225"].iloc[-1]) == 9.0


def test_download_empty_response(monkeypatch):
    monkeypatch.setattr(yfinance, "download", lambda tickers, **kw: pd.DataFrame())
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    assert D._download(["A.T"], "2024-01-01") == {}


def test_download_retries_only_missing(monkeypatch):
    calls = []

    def flaky(tickers, start):
        calls.append(list(tickers))
        if len(calls) == 1:
            return _series_dict([tickers[0]])      # 1回目は1銘柄しか返ってこない
        return _series_dict(tickers)

    monkeypatch.setattr(D, "_download_once", flaky)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    got = D._download(["A.T", "B.T"], "2024-01-01")
    assert set(got) == {"A.T", "B.T"}
    assert calls == [["A.T", "B.T"], ["B.T"]]


def test_download_gives_up_after_retries(monkeypatch):
    calls = []

    def boom(tickers, start):
        calls.append(1)
        raise ConnectionError("no network")

    monkeypatch.setattr(D, "_download_once", boom)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    assert D._download(["A.T"], "2024-01-01") == {}
    assert len(calls) == D.RETRIES


def test_load_prices_caches_and_reuses(monkeypatch, tmp_path):
    calls = []

    def fake(tickers, start):
        calls.append(list(tickers))
        return _series_dict(tickers, start="2024-01-01", days=400)

    monkeypatch.setattr(D, "_download_once", fake)
    # 「今日」をデータの最終日の翌日にする（キャッシュが新しいと判定されるように）
    last = pd.bdate_range("2024-01-01", periods=400)[-1].date()
    monkeypatch.setattr(D, "_latest_business_day", lambda today=None: pd.Timestamp(last))

    df = D.load_prices(["A.T", "B.T"], start="2024-01-01", cache_dir=tmp_path)
    assert list(df.columns) == ["A.T", "B.T"] and len(df) == 400
    assert (tmp_path / "A.T.csv").exists() and json.loads((tmp_path / "A.T.meta.json").read_text())["start"] == "2024-01-01"

    df2 = D.load_prices(["A.T", "B.T"], start="2024-01-01", cache_dir=tmp_path)     # 2回目はキャッシュ
    assert len(calls) == 1
    pd.testing.assert_frame_equal(df2, df, check_freq=False)

    D.load_prices(["A.T"], start="2020-01-01", cache_dir=tmp_path)                  # もっと前からほしい → 取り直す
    assert len(calls) == 2 and calls[1] == ["A.T"]

    D.load_prices(["A.T"], start="2024-06-01", cache_dir=tmp_path)                  # もっと後からでよい → キャッシュ
    assert len(calls) == 2


def test_load_prices_falls_back_to_stale_cache(monkeypatch, tmp_path):
    good = lambda tickers, start: _series_dict(tickers, start="2024-01-01", days=400)
    monkeypatch.setattr(D, "_download_once", good)
    monkeypatch.setattr(D, "_latest_business_day", lambda today=None: pd.Timestamp("2024-06-01"))
    D.load_prices(["A.T"], start="2024-01-01", cache_dir=tmp_path)
    # 日が進んでキャッシュが古くなり、取り直しも失敗する
    monkeypatch.setattr(D, "_latest_business_day", lambda today=None: pd.Timestamp("2026-01-05"))
    monkeypatch.setattr(D, "_download_once", lambda tickers, start: (_ for _ in ()).throw(ConnectionError("x")))
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    df = D.load_prices(["A.T"], start="2024-01-01", cache_dir=tmp_path)
    assert "A.T" in df.columns and len(df) == 400                                   # 古いキャッシュで動く


def test_load_prices_raises_when_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_download_once", lambda tickers, start: {})
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        D.load_prices(["ZZZ.T"], start="2024-01-01", cache_dir=tmp_path)


def test_index_ticker_cache_filename(tmp_path):
    assert D._cache_path(tmp_path, "^N225").name == "_idx_N225.csv"


def test_suspicious_tickers_flags_split_like_jumps():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"OK": np.linspace(100, 150, 300), "BAD": np.linspace(100, 150, 300)}, index=idx)
    px.loc[idx[150]:, "BAD"] *= 3.0     # 分割の未調整のような +200%
    bad = D.suspicious_tickers(px)
    assert list(bad) == ["BAD"] and "+200%" in bad["BAD"]
    assert D.suspicious_tickers(px[["OK"]]) == {}
