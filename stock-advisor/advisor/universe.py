from __future__ import annotations

from pathlib import Path

DEFAULT_UNIVERSE_FILE = Path(__file__).resolve().parent.parent / "universe" / "jp-wide.txt"   # 2026-10-17 に 67 銘柄 → 266 銘柄
DEFAULT_BENCHMARK = "^N225"


def load_universe_info(path: str | Path | None = None) -> dict[str, tuple[str, str]]:
    """銘柄リストを読む。戻り値は {ティッカー: (社名, 業種)}。空行と # の行は無視。
    1行は「コード.T  社名  業種」。業種がなければ "不明"。"""
    p = Path(path) if path else DEFAULT_UNIVERSE_FILE
    out: dict[str, tuple[str, str]] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ticker = parts[0]
        name = parts[1] if len(parts) > 1 else ticker
        sector = parts[2] if len(parts) > 2 else "不明"
        out[ticker] = (name, sector)
    if not out:
        raise ValueError(f"銘柄リストが空です: {p}")
    return out


def load_universe(path: str | Path | None = None) -> dict[str, str]:
    """{ティッカー: 社名}"""
    return {t: n for t, (n, _) in load_universe_info(path).items()}


def load_sectors(path: str | Path | None = None) -> dict[str, str]:
    """{ティッカー: 業種}"""
    return {t: s for t, (_, s) in load_universe_info(path).items()}
