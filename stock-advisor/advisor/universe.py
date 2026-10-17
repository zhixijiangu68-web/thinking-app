from __future__ import annotations

from pathlib import Path

DEFAULT_UNIVERSE_FILE = Path(__file__).resolve().parent.parent / "universe" / "jp-large.txt"
DEFAULT_BENCHMARK = "^N225"


def load_universe(path: str | Path | None = None) -> dict[str, str]:
    """銘柄リストを読む。戻り値は {ティッカー: 社名}。空行と # の行は無視。"""
    p = Path(path) if path else DEFAULT_UNIVERSE_FILE
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        ticker = parts[0]
        name = parts[1].strip() if len(parts) > 1 else ticker
        out[ticker] = name
    if not out:
        raise ValueError(f"銘柄リストが空です: {p}")
    return out
