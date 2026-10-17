"""注文の実行先。

- PaperBroker: 紙上で約定させる（既定）。お金は動かない。
- 実際の証券会社につなぐ場合は Broker を実装したクラスを足す（例: auカブコム証券の kabuステーションAPI）。
  実際の発注は取り返しがつかないので、オーナーの明示的な判断なしには有効にしない。
"""
from __future__ import annotations

from typing import Protocol

from .portfolio import Order, Portfolio


class Broker(Protocol):
    def execute(self, orders: list[Order]) -> list[Order]:
        """注文を出し、約定した注文を返す。"""
        ...


class PaperBroker:
    def __init__(self, portfolio: Portfolio, when: str | None = None):
        self.pf = portfolio
        self.when = when

    def execute(self, orders: list[Order]) -> list[Order]:
        done: list[Order] = []
        # 売りを先に（現金を作ってから買う）
        for o in sorted(orders, key=lambda o: 0 if o.side == "sell" else 1):
            self.pf.apply(o, when=self.when)
            done.append(o)
        return done


class LiveBroker:
    """まだ未接続。使うと止まる（安全側）。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "実際の発注はまだつないでいません。証券会社のAPI（例: kabuステーションAPI）の接続は、"
            "ペーパートレードで数か月は確かめてから、オーナーの判断で足してください。"
        )
