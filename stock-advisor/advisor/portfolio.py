"""紙上の口座（ペーパートレード）。JSON に保存する。実際のお金は動かない。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

DEFAULT_PORTFOLIO_FILE = Path(__file__).resolve().parent.parent / "data" / "paper-portfolio.json"
LOT_SIZE = 100  # 東証の単元株


@dataclass
class Order:
    ticker: str
    side: str        # "buy" / "sell"
    shares: int
    price: float     # 想定約定値（直近終値）
    reason: str = ""

    @property
    def value(self) -> float:
        return self.shares * self.price


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)   # ticker -> 株数
    avg_cost: dict[str, float] = field(default_factory=dict)  # ticker -> 平均取得単価
    history: list[dict] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    peak_equity: float = 0.0        # 遮断器の基準（評価額の最高値）
    halted: bool = False            # 遮断器が動いた。人が `paper resume` するまで売買しない
    halted_reason: str = ""
    last_rebalance: str = ""        # 最後に入れ替えた日（YYYY-MM-DD）

    # --- 保存・読込 ---
    @classmethod
    def load(cls, path: str | Path = DEFAULT_PORTFOLIO_FILE) -> "Portfolio":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)

    def save(self, path: str | Path = DEFAULT_PORTFOLIO_FILE) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 評価 ---
    def market_value(self, prices: dict[str, float]) -> float:
        return sum(n * prices.get(t, self.avg_cost.get(t, 0.0)) for t, n in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    # --- 約定の反映 ---
    def apply(self, order: Order, when: str | None = None) -> None:
        when = when or datetime.now().isoformat(timespec="seconds")
        t, n = order.ticker, order.shares
        if order.side == "buy":
            if order.value > self.cash + 1e-6:
                raise ValueError(f"現金不足: {t} {n}株 {order.value:,.0f}円 > 現金 {self.cash:,.0f}円")
            held = self.positions.get(t, 0)
            self.avg_cost[t] = (held * self.avg_cost.get(t, 0.0) + n * order.price) / (held + n)
            self.positions[t] = held + n
            self.cash -= order.value
        elif order.side == "sell":
            held = self.positions.get(t, 0)
            if n > held:
                raise ValueError(f"保有以上の売り: {t} {n}株 > 保有 {held}株")
            self.positions[t] = held - n
            self.cash += order.value
            if self.positions[t] == 0:
                del self.positions[t]
                self.avg_cost.pop(t, None)
        else:
            raise ValueError(f"不明な side: {order.side}")
        self.history.append({"when": when, **asdict(order)})


def plan_orders(
    pf: Portfolio,
    targets: list[str],
    prices: dict[str, float],
    *,
    lot: int = LOT_SIZE,
    cash_reserve: float = 0.0,
    names: dict[str, str] | None = None,
) -> tuple[list[Order], list[str]]:
    """いまの保有を targets（同じ金額ずつ）に近づける注文を作る。売り → 買いの順。単元株に丸める。

    戻り値は (注文, 資金が足りず1単元も買えない銘柄)。
    1単元が高すぎて買えない銘柄は外し、残りで均等に分ける。余った現金は上位の銘柄から
    1単元ずつ足す（1銘柄あたり均等額の1.5倍まで）。
    """
    names = names or {}
    orders: list[Order] = []
    cash = pf.cash

    # 1) 対象から外れた銘柄は全部売る
    for t, n in list(pf.positions.items()):
        if t not in targets:
            if t not in prices:
                continue  # 株価が取れない銘柄は触らない
            orders.append(Order(t, "sell", n, prices[t], reason="対象から外れた"))
            cash += n * prices[t]

    targets = [t for t in targets if t in prices and prices[t] > 0]
    if not targets:
        return orders, []

    # 2) 1単元も買えない銘柄を外しながら、均等額を決める
    investable = max(0.0, (cash + sum(n * prices[t] for t, n in pf.positions.items() if t in targets)) - cash_reserve)
    active = list(targets)
    while active:
        per = investable / len(active)
        too_expensive = [t for t in active if prices[t] * lot > per and pf.positions.get(t, 0) == 0]
        if not too_expensive:
            break
        active = [t for t in active if t not in too_expensive]
    skipped = [t for t in targets if t not in active]
    if not active:
        return orders, skipped
    per = investable / len(active)
    target_shares = {t: int(math.floor(per / prices[t] / lot)) * lot for t in active}

    # 3) 減らす分を先に売る
    for t, want in target_shares.items():
        have = pf.positions.get(t, 0)
        if want < have:
            orders.append(Order(t, "sell", have - want, prices[t], reason="比率を下げる"))
            cash += (have - want) * prices[t]

    # 4) 増やす分を買う。現金が足りなければ、いちばん大きい買いから1単元ずつ減らす
    buys = {t: want - pf.positions.get(t, 0) for t, want in target_shares.items() if want > pf.positions.get(t, 0)}
    spend = lambda: sum(n * prices[t] for t, n in buys.items())
    while buys and spend() > cash - cash_reserve + 1e-6:
        biggest = max(buys, key=lambda t: buys[t] * prices[t])
        buys[biggest] -= lot
        if buys[biggest] <= 0:
            del buys[biggest]
    # 5) 余った現金で、上位から1単元ずつ足す（均等額の1.5倍まで）
    changed = True
    while changed:
        changed = False
        for t in active:
            held_after = pf.positions.get(t, 0) + buys.get(t, 0)
            if (held_after + lot) * prices[t] <= 1.5 * per and spend() + lot * prices[t] <= cash - cash_reserve + 1e-6:
                buys[t] = buys.get(t, 0) + lot
                changed = True
    for t, n in buys.items():
        orders.append(Order(t, "buy", n, prices[t], reason=f"新規・買い増し {names.get(t, '')}".strip()))
    return _net(orders), skipped


def _net(orders: list[Order]) -> list[Order]:
    """同じ銘柄の売りと買いをまとめて、1銘柄1注文にする。売りが先。"""
    net: dict[str, int] = {}
    reason: dict[str, str] = {}
    price: dict[str, float] = {}
    for o in orders:
        net[o.ticker] = net.get(o.ticker, 0) + (o.shares if o.side == "buy" else -o.shares)
        reason.setdefault(o.ticker, o.reason)
        price[o.ticker] = o.price
    out = [Order(t, "sell", -n, price[t], reason[t]) for t, n in net.items() if n < 0]
    out += [Order(t, "buy", n, price[t], reason[t]) for t, n in net.items() if n > 0]
    return out
