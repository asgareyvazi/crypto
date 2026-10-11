"""Order, Trade and accounting primitives."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    STOP = "stop"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"


@dataclass
class Order:
    id: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    created_at: pd.Timestamp
    quantity: float
    price: float | None = None  # filled price (average) when filled
    stop_price: float | None = None
    fee: float = 0.0
    fee_ccy: str = "quote"
    reason: str = ""

    @staticmethod
    def new_market_buy(created_at: pd.Timestamp, quantity: float, reason: str = "") -> Order:
        return Order(
            id=_new_id("buy"),
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            status=OrderStatus.NEW,
            created_at=created_at,
            quantity=quantity,
            reason=reason,
        )

    @staticmethod
    def new_stop_sell(
        created_at: pd.Timestamp,
        quantity: float,
        stop_price: float,
        reason: str = "stop",
    ) -> Order:
        return Order(
            id=_new_id("stp"),
            side=OrderSide.SELL,
            type=OrderType.STOP,
            status=OrderStatus.NEW,
            created_at=created_at,
            quantity=quantity,
            stop_price=stop_price,
            reason=reason,
        )

    @staticmethod
    def new_market_sell(created_at: pd.Timestamp, quantity: float, reason: str = "") -> Order:
        return Order(
            id=_new_id("sel"),
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            status=OrderStatus.NEW,
            created_at=created_at,
            quantity=quantity,
            reason=reason,
        )

    def fill(self, price: float, fee: float, filled_at: pd.Timestamp | None = None) -> None:
        self.status = OrderStatus.FILLED
        self.price = float(price)
        self.fee = float(fee)
        if filled_at is not None:
            self.created_at = filled_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "side": self.side.value,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "fee": self.fee,
            "reason": self.reason,
        }


@dataclass
class Trade:
    """A single completed round-trip (buy -> sell)."""

    id: str
    symbol: str
    enter_time: pd.Timestamp
    enter_price: float
    quantity: float
    enter_fee: float
    enter_reason: str
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_fee: float = 0.0
    exit_reason: str = ""
    stop_price: float | None = None
    trailing_stop_activated: bool = False
    max_profit_pct: float = 0.0
    is_open: bool = True

    @staticmethod
    def open_new(symbol: str, enter_order: Order) -> Trade:
        return Trade(
            id=_new_id("trd"),
            symbol=symbol,
            enter_time=enter_order.created_at,
            enter_price=enter_order.price or 0.0,
            quantity=enter_order.quantity,
            enter_fee=enter_order.fee,
            enter_reason=enter_order.reason,
        )

    # ---- convenience ----
    @property
    def cost(self) -> float:
        return self.enter_price * self.quantity + self.enter_fee

    @property
    def proceeds(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.exit_price * self.quantity - self.exit_fee

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost

    @property
    def pnl_pct(self) -> float:
        if self.cost <= 0:
            return 0.0
        return self.pnl / self.cost

    @property
    def hold_minutes(self) -> float | None:
        if self.exit_time is None:
            return None
        return (self.exit_time - self.enter_time).total_seconds() / 60.0

    def close(self, exit_order: Order) -> None:
        self.exit_time = exit_order.created_at
        self.exit_price = exit_order.price or 0.0
        self.exit_fee = exit_order.fee
        self.exit_reason = exit_order.reason
        self.is_open = False

    def update_trailing(self, current_rate: float, offset: float, positive: float) -> None:
        """Update stoploss according to trailing-positive model (matches Freqtrade)."""
        profit_ratio = current_rate / self.enter_price - 1.0
        if profit_ratio < offset:
            return
        # Move stop to current - positive% of current
        new_stop = current_rate * (1.0 - positive)
        if self.stop_price is None or new_stop > self.stop_price:
            self.stop_price = new_stop
            self.trailing_stop_activated = True
        self.max_profit_pct = max(self.max_profit_pct, profit_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "enter_time": self.enter_time.isoformat(),
            "enter_price": self.enter_price,
            "quantity": self.quantity,
            "enter_fee": self.enter_fee,
            "enter_reason": self.enter_reason,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_fee": self.exit_fee,
            "exit_reason": self.exit_reason,
            "stop_price": self.stop_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "hold_minutes": self.hold_minutes,
            "max_profit_pct": self.max_profit_pct,
            "is_open": self.is_open,
        }
