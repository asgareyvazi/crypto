"""Engine configuration: capital, fees, slippage, position limits, fill model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FillModel(StrEnum):
    """How to fill market orders triggered by a candle close signal.

    NEXT_OPEN
        Conservative (default): signal is emitted at candle close T; the order
        fills at the next candle's open price. This avoids lookahead.
    NEXT_CLOSE
        Signal emitted at T fills at the close of T+1.
    SAME_CLOSE
        **Not recommended** - fills at the signal candle's close; only used for
        comparison/debugging and must be explicitly opted in.
    """

    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"
    SAME_CLOSE = "same_close"


class SlippageModel(StrEnum):
    NONE = "none"
    FIXED_BPS = "fixed_bps"  # constant basis points of adverse price movement
    VOLATILITY = "volatility"  # fraction of current ATR


@dataclass
class EngineConfig:
    # Capital & sizing
    initial_cash: float = 10_000.0
    stake_amount: float | None = None  # absolute; None => stake_pct * free_cash
    stake_pct: float = 0.10  # fraction of free cash per entry
    max_open_trades: int = 3

    # Execution costs
    fee_rate: float = 0.001  # 10 bps per trade (maker/taker flat)
    slippage_model: SlippageModel = SlippageModel.FIXED_BPS
    slippage_bps: float = 5.0  # 5 bps adverse slip on entries & exits
    slippage_atr_fraction: float = 0.05

    # Fill semantics
    fill_model: FillModel = FillModel.NEXT_OPEN

    # Risk/exit defaults (can be overridden by strategy)
    stoploss: float | None = None  # e.g. -0.05; None => use strategy value
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.03
    use_roi: bool = True
    minimal_roi: dict[str, float] = field(default_factory=dict)

    # Backtest window
    start: str | None = None  # ISO datetime or None => dataset start
    end: str | None = None

    # Misc
    fail_on_gaps: bool = False  # if True, validation ERRORs abort the run
    enable_buy_and_hold: bool = True

    # Reserved for future exchange metadata.
    exchange: str | None = None
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fill_model"] = self.fill_model.value
        d["slippage_model"] = self.slippage_model.value
        return d
