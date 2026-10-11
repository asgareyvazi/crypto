"""Stable strategy interface for ICBT.

Every strategy must subclass :class:`Strategy`, declare its base timeframe,
informative timeframes, startup warm-up length, and populate a ``signal``
column whose per-row values are one of :class:`SignalType`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(int, Enum):
    """Signal emitted by a strategy for a single candle.

    Signals are evaluated at candle **close**; fills occur on the subsequent
    candle's open (see :mod:`icbt.engine.engine`).
    """

    NONE = 0
    LONG = 1  # Enter long
    EXIT = 2  # Exit existing long
    # SHORT = 3 reserved for future; spot-only in v1.


@dataclass
class Signal:
    """One signal event tied to a candle close."""

    timestamp: pd.Timestamp
    type: SignalType
    reason: str = ""
    price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Runtime-configurable strategy parameters."""

    # Stake sizing
    stake_amount: float | None = None  # absolute; None => use stake_pct
    stake_pct: float = 0.10  # fraction of free cash per trade
    max_open_trades: int = 3
    # Risk controls
    stoploss: float | None = None  # e.g. -0.05; None => strategy default
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.03
    # ROI table (profit_pct -> minutes holding), mimicking Freqtrade's minimal_roi
    minimal_roi: dict[str, float] = field(default_factory=dict)
    # Additional strategy-specific overrides
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stake_amount": self.stake_amount,
            "stake_pct": self.stake_pct,
            "max_open_trades": self.max_open_trades,
            "stoploss": self.stoploss,
            "trailing_stop": self.trailing_stop,
            "trailing_stop_positive": self.trailing_stop_positive,
            "trailing_stop_positive_offset": self.trailing_stop_positive_offset,
            "minimal_roi": self.minimal_roi,
            "extra": self.extra,
        }


class Strategy:
    """Base class for all ICBT strategies.

    Subclasses must set class attributes and implement :meth:`populate_indicators`
    and :meth:`populate_signals`.
    """

    # Subclasses MUST set these
    name: str = "BaseStrategy"
    version: str = "0.0.0"
    timeframe: str = "5m"
    inf_timeframes: list[str] = []
    btc_inf_timeframes: list[str] = []
    startup_candle_count: int = 200
    use_custom_stoploss: bool = False
    stoploss: float = -0.10
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.03
    minimal_roi: dict[str, float] = {}

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    # ------------------------------------------------------------------ info
    def info(self) -> dict[str, Any]:
        v = self.version
        if v.startswith("v"):
            v = v[1:]
        return {
            "name": self.name,
            "version": v,
            "timeframe": self.timeframe,
            "inf_timeframes": self.inf_timeframes,
            "btc_inf_timeframes": self.btc_inf_timeframes,
            "startup_candle_count": self.startup_candle_count,
            "stoploss": self.config.stoploss if self.config.stoploss is not None else self.stoploss,
            "minimal_roi": self.config.minimal_roi or self.minimal_roi,
            "trailing_stop": self.config.trailing_stop or self.trailing_stop,
        }

    # ------------------------------------------------------------ indicator/signal population
    def populate_indicators(
        self,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Attach indicator columns to ``df`` and return it.

        ``df`` is the base (lowest) timeframe frame already enriched with
        informative-timeframe columns (see :meth:`informative_pairs` and
        :func:`icbt.data.timeframes.merge_informative`).
        """
        raise NotImplementedError("Subclasses must implement populate_indicators")

    def populate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set ``df['enter_long']`` to True/1 on candles that generate a long entry."""
        raise NotImplementedError

    def populate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set ``df['exit_long']`` to True/1 on candles that generate a long exit."""
        raise NotImplementedError

    # ------------------------------------------------------ optional hooks
    def custom_stoploss(
        self,
        trade: Any,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> float:
        """Optional per-trade dynamic stoploss. Return fraction (e.g. -0.05).

        Default returns the static stoploss (no custom behaviour).
        """
        return self.config.stoploss if self.config.stoploss is not None else self.stoploss

    def custom_exit(
        self,
        trade: Any,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> bool | None:
        """Optional per-trade exit logic. Return ``True`` (and a reason) to exit."""
        return False

    # ------------------------------------------------------- informative requirements
    def informative_pairs(self) -> list[tuple[str, str]]:
        """Return a list of (symbol, timeframe) informative pairs required.

        By default returns the base strategy's ``inf_timeframes`` for the
        primary pair plus ``btc_inf_timeframes`` for BTC/USDT if configured.
        """
        pairs: list[tuple[str, str]] = []
        # Populated at runtime by the engine with the primary symbol.
        return pairs

    def confirm_required_timeframes_available(
        self, available: dict[str, pd.DataFrame]
    ) -> list[str]:
        """Return list of missing required informative timeframes."""
        missing = []
        for tf in self.inf_timeframes + self.btc_inf_timeframes:
            if tf != self.timeframe and tf not in available:
                missing.append(tf)
        return missing
