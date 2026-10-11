"""A simple SMA-crossover reference strategy used for tests and examples."""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .indicators import RSI, cross_down, cross_up


class SmaCrossStrategy(Strategy):
    name = "SmaCross"
    version = "1.0.0"
    timeframe = "5m"
    inf_timeframes: list[str] = []
    btc_inf_timeframes: list[str] = []
    startup_candle_count = 40
    stoploss = -0.05
    minimal_roi = {"0": 0.10, "60": 0.05, "120": 0.02}

    fast_period = 10
    slow_period = 30
    rsi_period = 14
    rsi_overbought = 75

    def populate_indicators(self, df: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
        df = df.copy()
        df["sma_fast"] = df["close"].rolling(self.fast_period, min_periods=self.fast_period).mean()
        df["sma_slow"] = df["close"].rolling(self.slow_period, min_periods=self.slow_period).mean()
        df["rsi"] = RSI(df["close"], self.rsi_period)
        return df

    def populate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Classic SMA crossover: fast crosses above slow (trend-following entry).
        df["enter_long"] = (
            (
                cross_up(df["sma_fast"], df["sma_slow"])
                & df["sma_fast"].notna()
                & df["sma_slow"].notna()
            )
            .fillna(False)
            .astype(bool)
        )
        return df

    def populate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Exit when fast crosses below slow OR RSI indicates strong overbought exhaustion.
        df["exit_long"] = (
            (cross_down(df["sma_fast"], df["sma_slow"]) | (df["rsi"] > 85))
            .fillna(False)
            .astype(bool)
        )
        return df
