"""Deterministic synthetic OHLCV generator for tests and examples.

All synthetic data is clearly marked as such in metadata. Never use synthetic
data in production reports without explicit user consent.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .ohlcv import OHLCV, OHLCVMetadata, write_csv


def generate_random_walk(
    start_price: float = 30_000.0,
    start_time: datetime | None = None,
    n_candles: int = 5_000,
    timeframe: str = "5m",
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.005,
    exchange: str = "synthetic",
    symbol: str = "BTC/USDT",
    label: str = "",
) -> OHLCV:
    """Generate a synthetic geometric-Brownian-motion candle series.

    Every candle has Open/High/Low/Close built from the same close-to-close
    return, with a uniform intra-candle range drawn from ``|ret| + U(0, vol)``.
    Volume is log-normal and uncorrelated.
    """
    if start_time is None:
        start_time = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(seed)
    delta = _tf_to_td(timeframe)
    ts = pd.date_range(start=start_time, periods=n_candles, freq=pd.Timedelta(delta), tz="UTC")
    rets = rng.normal(loc=drift, scale=vol, size=n_candles)
    closes = start_price * np.exp(np.cumsum(rets))
    opens = np.concatenate([[start_price], closes[:-1]])
    # Intrabar highs/lows
    ranges = np.abs(rets) + rng.uniform(0, vol, size=n_candles)
    highs = np.maximum(opens, closes) * (1.0 + ranges)
    lows = np.minimum(opens, closes) * (1.0 - ranges)
    volumes = rng.lognormal(mean=np.log(100), sigma=0.5, size=n_candles)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    ).set_index(pd.DatetimeIndex(ts, tz="UTC"))
    df.index.name = "date"
    note = "SYNTHETIC" + (f":{label}" if label else "") + " - not real exchange data"
    md = OHLCVMetadata(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_utc=ts[0].to_pydatetime(),
        end_utc=ts[-1].to_pydatetime(),
        n_rows=len(df),
        columns=tuple(df.columns),
        notes=note,
    )
    return OHLCV(df=df, metadata=md)


def generate_trend_with_cycles(
    start_price: float = 30_000.0,
    start_time: datetime | None = None,
    n_candles: int = 8_000,
    timeframe: str = "5m",
    seed: int = 7,
    exchange: str = "synthetic",
    symbol: str = "BTC/USDT",
) -> OHLCV:
    """Generate a trending-but-cyclic synthetic series that produces multiple trades."""
    if start_time is None:
        start_time = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(seed)
    delta = _tf_to_td(timeframe)
    ts = pd.date_range(start=start_time, periods=n_candles, freq=pd.Timedelta(delta), tz="UTC")
    t = np.arange(n_candles)
    # multi-frequency cycles + mild drift
    cycle = (
        np.sin(2 * np.pi * t / 288) * 0.04
        + np.sin(2 * np.pi * t / (288 * 6)) * 0.08
        + np.sin(2 * np.pi * t / (288 * 24)) * 0.12
    )
    drift = 0.00002 * t
    noise = rng.normal(0, 0.003, size=n_candles).cumsum() * 0.02
    logp = np.log(start_price) + drift + cycle + noise * 0.3
    closes = np.exp(logp)
    opens = np.concatenate([[start_price], closes[:-1]])
    ranges = 0.003 + np.abs(rng.normal(0, 0.003, n_candles))
    highs = np.maximum(opens, closes) * (1.0 + ranges)
    lows = np.minimum(opens, closes) * (1.0 - ranges)
    volumes = rng.lognormal(mean=np.log(150), sigma=0.5, size=n_candles) * (
        1.0 + 0.5 * np.abs(cycle)
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    ).set_index(pd.DatetimeIndex(ts, tz="UTC"))
    df.index.name = "date"
    md = OHLCVMetadata(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_utc=ts[0].to_pydatetime(),
        end_utc=ts[-1].to_pydatetime(),
        n_rows=len(df),
        columns=tuple(df.columns),
        notes="SYNTHETIC:trend-with-cycles - not real exchange data",
    )
    return OHLCV(df=df, metadata=md)


def _tf_to_td(tf: str) -> timedelta:
    from .timeframes import timeframe_to_timedelta

    return timeframe_to_timedelta(tf)


def write_fixture_csv(data: OHLCV, path: str | Path) -> Path:
    return write_csv(data, path)
