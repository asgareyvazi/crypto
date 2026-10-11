"""Negative-path tests: the engine must reject bad input rather than silently returning bogus results."""

import numpy as np
import pandas as pd
import pytest

from icbt.data.ohlcv import OHLCV, OHLCVMetadata
from icbt.engine import BacktestEngine, EngineConfig


def _make_df(n, prices=None):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    if prices is None:
        prices = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "open": np.concatenate([[prices[0]], prices[:-1]]),
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 100,
            "timestamp": idx,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


def test_missing_column_errors_early(tmp_path):
    df = _make_df(100)
    bad = df.drop(columns=["volume"])
    md = OHLCVMetadata(
        exchange="t",
        symbol="TST/USDT",
        timeframe="5m",
        n_rows=len(bad),
        columns=tuple(bad.columns),
        start_utc=bad.index[0],
        end_utc=bad.index[-1],
    )
    OHLCV(df=bad, metadata=md)
    # save then load_csv should raise
    p = tmp_path / "bad.csv"
    bad.to_csv(p)
    with pytest.raises(ValueError):
        from icbt.data import load_csv

        load_csv(p)


def test_non_numeric_errors(tmp_path):
    p = tmp_path / "nonnum.csv"
    p.write_text("timestamp,open,high,low,close,volume\n2026-01-01,100,101,99,100,abc\n")
    from icbt.data import load_csv

    with pytest.raises(ValueError):
        load_csv(p)


def test_impossible_ohlc_fails_validation():
    df = _make_df(200)
    df.loc[df.index[10], "high"] = 90  # high < low (which is ~99)
    md = OHLCVMetadata(
        exchange="t",
        symbol="TST/USDT",
        timeframe="5m",
        n_rows=len(df),
        columns=tuple(df.columns),
        start_utc=df.index[0],
        end_utc=df.index[-1],
    )
    data = OHLCV(df=df, metadata=md)
    from icbt.data.validation import validate_ohlcv

    r = validate_ohlcv(data)
    assert not r.ok


def test_engine_rejects_missing_btc_when_required():
    from icbt.strategy.nfi_x8 import NostalgiaForInfinityX8

    df = _make_df(5000)
    md = OHLCVMetadata(
        exchange="t",
        symbol="TST/USDT",
        timeframe="5m",
        n_rows=len(df),
        columns=tuple(df.columns),
        start_utc=df.index[0],
        end_utc=df.index[-1],
    )
    data = OHLCV(df=df, metadata=md)
    cfg = EngineConfig(fail_on_gaps=True)
    eng = BacktestEngine(cfg)
    r = eng.run(data, NostalgiaForInfinityX8(), btc_data=None)
    assert any("BTC" in e for e in r.errors), r.errors
