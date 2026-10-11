"""Tests for informative-timeframe column wiring and no-lookahead semantics at engine level."""
from __future__ import annotations

import pandas as pd

from icbt.data import resample_ohlcv
from icbt.data.synthetic import generate_trend_with_cycles
from icbt.data.timeframes import merge_informative
from icbt.engine import BacktestEngine, EngineConfig
from icbt.strategy.nfi_x8 import NostalgiaForInfinityX8
from icbt.strategy.sma_cross import SmaCrossStrategy


def test_merge_informative_suffix_produces_expected_columns():
    data = generate_trend_with_cycles(n_candles=500)
    h1 = resample_ohlcv(data.df[["open", "high", "low", "close", "volume"]], "5m", "1h")
    merged = merge_informative(data.df, h1, "1h", "5m", "1h")
    for col in ("open_1h", "high_1h", "low_1h", "close_1h", "volume_1h"):
        assert col in merged.columns, f"missing informative column {col}"


def test_merge_informative_btc_suffix():
    data = generate_trend_with_cycles(n_candles=500, seed=1)
    btc = generate_trend_with_cycles(n_candles=500, seed=2, symbol="BTC/USDT")
    btc_4h = resample_ohlcv(btc.df[["open", "high", "low", "close", "volume"]], "5m", "4h")
    merged = merge_informative(data.df, btc_4h, "btc_4h", "5m", "4h")
    for col in ("open_btc_4h", "high_btc_4h", "low_btc_4h", "close_btc_4h", "volume_btc_4h"):
        assert col in merged.columns


def test_engine_populates_all_expected_informative_columns_for_x8():
    """Running NFI-X8 end-to-end on 5m (+ BTC 5m) must expose informative columns
    close_15m, close_1h, close_4h, close_1d, close_btc_4h in the enriched frame,
    which the strategy depends on."""
    data = generate_trend_with_cycles(n_candles=9000, seed=11)  # ~31 days of 5m -> enough for warmup + 1d TF
    btc = generate_trend_with_cycles(n_candles=9000, seed=22, symbol="BTC/USDT")
    cfg = EngineConfig(initial_cash=10000, fee_rate=0, slippage_bps=0)
    eng = BacktestEngine(cfg)
    # Monkey-patch to inspect the enriched frame
    seen = {}
    original_populate = NostalgiaForInfinityX8.populate_indicators

    def spy(self, df, metadata=None):
        seen["cols"] = list(df.columns)
        return original_populate(self, df, metadata)

    NostalgiaForInfinityX8.populate_indicators = spy
    try:
        result = eng.run(data, NostalgiaForInfinityX8(), btc_data=btc)
    finally:
        NostalgiaForInfinityX8.populate_indicators = original_populate
    cols = set(seen["cols"])
    expected = {"close_15m", "close_1h", "close_4h", "close_1d", "close_btc_4h"}
    missing = expected - cols
    assert not missing, f"NFI-X8 did not see expected informative columns: {missing}"
    # Run must not have crashed even if NFI-X8 produces zero signals on synthetic data
    assert isinstance(result.metrics.final_equity, float)


def test_x8_fails_closed_without_btc():
    data = generate_trend_with_cycles(n_candles=2000, seed=1)
    cfg = EngineConfig()
    r = BacktestEngine(cfg).run(data, NostalgiaForInfinityX8(), btc_data=None)
    assert any("BTC" in e for e in r.errors)


def test_x8_rejects_insufficient_warmup():
    data = generate_trend_with_cycles(n_candles=500)
    btc = generate_trend_with_cycles(n_candles=500, seed=2, symbol="BTC/USDT")
    cfg = EngineConfig()
    r = BacktestEngine(cfg).run(data, NostalgiaForInfinityX8(), btc_data=btc)
    assert any("Insufficient" in e or "insufficient" in e for e in r.errors), r.errors


def test_no_lookahead_on_btc_informative():
    """For a base candle at time t, close_btc_4h must come from a BTC candle that
    closed at or before t."""
    data = generate_trend_with_cycles(n_candles=2000, seed=1)
    btc = generate_trend_with_cycles(n_candles=2000, seed=2, symbol="BTC/USDT")
    btc_4h = resample_ohlcv(btc.df[["open", "high", "low", "close", "volume"]], "5m", "4h")
    merged = merge_informative(data.df, btc_4h, "btc_4h", "5m", "4h")
    delta_4h = pd.Timedelta(hours=4)
    # Sample points after warm-up
    for ts in merged.index[1000::173]:
        # Expected: last 4h candle whose close (start+4h) <= ts
        # The last candle starting at floor_h where (start + 4h) <= ts
        floor_4h = ts.floor("4h")
        # Candle starting at floor_4h covers [floor_4h, floor_4h+4h), closes at floor_4h+4h
        expected_start = floor_4h - delta_4h if ts < floor_4h + delta_4h else floor_4h
        if expected_start not in btc_4h.index:
            continue
        assert abs(merged.loc[ts, "close_btc_4h"] - btc_4h.loc[expected_start, "close"]) < 1e-6


def test_smacross_runs_end_to_end_with_engine_merged_columns():
    data = generate_trend_with_cycles(n_candles=1000, seed=4)
    cfg = EngineConfig(initial_cash=10000, fee_rate=0.001, slippage_bps=0)
    r = BacktestEngine(cfg).run(data, SmaCrossStrategy())
    assert r.metrics.closed_trades > 0
