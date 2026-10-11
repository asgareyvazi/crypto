"""Tests for informative-timeframe column wiring and no-lookahead semantics at engine level."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icbt.data import resample_ohlcv
from icbt.data.synthetic import generate_trend_with_cycles
from icbt.data.timeframes import merge_informative
from icbt.engine import BacktestEngine, EngineConfig
from icbt.strategy.indicators import EMA, RSI
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


# --------------------------------------------------------------------- informative-indicator correctness (no repeated-EMAs bug)


def test_x8_informative_ema_match_native_calculation():
    """ema_50_1h, ema_200_1h, ema_50_4h, ema_200_4h must equal EMA computed on
    native informative candles and then asof-merged, not EMA computed over the
    repeated-close tail on the 5m frame."""
    # Need enough data for 200-day EMA on the 1d frame (~300 days) plus
    # shorter warm-up for 4h/1h. 300 days * 24h * 12 candles/h = 86,400.
    n = 90000
    pair = generate_trend_with_cycles(n_candles=n, seed=11).df[["open", "high", "low", "close", "volume"]]
    s = NostalgiaForInfinityX8()

    # Independent reference: compute indicators on native candles ourselves.
    def _expected(pair_df, tf):
        inf = resample_ohlcv(pair_df, "5m", tf)
        inf[f"ema_50_{tf}"] = EMA(inf["close"], 50)
        inf[f"ema_200_{tf}"] = EMA(inf["close"], 200)
        return inf

    # Run through the strategy's native-TF hook (what the engine uses).
    enriched = pair.copy()
    for tf in s.inf_timeframes:
        raw = resample_ohlcv(pair, "5m", tf)
        inf = s.populate_informative_indicators(raw, tf)
        enriched = merge_informative(enriched, inf, tf, "5m", tf)

    # Compare each 5m row's informative EMAs against the asof-joined reference.
    for tf in s.inf_timeframes:
        ref = _expected(pair, tf)
        ref_merged = merge_informative(pair, ref, tf, "5m", tf)
        joined = enriched[[f"ema_50_{tf}", f"ema_200_{tf}"]].join(
            ref_merged[[f"ema_50_{tf}", f"ema_200_{tf}"]].rename(
                columns={f"ema_50_{tf}": "e50_ref", f"ema_200_{tf}": "e200_ref"}
            ),
            how="inner",
        ).dropna()
        assert len(joined) > 0, f"no comparable rows for {tf}"
        assert np.max(np.abs(joined[f"ema_50_{tf}"] - joined["e50_ref"])) < 1e-9
        assert np.max(np.abs(joined[f"ema_200_{tf}"] - joined["e200_ref"])) < 1e-9


def test_x8_informative_indicators_constant_between_informative_closes():
    """Between two consecutive informative candles, indicator values on the 5m
    frame must stay constant (they reflect the most recently closed informative
    candle until the next one closes)."""
    n = 9000
    pair = generate_trend_with_cycles(n_candles=n, seed=11).df[["open", "high", "low", "close", "volume"]]
    s = NostalgiaForInfinityX8()
    enriched = pair.copy()
    for tf in s.inf_timeframes:
        raw = resample_ohlcv(pair, "5m", tf)
        inf = s.populate_informative_indicators(raw, tf)
        enriched = merge_informative(enriched, inf, tf, "5m", tf)

    # Look at a region after warmup.
    sub = enriched.iloc[6000:7000]
    # Within each 1h bucket, ema_50_1h should not change.
    groups = sub.dropna(subset=["ema_50_1h"]).groupby(sub.dropna(subset=["ema_50_1h"]).index.floor("1h"))
    max_std = groups["ema_50_1h"].std().fillna(0).max()
    assert max_std == 0.0, f"ema_50_1h changed within a 1h bucket (max stdev={max_std})"

    # Within each 4h bucket, ema_50_4h should not change.
    sub4 = sub.dropna(subset=["ema_50_4h"])
    groups4 = sub4.groupby(sub4.index.floor("4h"))
    max_std4 = groups4["ema_50_4h"].std().fillna(0).max()
    assert max_std4 == 0.0


def test_x8_informative_indicators_no_lookahead_at_boundaries():
    """At an informative candle boundary, the new indicator value must not
    appear before the informative candle has actually closed.

    Semantics (see merge_informative): the informative candle starting at B
    represents the interval [B, B+tf); it CLOSES at B+tf - 1us, so it only
    becomes visible to base candles whose open time is >= B+tf. Base candles
    with open time in [B, B+tf) must still see the indicator value from the
    PREVIOUS informative candle (B-tf).

    We verify this by selecting a concrete 1h boundary from the synthetic data
    and asserting ema_50_1h equals the previous candle's value throughout the
    hour, then jumps to the new value at B+1h.
    """
    n = 9000
    pair = generate_trend_with_cycles(n_candles=n, seed=11).df[["open", "high", "low", "close", "volume"]].copy()
    s = NostalgiaForInfinityX8()
    raw_1h = resample_ohlcv(pair, "5m", "1h")
    inf = s.populate_informative_indicators(raw_1h, "1h")
    enriched = merge_informative(pair, inf, "1h", "5m", "1h")

    # Pick an hour boundary after warmup (EMA 50 needs 50h of history).
    warmup_1h = 60  # a bit past the 50 needed
    candidates = list(raw_1h.index[warmup_1h:-1])
    if not candidates:
        pytest.skip("not enough data for boundary test")
    boundary = candidates[20]  # arbitrary interior boundary
    prev_b = boundary - pd.Timedelta(hours=1)
    next_b = boundary + pd.Timedelta(hours=1)
    new_val_at_boundary = inf.loc[boundary, "ema_50_1h"]
    prev_val = inf.loc[prev_b, "ema_50_1h"]
    # Base candles with open in [boundary, next_b) must see prev_val.
    region = enriched.loc[(enriched.index >= boundary) & (enriched.index < next_b)]
    region = region.dropna(subset=["ema_50_1h"])
    assert len(region) > 0, "no rows in boundary region"
    assert np.all(np.abs(region["ema_50_1h"] - prev_val) < 1e-9), \
        "ema_50_1h jumped to new 1h value before that candle had closed (lookahead leak)"
    # First 5m candle AFTER next_b (i.e. the first bar of the following hour)
    # must have the new value (which was computed using candle 'boundary').
    if next_b in enriched.index:
        after = enriched.loc[next_b, "ema_50_1h"]
        if not pd.isna(after):
            assert abs(after - new_val_at_boundary) < 1e-9, \
                "ema_50_1h did not update to the new 1h value after the candle closed"


def test_x8_incomplete_informative_buckets_do_not_produce_fabricated_indicators():
    """When the resampled informative frame is missing candles (gaps), merge
    must not fabricate indicator values; those periods should remain NaN until
    a complete informative candle closes."""
    n = 5000
    pair = generate_trend_with_cycles(n_candles=n, seed=33).df[["open", "high", "low", "close", "volume"]].copy()
    s = NostalgiaForInfinityX8()
    # Build 1h with a deliberate gap (drop one hour's 1h candle).
    raw_1h = resample_ohlcv(pair, "5m", "1h")
    drop_ts = raw_1h.index[100]  # arbitrary hour after a little history
    raw_1h_gapped = raw_1h.drop(drop_ts)
    inf = s.populate_informative_indicators(raw_1h_gapped, "1h")
    enriched = merge_informative(pair, inf, "1h", "5m", "1h")
    # The 5m candles that fall inside the dropped 1h bucket should keep the
    # prior hour's values (backward fill), which is correct no-lookahead
    # behaviour; but crucially there must be NO row where ema_50_1h is
    # associated with a close_1h that doesn't exist in raw_1h_gapped (i.e.
    # we must not have re-invented close_1h for the missing hour).
    gap_open = drop_ts
    gap_close = drop_ts + pd.Timedelta(hours=1)
    during_gap = enriched.loc[(enriched.index >= gap_open) & (enriched.index < gap_close)]
    # close_1h during the gap must equal the close of the last complete hour
    # prior to the gap, NOT a fabricated candle.
    prior_ts = raw_1h_gapped.index[raw_1h_gapped.index < drop_ts][-1]
    assert bool((during_gap["close_1h"] == raw_1h_gapped.loc[prior_ts, "close"]).all()), \
        "missing informative bucket produced a fabricated close_1h rather than holding the prior value"


def test_x8_btc_indicators_match_native_calculation():
    """btc_ema_50_4h and btc_rsi_4h must match RSI/EMA computed on native BTC 4h
    candles, not on the repeated-close tail after merge."""
    n = 9000
    btc = generate_trend_with_cycles(n_candles=n, seed=22, symbol="BTC/USDT").df[["open", "high", "low", "close", "volume"]]
    s = NostalgiaForInfinityX8()
    raw4 = resample_ohlcv(btc, "5m", "4h")
    inf = s.populate_btc_informative_indicators(raw4, "4h")
    # Reference.
    ref = raw4.copy()
    ref["btc_ema_50_4h"] = EMA(ref["close"], 50)
    ref["btc_rsi_4h"] = RSI(ref["close"], 14)
    for col in ("btc_ema_50_4h", "btc_rsi_4h"):
        diff = np.max(np.abs(inf[col].dropna() - ref[col].dropna()))
        assert diff < 1e-9, f"{col} differs from native calculation (max diff={diff})"
    assert "btc_ema_50_4h" in inf.columns
    assert "btc_rsi_4h" in inf.columns


def test_engine_x8_exposes_informative_indicator_columns_not_just_close():
    """After engine.run, NFI-X8 must see not just close_1h but also the
    pre-computed ema_50_1h, ema_200_1h, btc_ema_50_4h, btc_rsi_4h columns."""
    data = generate_trend_with_cycles(n_candles=9000, seed=11)
    btc = generate_trend_with_cycles(n_candles=9000, seed=22, symbol="BTC/USDT")
    cfg = EngineConfig(initial_cash=10000, fee_rate=0, slippage_bps=0)
    eng = BacktestEngine(cfg)
    seen: dict[str, set] = {"cols": set()}
    original = NostalgiaForInfinityX8.populate_indicators

    def spy(self, df, metadata=None):
        seen["cols"] = set(df.columns)
        return original(self, df, metadata)

    NostalgiaForInfinityX8.populate_indicators = spy
    try:
        eng.run(data, NostalgiaForInfinityX8(), btc_data=btc)
    finally:
        NostalgiaForInfinityX8.populate_indicators = original
    required = {"ema_50_1h", "ema_200_1h", "ema_50_4h", "ema_200_4h", "ema_50_15m",
                "ema_200_15m", "ema_50_1d", "ema_200_1d", "btc_ema_50_4h", "btc_rsi_4h"}
    missing = required - seen["cols"]
    assert not missing, f"engine did not expose informative-indicator columns: {missing}"
