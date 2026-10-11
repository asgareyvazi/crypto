import tempfile
from pathlib import Path

import pandas as pd
import pytest

from icbt.data import OHLCV, load_csv, manifest_hash, write_csv
from icbt.data.synthetic import generate_trend_with_cycles
from icbt.data.timeframes import (
    ResamplingError,
    merge_informative,
    resample_ohlcv,
    timeframe_to_minutes,
    timeframe_to_timedelta,
)
from icbt.data.validation import Severity, validate_ohlcv


def test_timeframe_conversions():
    assert timeframe_to_minutes("5m") == 5
    assert timeframe_to_minutes("1h") == 60
    assert timeframe_to_minutes("4h") == 240
    assert timeframe_to_minutes("1d") == 1440
    assert timeframe_to_timedelta("15m").total_seconds() == 900


def test_roundtrip_csv():
    with tempfile.TemporaryDirectory() as d:
        data = generate_trend_with_cycles(n_candles=300)
        p = Path(d) / "candles_synth_btcusdt_5m.csv"
        write_csv(data, p)
        loaded = load_csv(p)
        assert len(loaded) == len(data)
        assert loaded.exchange == "synth"
        assert loaded.symbol == "BTCUSDT"
        assert loaded.timeframe == "5m"
        assert loaded.metadata.source_sha256 is not None
        # explicit overrides work
        loaded2 = load_csv(p, exchange="wallex", symbol="BTC/USDT", timeframe="5m")
        assert loaded2.exchange == "wallex"
        assert loaded2.symbol == "BTC/USDT"


def test_validate_clean_dataset():
    data = generate_trend_with_cycles(n_candles=500)
    r = validate_ohlcv(data)
    assert r.ok
    assert r.invalid_ohlc_rows == 0
    assert r.duplicates == 0
    assert len(r.gaps) == 0


def test_validate_detects_invalid_ohlc():
    data = generate_trend_with_cycles(n_candles=200)
    df = data.df.copy()
    df.loc[df.index[10], "high"] = df.loc[df.index[10], "low"] - 1  # high < low
    md = data.metadata
    bad = OHLCV(df=df, metadata=md)
    r = validate_ohlcv(bad)
    assert not r.ok
    assert r.invalid_ohlc_rows == 1


def test_validate_detects_duplicates():
    data = generate_trend_with_cycles(n_candles=200)
    df = data.df.copy()
    dup = df.iloc[[5]].copy()
    df = pd.concat([df, dup]).sort_index()
    md = data.metadata
    bad = OHLCV(df=df, metadata=md)
    r = validate_ohlcv(bad)
    assert r.duplicates >= 1
    assert not r.ok


def test_validate_detects_gaps():
    data = generate_trend_with_cycles(n_candles=200)
    df = data.df.drop(data.df.index[50:55])
    md = data.metadata
    gapped = OHLCV(df=df, metadata=md)
    r = validate_ohlcv(gapped)
    assert len(r.gaps) >= 1
    assert any(i.code == "GAPS" for i in r.issues)
    # Gaps are warnings (not hard errors) by default
    assert r.ok or any(i.severity == Severity.ERROR for i in r.issues) is False


def test_resample_5m_to_1h_drops_partial_buckets():
    """Partial first/last buckets are dropped; only complete 12-candle hourly buckets remain."""
    # 126 candles = 10 full hours (00-09) + 6 extra in hour 10 (incomplete, dropped).
    n = 126
    d2 = generate_trend_with_cycles(n_candles=n, seed=1)
    h1 = resample_ohlcv(d2.df[["open", "high", "low", "close", "volume"]], "5m", "1h")
    # Expected: 10 complete buckets (hours 00:00-09:00), hour 10 is partial (only 6 candles) -> dropped
    assert len(h1) == 10, f"expected 10 complete hourly buckets, got {len(h1)}"
    assert h1.index[0] == d2.df.index[0]
    assert (h1["high"] >= h1[["open", "close"]].max(axis=1)).all()
    assert (h1["low"] <= h1[["open", "close"]].min(axis=1)).all()
    # Each bucket must have EXACTLY 12 candles by checking counts
    counts = d2.df.resample("1h", label="left", closed="left").size()
    complete = counts[counts == 12]
    assert list(h1.index) == list(complete.index), "complete bucket timestamps must match"


def test_resample_5m_to_1h_excludes_internal_gap_buckets():
    """If source candles are missing from the middle of an hour, that bucket must be dropped."""
    data = generate_trend_with_cycles(n_candles=200, seed=5)
    df = data.df[["open", "high", "low", "close", "volume"]].copy()
    # Drop 2 candles from hour index 5 (candles at 05:05 and 05:10)
    gap_time = df.index[0] + pd.Timedelta(hours=5)
    drop_idx = [gap_time + pd.Timedelta(minutes=5), gap_time + pd.Timedelta(minutes=10)]
    df_gapped = df.drop(drop_idx)
    h1 = resample_ohlcv(df_gapped, "5m", "1h")
    # The bucket at 05:00 should be missing (incomplete)
    assert gap_time not in h1.index, "bucket with internal gap must be excluded"
    # Other complete buckets should still be present.
    assert df.index[0] in h1.index


def test_resample_4h_correct_bucket_count():
    data = generate_trend_with_cycles(n_candles=48 * 12 + 7, seed=3)  # 48 hours = 576 5m candles + 7 extra (partial last)
    h4 = resample_ohlcv(data.df[["open", "high", "low", "close", "volume"]], "5m", "4h")
    assert len(h4) == 12  # 48 hours / 4 per hour = 12 complete 4h buckets


def test_resample_rejects_off_grid_source():
    data = generate_trend_with_cycles(n_candles=120)
    df = data.df[["open", "high", "low", "close", "volume"]].copy()
    bad = df.copy()
    bad.index = bad.index + pd.Timedelta(minutes=1)  # misalign by 1 minute
    with pytest.raises(ResamplingError):
        resample_ohlcv(bad, "5m", "1h")


def test_resample_deduplicates_source():
    data = generate_trend_with_cycles(n_candles=120)
    df = data.df[["open", "high", "low", "close", "volume"]].copy()
    dup = df.iloc[[10]].copy()
    df_with_dup = pd.concat([df, dup]).sort_index()
    h1 = resample_ohlcv(df_with_dup, "5m", "1h")
    # The hour containing the duplicate row now has 13 rows (one duplicate), so it should be dropped.
    # That hour is at index[10] -> floor to hour.
    dup_hour = df.index[10].floor("1h")
    assert dup_hour not in h1.index, "bucket with duplicate timestamp must be considered incomplete"


def test_resample_preserves_tz_and_alignment():
    data = generate_trend_with_cycles(n_candles=240)
    h1 = resample_ohlcv(data.df[["open", "high", "low", "close", "volume"]], "5m", "1h")
    assert str(h1.index.tz) == "UTC"
    assert (h1.index.minute == 0).all()
    assert (h1.index.second == 0).all()


def test_merge_informative_no_lookahead():
    """Ensure informative value at base candle t uses only candles closed by t."""
    data = generate_trend_with_cycles(n_candles=500)
    h1 = resample_ohlcv(data.df[["open", "high", "low", "close", "volume"]], "5m", "1h")
    merged = merge_informative(data.df, h1, "1h", "5m", "1h")
    assert "close_1h" in merged.columns
    # For every base candle after warm-up, close_1h must equal close of last 1h candle whose
    # closing edge is <= base candle's timestamp.
    import pandas as pd

    delta_h = pd.Timedelta(hours=1)
    for ts in merged.index[200:500:13]:  # sample points
        ts - pd.Timedelta(minutes=(ts.minute + 5) % 60) - pd.Timedelta(minutes=60 - 5)
        # Compute correctly: aligned previous hour boundary minus offset -> last fully-closed hour.
        floor = ts.floor("1h")
        last_closed_start = floor - delta_h if ts < floor + delta_h else floor
        # If ts is exactly on hour boundary (minute==0), the previous hour closed at ts.
        if ts.minute == 0 and ts.second == 0:
            last_closed_start = ts - delta_h
        expected_close = h1.loc[last_closed_start, "close"]
        actual = merged.loc[ts, "close_1h"]
        assert abs(actual - expected_close) < 1e-6, (ts, last_closed_start, actual, expected_close)


def test_manifest_hash_changes_with_content():
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.csv"
        p2 = Path(d) / "b.csv"
        d1 = generate_trend_with_cycles(n_candles=10, seed=1)
        d2 = generate_trend_with_cycles(n_candles=10, seed=2)
        write_csv(d1, p1)
        write_csv(d2, p2)
        assert manifest_hash(p1) != manifest_hash(p2)
