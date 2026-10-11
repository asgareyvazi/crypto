"""Unit tests for scripts/download_wallex.py.

These tests cover the 1m->5m aggregation logic and the full download() flow
for 5m (using a mocked fetch_chunk). They do not access the network.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_wallex as dw  # noqa: E402


def _make_1m_frame(
    start: datetime,
    n_minutes: int,
    *,
    open_start: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a deterministic n_minutes-length 1m OHLCV frame starting at ``start``."""
    import numpy as np
    rng = np.random.default_rng(seed)
    times = [start + timedelta(minutes=i) for i in range(n_minutes)]
    closes = open_start + np.cumsum(rng.normal(0, 0.5, n_minutes))
    opens = [open_start] + list(closes[:-1])
    highs = [max(o, c) + abs(rng.normal(0, 0.3)) for o, c in zip(opens, closes, strict=False)]
    lows = [min(o, c) - abs(rng.normal(0, 0.3)) for o, c in zip(opens, closes, strict=False)]
    vols = rng.uniform(0.5, 5.0, n_minutes)
    df = pd.DataFrame({
        "timestamp": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })
    df.index = pd.DatetimeIndex(times, tz="UTC", name="date")
    return df


# --------------------------------------------------------------------- aggregation tests


def test_aggregate_ohlcv_correct_per_bucket_values():
    """For 5 complete 5m buckets, open=first-minute open, high=max(highs),
    low=min(lows), close=last close, volume=sum."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for b in range(5):
        b0 = start + timedelta(minutes=5 * b)
        for m in range(5):
            ts = b0 + timedelta(minutes=m)
            rows.append({
                "timestamp": ts,
                "open": 100 + b * 10 + (1 if m == 0 else 0),  # open of minute 0 is 100+b*10+1
                "high": 102 + b * 10 + m,                     # highest = minute 4 -> 106+b*10
                "low": 98 + b * 10 - m,                       # lowest  = minute 4 -> 94+b*10
                "close": 100 + b * 10 + m,
                "volume": 1.0 + m,                            # sum over m=0..4 = 5+10=15? wait 1+2+3+4+5=15
            })
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    result = dw.aggregate_1m_to_5m(df, start, start + timedelta(minutes=20))
    assert result.complete_buckets == 5
    assert result.dropped_buckets_missing == 0
    out = result.df
    # Bucket 0 open @ 00:00 -> first open = 101 (m=0), high=106 (m=4), low=94 (m=4),
    # close=104 (m=4), volume=1+2+3+4+5=15
    assert out.iloc[0]["open"] == 101
    assert out.iloc[0]["high"] == 106
    assert out.iloc[0]["low"] == 94
    assert out.iloc[0]["close"] == 104
    assert out.iloc[0]["volume"] == 15
    # Timestamps aligned to 5-minute boundaries.
    for ts in out.index:
        assert ts.minute % 5 == 0 and ts.second == 0 and ts.microsecond == 0


def test_aggregate_drops_bucket_with_missing_minute():
    """If one minute out of five is missing, that bucket must be dropped
    (not fabricated, not partial)."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    df = _make_1m_frame(start, 15, seed=1)  # 3 buckets x 5 minutes
    # Drop minute 7 (second bucket, minute 2 -> bucket 00:05 incomplete).
    df = df.drop(df.index[7])
    result = dw.aggregate_1m_to_5m(df, start, start + timedelta(minutes=10))
    assert result.complete_buckets == 2  # buckets 00:00 and 00:10
    assert result.dropped_buckets_missing == 1
    assert pd.Timestamp("2025-01-01 00:05", tz="UTC") not in result.df.index
    assert pd.Timestamp("2025-01-01 00:00", tz="UTC") in result.df.index
    assert pd.Timestamp("2025-01-01 00:10", tz="UTC") in result.df.index


def test_aggregate_alignment_to_utc_5m_boundaries():
    """Output index must land exactly on HH:00, HH:05, HH:10, ... in UTC,
    regardless of where the input starts (we test by providing 1m data
    slightly off-grid and verifying alignment is to 5m buckets)."""
    # Provide data from HH:01 for several minutes; no complete 5m bucket
    # containing HH:00..HH:04 should appear, and the first complete bucket
    # should be HH:05 (which needs HH:05..HH:09).
    start = datetime(2025, 3, 15, 12, 0, tzinfo=UTC)
    rows = []
    # 20 minutes starting at 12:01 -> minutes 12:01..12:20
    for i in range(20):
        ts = start + timedelta(minutes=1 + i)
        rows.append({
            "timestamp": ts,
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100.5 + i,
            "volume": 1.0,
        })
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    result = dw.aggregate_1m_to_5m(
        df,
        datetime(2025, 3, 15, 12, 5, tzinfo=UTC),
        datetime(2025, 3, 15, 12, 15, tzinfo=UTC),
    )
    # Complete buckets present: 12:05 (needs 05,06,07,08,09) and 12:10 (10..14).
    # 12:00 bucket needs 00..04 but we're missing 00,01-> dropped (we don't even
    # ask for it). 12:15 needs 15..19 which is present in input (15,16,17,18,19
    # are all in range i=14..18); 12:20 needs 20..24 but we only have 20.
    assert result.complete_buckets == 3
    expected = [
        pd.Timestamp("2025-03-15 12:05", tz="UTC"),
        pd.Timestamp("2025-03-15 12:10", tz="UTC"),
        pd.Timestamp("2025-03-15 12:15", tz="UTC"),
    ]
    assert list(result.df.index) == expected


def test_aggregate_zero_volume_candles_are_preserved():
    """Zero-volume minutes are valid (illiquid periods); a bucket of five
    zero-volume minutes should aggregate with volume=0 (not NaN, not dropped)."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for m in range(5):
        ts = start + timedelta(minutes=m)
        rows.append({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 0.0})
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    result = dw.aggregate_1m_to_5m(df, start, start)
    assert result.complete_buckets == 1
    assert result.df.iloc[0]["volume"] == 0.0
    assert result.df.iloc[0]["high"] == 101
    assert result.df.iloc[0]["low"] == 99


def test_aggregate_duplicate_timestamps_last_write_wins():
    """Duplicate 1m timestamps: last-write-wins; count reported in duplicates_1m."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for m in range(5):
        ts = start + timedelta(minutes=m)
        rows.append({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1.0})
    # Duplicate minute 2 with a different close; last-write-wins -> close=200.
    rows.append({"timestamp": start + timedelta(minutes=2), "open": 150, "high": 210, "low": 99, "close": 200, "volume": 5.0})
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    result = dw.aggregate_1m_to_5m(df, start, start)
    assert result.duplicates_1m == 1
    assert result.complete_buckets == 1
    # High should reflect the higher of the duplicates (210 from the duplicate).
    assert result.df.iloc[0]["high"] == 210
    # The duplicate replaced the original minute 2, so volume reflects the duplicate (1+1+5+1+1 = 9).
    assert result.df.iloc[0]["volume"] == 9


def test_aggregate_final_candle_at_end_of_range_is_included():
    """For a requested end of 23:55 (5m bucket open), the bucket [23:55..24:00)
    MUST be included when all five 1m candles are present (23:55..23:59)."""
    start = datetime(2025, 1, 1, 23, 50, tzinfo=UTC)
    # 10 minutes: 23:50..23:59 covers two 5m buckets (23:50 and 23:55).
    rows = []
    for m in range(10):
        ts = start + timedelta(minutes=m)
        rows.append({"timestamp": ts, "open": 100 + m, "high": 102 + m, "low": 98 + m, "close": 101 + m, "volume": 1.0})
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    end_bucket = datetime(2025, 1, 1, 23, 55, tzinfo=UTC)
    result = dw.aggregate_1m_to_5m(df, start, end_bucket)
    assert result.complete_buckets == 2
    assert pd.Timestamp(end_bucket) in result.df.index
    # That last bucket's close should be the 23:59 minute close (=101+9=110).
    last = result.df.loc[pd.Timestamp(end_bucket)]
    assert last["close"] == 110


def test_aggregate_rejects_tz_naive_index():
    df = _make_1m_frame(datetime(2025, 1, 1, 0, 0), 5)
    df = df.tz_localize(None)
    with pytest.raises(ValueError):
        dw.aggregate_1m_to_5m(df, df.index[0], df.index[-1])


def test_aggregate_off_grid_minutes_excluded_and_counted():
    """1m candles with seconds != 0 (e.g. seconds=30) are off-grid, excluded,
    and counted."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for m in range(6):
        ts = start + timedelta(minutes=5, seconds=30) if m == 5 else start + timedelta(minutes=m)
        rows.append({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1.0})
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    result = dw.aggregate_1m_to_5m(df, start, start)
    assert result.off_grid_1m == 1
    # The off-grid candle is not a valid minute 5; the first bucket (00:00) needs
    # minutes 0-4 which are present -> complete.
    assert result.complete_buckets == 1


# --------------------------------------------------------------------- download() flow (5m)


def _udf_ok(t_arr, o_arr, h_arr, l_arr, c_arr, v_arr):
    return {"s": "ok", "t": t_arr, "o": o_arr, "h": h_arr, "l": l_arr, "c": c_arr, "v": v_arr}


def test_download_5m_fetches_1m_and_aggregates_with_mock(tmp_path):
    """Inject a fake fetch_chunk, request 3 candles of 5m starting at 00:00.
    The mock returns exactly 3*5=15 1m candles covering 00:00..00:14.
    Verify the output CSV has 3 rows at 00:00, 00:05, 00:10 and correct stats."""
    calls: list[tuple[str, int, int]] = []

    def fake_fetch(symbol, resolution, start_s, end_s, timeout=20):
        calls.append((resolution, start_s, end_s))
        assert resolution == "1", f"5m path must request resolution=1, got {resolution!r}"
        # Return 1m candles [start_s, end_s] inclusive (one per minute).
        n_minutes = (end_s - start_s) // 60 + 1
        t_arr = [start_s + 60 * i for i in range(n_minutes)]
        o_arr = [100 + i * 0.1 for i in range(n_minutes)]
        h_arr = [v + 1 for v in o_arr]
        l_arr = [v - 1 for v in o_arr]
        c_arr = [100 + i * 0.1 + 0.05 for i in range(n_minutes)]
        v_arr = [1.0 for _ in range(n_minutes)]
        return _udf_ok(t_arr, o_arr, h_arr, l_arr, c_arr, v_arr)

    out = tmp_path / "btc_5m.csv"
    stats = dw.download(
        symbol="BTCUSDT",
        timeframe="5m",
        start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 1, 0, 10, tzinfo=UTC),
        output_path=out,
        fetch_chunk_fn=fake_fetch,
    )
    assert stats.resolution == "5m"
    assert stats.expected == 3
    assert stats.received == 3
    assert stats.missing == 0
    assert stats.duplicates == 0
    # Mock was called with resolution=1.
    assert all(c[0] == "1" for c in calls)
    # The fetch end must cover the final bucket through T+4m (00:14Z) because
    # we requested end=00:10 (bucket open); candles 00:10..00:14 are needed.
    final_call_end = calls[-1][2]
    assert final_call_end == int(datetime(2025, 1, 1, 0, 14, tzinfo=UTC).timestamp())

    csv_df = pd.read_csv(out)
    assert len(csv_df) == 3
    # ISO timestamps should be at 00:00, 00:05, 00:10 UTC.
    assert "2025-01-01T00:00:00+00:00" in csv_df["timestamp"].iloc[0] or \
           "2025-01-01 00:00:00+00:00" in csv_df["timestamp"].iloc[0]
    # Volume sums per bucket should be 5.0 (five 1.0 minutes).
    for v in csv_df["volume"]:
        assert v == pytest.approx(5.0)


def test_download_5m_missing_minute_counts_as_missing_and_drops_bucket(tmp_path):
    """If one 1m minute is missing from the middle of a 5m bucket, that bucket
    must be excluded AND reported as a missing 5m candle in stats.missing."""

    def fake_fetch(symbol, resolution, start_s, end_s, timeout=20):
        n_minutes = (end_s - start_s) // 60 + 1
        t_arr = []
        for i in range(n_minutes):
            ts = start_s + 60 * i
            # Drop minute 7 (1m candle at 00:07) -> bucket 00:05 becomes incomplete.
            if ts == int(datetime(2025, 1, 1, 0, 7, tzinfo=UTC).timestamp()):
                continue
            t_arr.append(ts)
        o_arr = [100.0] * len(t_arr)
        h_arr = [101.0] * len(t_arr)
        l_arr = [99.0] * len(t_arr)
        c_arr = [100.0] * len(t_arr)
        v_arr = [1.0] * len(t_arr)
        return _udf_ok(t_arr, o_arr, h_arr, l_arr, c_arr, v_arr)

    out = tmp_path / "btc_5m.csv"
    stats = dw.download(
        symbol="BTCUSDT",
        timeframe="5m",
        start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 1, 0, 10, tzinfo=UTC),
        output_path=out,
        fetch_chunk_fn=fake_fetch,
    )
    # Expected 3 buckets (00:00, 00:05, 00:10). Bucket 00:05 missing one minute -> dropped.
    assert stats.expected == 3
    assert stats.received == 2
    assert stats.missing == 1
    csv_df = pd.read_csv(out)
    assert len(csv_df) == 2


def test_download_5m_uses_resolution_1_not_5(tmp_path):
    """Regression: ensure the Wallex call never uses resolution=5 (which returns
    'Invalid resolution!')."""
    seen_resolutions: list[str] = []

    def fake_fetch(symbol, resolution, start_s, end_s, timeout=20):
        seen_resolutions.append(resolution)
        n_minutes = (end_s - start_s) // 60 + 1
        t_arr = [start_s + 60 * i for i in range(n_minutes)]
        return _udf_ok(
            t_arr,
            [100.0] * len(t_arr),
            [101.0] * len(t_arr),
            [99.0] * len(t_arr),
            [100.0] * len(t_arr),
            [1.0] * len(t_arr),
        )

    out = tmp_path / "btc_5m.csv"
    dw.download(
        symbol="BTCUSDT",
        timeframe="5m",
        start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),  # two 5m buckets
        output_path=out,
        fetch_chunk_fn=fake_fetch,
    )
    assert "5" not in seen_resolutions
    assert set(seen_resolutions) == {"1"}


def test_download_non_5m_timeframe_uses_native_resolution(tmp_path):
    """Timeframes other than 5m must still use their original Wallex resolution
    (no silent behavior change)."""
    seen: list[tuple[str, int, int]] = []

    def fake_fetch(symbol, resolution, start_s, end_s, timeout=20):
        seen.append((resolution, start_s, end_s))
        # return 3 hourly candles
        step = 3600
        n = (end_s - start_s) // step + 1
        t_arr = [start_s + step * i for i in range(n)]
        return _udf_ok(
            t_arr,
            [100.0] * len(t_arr),
            [101.0] * len(t_arr),
            [99.0] * len(t_arr),
            [100.0] * len(t_arr),
            [1.0] * len(t_arr),
        )

    out = tmp_path / "btc_1h.csv"
    stats = dw.download(
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 1, 2, 0, tzinfo=UTC),
        output_path=out,
        fetch_chunk_fn=fake_fetch,
    )
    assert stats.resolution == "1h"
    assert stats.expected == 3
    assert all(r == "60" for r, _, _ in seen)


# --------------------------------------------------------------------- pandas datetime unit regression


def test_records_to_frame_and_aggregation_handle_microsecond_resolution_index(tmp_path):
    """Regression: when pandas builds a DatetimeIndex at microsecond
    resolution (``datetime64[us, UTC]``) — as newer pandas does by default —
    epoch-seconds extraction must not assume nanosecond storage. With 15
    valid minute-aligned candles (00:00..00:14 UTC) we must see
    ``off_grid == 0``, ``duplicates == 0``, and three complete 5m buckets.
    """
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    end_1m = start + timedelta(minutes=14)
    records: dict[int, dict] = {}
    for i in range(15):
        ts = start + timedelta(minutes=i)
        records[int(ts.timestamp())] = {
            "timestamp": ts,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1.0,
        }

    df, dup, og = dw._records_to_frame(
        records, step_s=60,
        start_s=int(start.timestamp()),
        end_s=int(end_1m.timestamp()),
    )
    assert dup == 0
    assert og == 0, f"expected 0 off-grid with native index, got {og}"
    assert len(df) == 15

    # Force the index to microsecond resolution to simulate pandas versions
    # where DatetimeIndex defaults to us.
    df_us = df.copy()
    df_us.index = df_us.index.astype("datetime64[us, UTC]")
    assert str(df_us.index.dtype) == "datetime64[us, UTC]"

    # Re-implement the epoch-second conversion the same way _records_to_frame
    # does after the fix and confirm 0 off-grid.
    epoch_ns = df_us.index.astype("datetime64[ns, UTC]").view("int64")
    epoch = epoch_ns // 10**9
    on_grid = (epoch % 60 == 0) & (epoch >= int(start.timestamp())) & (epoch <= int(end_1m.timestamp()))
    assert int((~on_grid).sum()) == 0, "us-resolution index must produce 0 off-grid counts"

    # Aggregation on the us-resolution frame must produce 3 complete 5m buckets.
    agg = dw.aggregate_1m_to_5m(df_us, start, start + timedelta(minutes=10))
    assert agg.complete_buckets == 3
    assert agg.dropped_buckets_missing == 0
    assert agg.duplicates_1m == 0
    assert agg.off_grid_1m == 0
    assert list(agg.df.index) == [
        pd.Timestamp("2025-01-01 00:00", tz="UTC"),
        pd.Timestamp("2025-01-01 00:05", tz="UTC"),
        pd.Timestamp("2025-01-01 00:10", tz="UTC"),
    ]


def test_download_5m_us_index_via_main_returns_exit_zero(tmp_path, monkeypatch):
    """End-to-end CLI path: ``main(argv)`` returns exit code 0 when the
    (mocked) API returns all expected candles. This simulates a short
    live download without hitting the network; exit code 0 asserts the
    post-fetch validation (missing/off-grid/dup) all report zero."""
    import io

    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    end_open = datetime(2025, 1, 1, 0, 10, tzinfo=UTC)  # three 5m buckets

    def fake_fetch(symbol, resolution, start_s, end_s, timeout=20):
        assert resolution == "1"
        n = (end_s - start_s) // 60 + 1
        t_arr = [start_s + 60 * i for i in range(n)]
        return {
            "s": "ok",
            "t": t_arr,
            "o": [100.0] * len(t_arr),
            "h": [101.0] * len(t_arr),
            "l": [99.0] * len(t_arr),
            "c": [100.0] * len(t_arr),
            "v": [1.0] * len(t_arr),
        }

    out = tmp_path / "btc_5m.csv"
    monkeypatch.setattr(dw, "fetch_chunk", fake_fetch)
    monkeypatch.setattr("sys.stdout", io.StringIO())
    rc = dw.main([
        "--symbol", "BTCUSDT",
        "--timeframe", "5m",
        "--start", start.isoformat().replace("+00:00", "Z"),
        "--end", end_open.isoformat().replace("+00:00", "Z"),
        "--output", str(out),
    ])
    assert rc == 0, f"expected exit code 0 (clean data), got {rc}"
    assert len(pd.read_csv(out)) == 3

