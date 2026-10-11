"""Timeframe helpers: delta conversion, OHLCV resampling and informative-pair merging."""

from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

_TF_RE = re.compile(r"^(\d+)([mhdwM])$")


class ResamplingError(ValueError):
    """Raised when resampling cannot produce complete target buckets."""


def timeframe_to_minutes(tf: str) -> int:
    """Return the nominal candle length for ``tf`` (e.g. ``"5m"`` -> 5, ``"1h"`` -> 60)."""
    return int(timeframe_to_timedelta(tf).total_seconds() // 60)


def timeframe_to_timedelta(tf: str) -> timedelta:
    m = _TF_RE.match(tf.strip())
    if not m:
        raise ValueError(f"Unrecognised timeframe: {tf!r}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    raise ValueError(f"Unsupported timeframe unit: {unit}")


def pandas_freq(tf: str) -> str:
    """Convert our timeframe notation to a pandas offset alias."""
    m = _TF_RE.match(tf.strip())
    if not m:
        raise ValueError(f"Unrecognised timeframe: {tf!r}")
    n, u = int(m.group(1)), m.group(2)
    unit_map = {"m": "min", "h": "h", "d": "D", "w": "W"}
    if n == 1:
        return unit_map[u]
    return f"{n}{unit_map[u]}"


def _expected_count(source_tf: str, target_tf: str) -> int:
    src = timeframe_to_timedelta(source_tf).total_seconds()
    dst = timeframe_to_timedelta(target_tf).total_seconds()
    if dst % src != 0:
        raise ValueError(f"Target timeframe {target_tf} is not an integer multiple of {source_tf}")
    return int(dst // src)


def _validate_source_grid(df: pd.DataFrame, source_tf: str) -> tuple[pd.DataFrame, int]:
    """Check alignment/order and return cleaned dataframe + count of duplicate timestamps removed."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ResamplingError("DataFrame must have a DatetimeIndex")
    work = df.copy()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    if not work.index.is_monotonic_increasing:
        raise ResamplingError("Source frame index is not monotonically increasing")
    dup_count = int(work.index.duplicated(keep="first").sum())
    work = work[~work.index.duplicated(keep="last")]
    src_delta = pd.Timedelta(timeframe_to_timedelta(source_tf))
    epoch = pd.Timestamp(0, tz="UTC")
    offsets = (work.index - epoch) % src_delta
    if (offsets != pd.Timedelta(0)).any():
        bad = int((offsets != pd.Timedelta(0)).sum())
        raise ResamplingError(
            f"{bad} source candles are not aligned to the {source_tf} grid"
        )
    return work, dup_count


def resample_ohlcv(
    df: pd.DataFrame,
    source_tf: str,
    target_tf: str,
    *,
    on_incomplete: str = "drop",
    require_full_buckets: bool = True,
) -> pd.DataFrame:
    """Aggregate OHLCV from ``source_tf`` into ``target_tf`` with strict completeness checks.

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by UTC DatetimeIndex with columns
        ``open, high, low, close, volume``.
    source_tf:
        Candle timeframe of the input, e.g. ``"5m"``.
    target_tf:
        Target timeframe for aggregation, e.g. ``"1h"``. Must be an exact
        integer multiple of ``source_tf``.
    on_incomplete:
        ``"drop"`` (default) silently drops incomplete buckets.
        ``"raise"`` raises :class:`ResamplingError` if any bucket is incomplete.
    require_full_buckets:
        When True (default) every kept target bucket must contain exactly the
        expected number of source candles and cover exactly one target period
        with no gaps or missing timestamps.

    Returns
    -------
    pd.DataFrame
        Resampled frame with DatetimeIndex at the **left** (open) edge of each
        COMPLETE target bucket in UTC, and columns
        ``open, high, low, close, volume, timestamp``.

    Notes
    -----
    * OHLCV aggregation is: open=first, high=max, low=min, close=last, volume=sum.
    * The first partial bucket (starting mid-period) and last partial bucket
      (ending mid-period) are always dropped.
    * No forward-filling or fabrication of missing candles is performed.
    * Duplicate timestamps are resolved by keeping the last row and count as one
      source candle (so the bucket will be flagged as incomplete).
    """
    if on_incomplete not in {"drop", "raise"}:
        raise ValueError("on_incomplete must be 'drop' or 'raise'")
    src = timeframe_to_timedelta(source_tf)
    dst = timeframe_to_timedelta(target_tf)
    if dst < src:
        raise ValueError(
            f"Cannot resample to a shorter timeframe ({target_tf} < {source_tf})"
        )
    if dst == src:
        out = df.copy()
        out["timestamp"] = out.index
        return out
    expected = _expected_count(source_tf, target_tf)
    work_clean, _dup_count = _validate_source_grid(df, source_tf)
    needed_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed_cols if c not in work_clean.columns]
    if missing:
        raise ResamplingError(f"Missing columns for resampling: {missing}")

    # Count duplicates per bucket BEFORE dedup so we can exclude buckets with duplicate timestamps.
    # The incoming df may not have been through validation yet, so do a safe pre-pass.
    work_raw = df.copy()
    if work_raw.index.tz is None:
        work_raw.index = work_raw.index.tz_localize("UTC")
    work_raw = work_raw.sort_index()
    dup_mask_raw = work_raw.index.duplicated(keep=False)
    buckets_with_dups = set()
    if dup_mask_raw.any():
        for ts in work_raw.index[dup_mask_raw]:
            buckets_with_dups.add(ts.floor(pandas_freq(target_tf)))

    work = work_clean[needed_cols]
    freq = pandas_freq(target_tf)

    # Aggregate OHLCV per bucket.
    agg = work.resample(freq, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    # Count source candles per bucket.
    counts = work.resample(freq, label="left", closed="left").size()

    # A bucket is complete iff:
    #   1. count == expected source candles
    #   2. all OHLCV columns are non-NaN (first/max/min/last/sum produced a value)
    #   3. the bucket does NOT contain any duplicate source timestamps
    complete_mask = (
        (counts == expected)
        & agg.notna().all(axis=1)
        & ~agg.index.isin(buckets_with_dups)
    )

    if on_incomplete == "raise":
        incomplete = agg[~complete_mask]
        # First and last buckets are expected to be partial; only raise for internal gaps.
        internal = incomplete.iloc[1:-1] if len(incomplete) > 2 else incomplete.iloc[0:0]
        if len(internal) > 0:
            raise ResamplingError(
                f"{len(internal)} incomplete {target_tf} bucket(s) inside the series "
                f"(expected {expected} source candles per bucket)."
            )

    if require_full_buckets:
        agg = agg[complete_mask]

    agg["timestamp"] = agg.index
    return agg


def merge_informative(
    base: pd.DataFrame,
    informative: pd.DataFrame,
    suffix: str,
    base_tf: str,
    inf_tf: str,
) -> pd.DataFrame:
    """Merge an informative (higher) timeframe onto the base (lower) timeframe.

    The merge guarantees **no lookahead**: for every base candle the attached
    informative values come from the informative candle whose closing time is
    strictly <= the base candle's open time (the Freqtrade
    ``merge_informative_pair`` convention). Informative columns are suffixed
    with ``f"_{suffix}"``.
    """
    if base_tf != "5m" and timeframe_to_minutes(base_tf) >= timeframe_to_minutes(inf_tf):
        raise ValueError(
            f"Informative timeframe {inf_tf} must be strictly longer than base timeframe {base_tf}"
        )
    col_suffix = suffix if suffix.startswith("_") else f"_{suffix}"

    inf = informative.copy()
    cols = [c for c in inf.columns if c != "timestamp"]
    inf = inf[cols]
    inf = inf.rename(columns={c: f"{c}{col_suffix}" for c in cols})

    delta_inf = pd.Timedelta(timeframe_to_timedelta(inf_tf))
    base_sorted = base.sort_index()
    # Shift the informative index to represent each candle by its close - 1us so
    # merge_asof(..., direction='backward') matches the last candle fully closed
    # by the base candle's timestamp.
    inf_shifted = inf.copy()
    inf_shifted.index = inf.index + delta_inf - pd.Timedelta(microseconds=1)
    merged = pd.merge_asof(
        base_sorted,
        inf_shifted,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged
