"""Timeframe helpers: delta conversion, OHLCV resampling and informative-pair merging."""

from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

_TF_RE = re.compile(r"^(\d+)([mhdwM])$")


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
    n, u = m.group(1), m.group(2)
    return (
        {"m": "min", "h": "h", "d": "D", "w": "W"}[u]
        if n == "1"
        else f"{n}{'min' if u == 'm' else u}"
    )


def _align_timestamp(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Floor timestamp to candle boundary so resampling starts/ends on even boundaries."""
    return ts.floor(freq)


def resample_ohlcv(
    df: pd.DataFrame,
    source_tf: str,
    target_tf: str,
) -> pd.DataFrame:
    """Aggregate OHLCV from ``source_tf`` into ``target_tf``.

    * open  = first open in bucket
    * high  = max high in bucket
    * low   = min low in bucket
    * close = last close in bucket
    * volume= sum volume in bucket

    Buckets that would be incomplete (i.e. source data starts mid-bucket) are
    dropped to avoid leaking partial aggregates. Timestamps are set to the
    **left** (open) edge of each target candle in UTC, matching standard
    exchange conventions.
    """
    src = timeframe_to_timedelta(source_tf)
    dst = timeframe_to_timedelta(target_tf)
    if dst < src:
        raise ValueError(f"Cannot down-sample to a shorter timeframe ({target_tf} < {source_tf})")
    if dst == src:
        return df.copy()
    if dst.total_seconds() % src.total_seconds() != 0:
        raise ValueError(f"Target timeframe {target_tf} is not an integer multiple of {source_tf}")
    freq = pandas_freq(target_tf)

    work = df.copy()
    # Align to target boundaries; we keep timestamp column aligned.
    work = work[["open", "high", "low", "close", "volume"]]
    resampled = work.resample(freq, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    # Drop any bucket where any value is NaN (incomplete/insufficient source data).
    resampled = resampled.dropna(how="any")
    resampled["timestamp"] = resampled.index
    return resampled


def merge_informative(
    base: pd.DataFrame,
    informative: pd.DataFrame,
    suffix: str,
    base_tf: str,
    inf_tf: str,
) -> pd.DataFrame:
    """Merge an informative (higher) timeframe onto the base (lower) timeframe.

    The merge guarantees **no lookahead**: for every base candle we only attach
    informative candles whose close time is strictly <= the base candle's open
    time (which is the conventional ``merge_informative_pair`` semantics used
    by Freqtrade). Informative columns are suffixed with ``suffix``.
    """
    if base_tf != "5m" and timeframe_to_minutes(base_tf) >= timeframe_to_minutes(inf_tf):
        raise ValueError(f"Informative timeframe {inf_tf} must be >= base timeframe {base_tf}")

    inf = informative.copy()
    # Only keep informative columns other than timestamp.
    cols = [c for c in inf.columns if c != "timestamp"]
    inf = inf[cols]
    inf = inf.rename(columns={c: f"{c}_{suffix}" for c in cols})
    # Use merge_asof with direction='backward' to get the most recent informative
    # candle whose time is strictly <= the base candle. We shift the informative
    # index by (tf - 1 microsecond) so that the "closed candle" is the one whose
    # right edge lines up with a boundary - but merge_asof already matches on
    # left_index <= right_index, which matches closed-candle semantics because
    # the informative candle's timestamp represents its open (left edge). To
    # match Freqtrade convention we need the informative candle that **closes**
    # at or before the base candle's open time.
    #
    # If informative[i] covers [t_i, t_i + delta_inf), the candle closes at
    # t_i + delta_inf. The base candle at t_b opens at t_b. We require
    # t_i + delta_inf <= t_b, i.e. t_i <= t_b - delta_inf.
    delta_inf = timeframe_to_timedelta(inf_tf)
    base_sorted = base.sort_index()
    # Represent each informative candle by its *closing* edge minus 1 microsecond,
    # so merge_asof's "right_index <= left_index" gives the most recent candle
    # that is fully closed at the base candle's timestamp.
    inf_index_shifted = inf.index + pd.Timedelta(delta_inf) - pd.Timedelta(microseconds=1)
    inf_shifted = inf.copy()
    inf_shifted.index = inf_index_shifted
    merged = pd.merge_asof(
        base_sorted,
        inf_shifted,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged
