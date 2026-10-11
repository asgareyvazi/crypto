"""Wallex market-data downloader (UDF/TradingView-compatible history endpoint).

Wallex public API: GET https://api.wallex.ir/v1/udf/history

Parameters (per official docs https://api-docs.wallex.ir):
    symbol      e.g. BTCUSDT, BTCTMN
    resolution  one of: 1, 5, 15, 30, 60, 120, 180, 240, 360, 720, "D", "W", "M"
                (numeric values are MINUTES; D/W/M are daily/weekly/monthly).
                NOTE: As of 2026 the Wallex endpoint returns "Invalid resolution!"
                for resolution=5 even though it is documented. For timeframe 5m
                we therefore request resolution=1 (one-minute candles) and
                locally aggregate into strict UTC-aligned 5-minute candles,
                dropping any bucket that does not contain exactly 5 valid
                1-minute source candles (no fabrication).
    from        Unix epoch seconds, inclusive (open time of the first candle)
    to          Unix epoch seconds, inclusive (open time of the last candle)

Response: TradingView UDF format
    { "s": "ok"|"error"|"no_data", "t":[...], "o":[...], "h":[...], "l":[...],
      "c":[...], "v":[...] }   (t is seconds, open-time of each candle).

The script paginates requests in chunks to avoid request-size limits, validates
returned timestamps against the expected grid, reports missing/off-grid candles,
and writes a CSV compatible with icbt's loader. It never fabricates candles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

API_URL = "https://api.wallex.ir/v1/udf/history"
USER_AGENT = "icbt/0.1 (+https://github.com/asgareyvazi/crypto)"

# Wallex UDF resolution codes.
# NOTE: "5m" is mapped to "1" (one-minute source) because Wallex currently
# rejects resolution=5 with "Invalid resolution!". The downloader aggregates
# 1m candles locally into strict 5m UTC-aligned buckets; see
# `aggregate_1m_to_5m`.
RESOLUTION_MAP = {
    "1m": "1",
    "5m": "1",   # fetch 1m, aggregate locally to 5m
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "3h": "180",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
}


@dataclass
class DownloadStats:
    symbol: str
    resolution: str
    start_utc: datetime
    end_utc: datetime
    expected: int
    received: int
    missing: int
    off_grid: int
    duplicates: int
    sha256: str
    output_path: Path
    first_ts: datetime | None
    last_ts: datetime | None


@dataclass
class AggregationResult:
    """Output of :func:`aggregate_1m_to_5m`."""

    df: pd.DataFrame
    duplicates_1m: int
    off_grid_1m: int
    complete_buckets: int
    dropped_buckets_missing: int


def resolution_to_seconds(res: str) -> int:
    if res == "D":
        return 86400
    if res == "W":
        return 7 * 86400
    if res == "M":
        return 30 * 86400  # approximate for validation only
    return int(res) * 60


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def fetch_chunk(
    symbol: str,
    resolution: str,
    start_s: int,
    end_s: int,
    timeout: int = 20,
) -> dict:
    qs = urlencode({"symbol": symbol, "resolution": resolution, "from": start_s, "to": end_s})
    url = f"{API_URL}?{qs}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_err}")


def _records_to_frame(
    records: dict[int, dict],
    step_s: int,
    start_s: int,
    end_s: int,
) -> tuple[pd.DataFrame, int, int]:
    """Convert a ``{ts: record}`` dict to a tz-aware UTC DataFrame and compute
    duplicates/off-grid counts against the expected ``[start_s, end_s]`` grid
    at ``step_s`` cadence.
    """
    if not records:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="date")
        return (
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"], index=empty_idx),
            0,
            0,
        )
    df = pd.DataFrame.from_records(list(records.values()))
    duplicates = len(df) - len(df["timestamp"].unique())
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC", name="date")
    epoch = df.index.view("int64") // 10**9
    on_grid = (epoch % step_s == 0) & (epoch >= start_s) & (epoch <= end_s)
    off_grid = int((~on_grid).sum())
    return df, duplicates, off_grid


def _fetch_grid(
    symbol: str,
    res: str,
    step_s: int,
    start_s: int,
    end_s: int,
    chunk_size: int = 1000,
    label: str = "",
) -> tuple[pd.DataFrame, int, int]:
    """Fetch the exact grid ``[start_s, end_s]`` inclusive at ``step_s``
    resolution using the globally-bound :func:`fetch_chunk`.
    Returns ``(df, duplicates, off_grid)``.
    """
    records: dict[int, dict] = {}
    cursor = start_s
    while cursor <= end_s:
        chunk_end = min(cursor + chunk_size * step_s - step_s, end_s)
        t_cursor = datetime.fromtimestamp(cursor, tz=UTC).isoformat()
        t_ce = datetime.fromtimestamp(chunk_end, tz=UTC).isoformat()
        prefix = f"[{label}]" if label else "[wallex]"
        print(f"{prefix} fetching {symbol} {t_cursor} -> {t_ce}", file=sys.stderr)
        data = fetch_chunk(symbol, res, cursor, chunk_end)
        status = data.get("s")
        if status == "no_data":
            cursor = chunk_end + step_s
            continue
        if status != "ok":
            err_msg = data.get("errmsg") or data.get("s") or "unknown error"
            raise RuntimeError(f"Wallex API returned status={status}: {err_msg}")
        t_arr = data.get("t", [])
        o_arr = data.get("o", [])
        h_arr = data.get("h", [])
        l_arr = data.get("l", [])
        c_arr = data.get("c", [])
        v_arr = data.get("v", [])
        n = min(len(t_arr), len(o_arr), len(h_arr), len(l_arr), len(c_arr), len(v_arr))
        for i in range(n):
            ts = int(t_arr[i])
            records[ts] = {
                "timestamp": datetime.fromtimestamp(ts, tz=UTC),
                "open": float(o_arr[i]),
                "high": float(h_arr[i]),
                "low": float(l_arr[i]),
                "close": float(c_arr[i]),
                "volume": float(v_arr[i]),
            }
        cursor = chunk_end + step_s
    return _records_to_frame(records, step_s, start_s, end_s)


def aggregate_1m_to_5m(
    df_1m: pd.DataFrame,
    target_start: datetime,
    target_end: datetime,
) -> AggregationResult:
    """Aggregate a 1-minute OHLCV frame into strict UTC-aligned 5-minute candles.

    Rules (no fabrication, strict):

    * Output timestamps are 5-minute bucket **OPEN** times aligned to UTC
      (i.e. ``minute % 5 == 0``).
    * For each 5-minute bucket ``[T, T+5min)`` the aggregate requires ALL FIVE
      1-minute candles (``T, T+1m, T+2m, T+3m, T+4m``). If any is missing the
      whole bucket is dropped (reported in ``dropped_buckets_missing``),
      never forward-filled or partial-filled.
    * ``open``   = first minute's open (by timestamp sort)
    * ``high``   = max of the five highs
    * ``low``    = min of the five lows
    * ``close``  = last minute's close
    * ``volume`` = sum of the five volumes
    * Duplicate 1m timestamps: last-write-wins, counted in ``duplicates_1m``.
    * Off-grid 1m minutes (seconds/microseconds != 0) are counted in
      ``off_grid_1m`` and excluded from aggregation.
    * Only buckets with open time in ``[target_start, target_end]``
      (inclusive) are returned, so the final bucket whose open time is
      ``target_end`` (e.g. ``23:55``) is included when T+0..T+4m are all
      present.
    """
    if df_1m is None or len(df_1m) == 0:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="date")
        return AggregationResult(
            df=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"], index=empty_idx),
            duplicates_1m=0,
            off_grid_1m=0,
            complete_buckets=0,
            dropped_buckets_missing=0,
        )

    work = df_1m.copy().sort_index()
    if not isinstance(work.index, pd.DatetimeIndex) or work.index.tz is None:
        raise ValueError("aggregate_1m_to_5m requires a tz-aware DatetimeIndex")
    work = work.tz_convert("UTC")

    duplicates_1m = int(work.index.duplicated().sum())
    work = work[~work.index.duplicated(keep="last")]

    secs = work.index.second + work.index.microsecond / 1_000_000
    on_grid_mask = secs == 0
    off_grid_1m = int((~on_grid_mask).sum())
    work = work.loc[on_grid_mask]

    bucket = work.index.floor("5min")
    counts = work.groupby(bucket).size()
    complete_buckets_idx = counts[counts == 5].index

    if len(complete_buckets_idx) == 0:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="date")
        return AggregationResult(
            df=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"], index=empty_idx),
            duplicates_1m=duplicates_1m,
            off_grid_1m=off_grid_1m,
            complete_buckets=0,
            dropped_buckets_missing=int(len(counts)),
        )

    complete = work.loc[bucket.isin(complete_buckets_idx)]
    g = complete.groupby(complete.index.floor("5min"))
    agg = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })

    dropped_buckets_missing = int((counts < 5).sum())

    ts_start = pd.Timestamp(target_start).astimezone(UTC)
    ts_end = pd.Timestamp(target_end).astimezone(UTC)
    agg = agg.loc[(agg.index >= ts_start) & (agg.index <= ts_end)]

    agg["timestamp"] = agg.index.to_pydatetime()
    agg.index.name = "date"
    agg = agg[["timestamp", "open", "high", "low", "close", "volume"]]

    return AggregationResult(
        df=agg,
        duplicates_1m=duplicates_1m,
        off_grid_1m=off_grid_1m,
        complete_buckets=int(len(agg)),
        dropped_buckets_missing=dropped_buckets_missing,
    )


def _write_csv(df: pd.DataFrame, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = out["timestamp"].map(lambda t: t.isoformat())
    out.to_csv(output_path, index=False)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def download(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    chunk_size: int = 1000,
    fetch_chunk_fn=None,
) -> DownloadStats:
    """Download ``timeframe`` OHLCV for ``symbol`` over ``[start, end]`` inclusive.

    ``fetch_chunk_fn`` is injectable for unit testing; when provided it
    temporarily replaces the module-level :func:`fetch_chunk` for the
    duration of the call (no global leakage on return).
    """
    if timeframe not in RESOLUTION_MAP:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Supported: {sorted(RESOLUTION_MAP)}")

    if fetch_chunk_fn is not None:
        _orig = globals()["fetch_chunk"]
        globals()["fetch_chunk"] = fetch_chunk_fn
    else:
        _orig = None
    try:
        return _download_impl(symbol, timeframe, start, end, output_path, chunk_size)
    finally:
        if _orig is not None:
            globals()["fetch_chunk"] = _orig


def _download_impl(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    chunk_size: int = 1000,
) -> DownloadStats:
    res = RESOLUTION_MAP[timeframe]

    # 5m -> fetch 1m and aggregate locally (Wallex rejects resolution=5).
    if timeframe == "5m":
        target_step = 300
        start_mod = start.timestamp() % target_step
        start_5m = start - timedelta(seconds=start_mod)
        end_mod = end.timestamp() % target_step
        end_5m_open = end - timedelta(seconds=end_mod)
        fetch_start_s = int(start_5m.timestamp())
        # Final bucket [end_5m_open, end_5m_open+5m) needs minutes +0..+4.
        fetch_end_s = int((end_5m_open + timedelta(minutes=4)).timestamp())
        expected = (int(end_5m_open.timestamp()) - fetch_start_s) // target_step + 1

        df_1m, dup1, og1 = _fetch_grid(
            symbol, res="1", step_s=60,
            start_s=fetch_start_s, end_s=fetch_end_s,
            chunk_size=chunk_size, label="wallex 1m",
        )
        agg = aggregate_1m_to_5m(df_1m, start_5m, end_5m_open)
        df = agg.df
        duplicates = dup1 + agg.duplicates_1m
        off_grid = og1 + agg.off_grid_1m

        expected_index = pd.date_range(
            start=start_5m, end=end_5m_open,
            freq=pd.Timedelta(seconds=target_step), tz="UTC",
        )
        missing = len(expected_index.difference(df.index))

        first_ts = df.index[0].to_pydatetime() if len(df) else None
        last_ts = df.index[-1].to_pydatetime() if len(df) else None
        sha = _write_csv(df, output_path)

        return DownloadStats(
            symbol=symbol, resolution=timeframe,
            start_utc=start_5m, end_utc=end_5m_open,
            expected=expected, received=len(df),
            missing=missing, off_grid=off_grid, duplicates=duplicates,
            sha256=sha, output_path=output_path,
            first_ts=first_ts, last_ts=last_ts,
        )

    # All other timeframes: direct request at native resolution.
    step_s = resolution_to_seconds(res)
    start_s = int(start.timestamp())
    end_s = int(end.timestamp())
    start_s -= start_s % step_s
    end_s -= end_s % step_s
    expected = (end_s - start_s) // step_s + 1

    df, duplicates, off_grid = _fetch_grid(
        symbol, res=res, step_s=step_s,
        start_s=start_s, end_s=end_s, chunk_size=chunk_size,
        label=f"wallex {timeframe}",
    )
    expected_index = pd.date_range(
        start=datetime.fromtimestamp(start_s, tz=UTC),
        end=datetime.fromtimestamp(end_s, tz=UTC),
        freq=pd.Timedelta(seconds=step_s), tz="UTC",
    )
    missing = len(expected_index.difference(df.index))
    first_ts = df.index[0].to_pydatetime() if len(df) else None
    last_ts = df.index[-1].to_pydatetime() if len(df) else None
    sha = _write_csv(df, output_path)
    return DownloadStats(
        symbol=symbol, resolution=timeframe,
        start_utc=datetime.fromtimestamp(start_s, tz=UTC),
        end_utc=datetime.fromtimestamp(end_s, tz=UTC),
        expected=expected, received=len(df),
        missing=missing, off_grid=off_grid, duplicates=duplicates,
        sha256=sha, output_path=output_path,
        first_ts=first_ts, last_ts=last_ts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Wallex OHLCV candles.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="5m", choices=sorted(RESOLUTION_MAP.keys()))
    parser.add_argument("--start", required=True, help="ISO UTC start, e.g. 2025-01-01T00:00:00Z")
    parser.add_argument(
        "--end", required=True,
        help="ISO UTC end (candle open time, inclusive), e.g. 2025-12-31T23:55:00Z",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (defaults to data/candles_wallex_<symbol>_<tf>.csv)",
    )
    args = parser.parse_args(argv)

    start = parse_iso(args.start)
    end = parse_iso(args.end)
    if end <= start:
        print("--end must be after --start", file=sys.stderr)
        return 2
    out_path = (
        Path(args.output) if args.output
        else Path(f"data/candles_wallex_{args.symbol.lower()}_{args.timeframe}.csv")
    )

    try:
        stats = download(args.symbol, args.timeframe, start, end, out_path)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print(f"Downloaded {stats.received}/{stats.expected} expected candles")
    print(f"  symbol:     {stats.symbol}")
    print(f"  resolution: {stats.resolution}")
    print(f"  range:      {stats.start_utc.isoformat()} -> {stats.end_utc.isoformat()}")
    print(f"  first ts:   {stats.first_ts}")
    print(f"  last ts:    {stats.last_ts}")
    print(f"  missing:    {stats.missing}")
    print(f"  off-grid:   {stats.off_grid}")
    print(f"  duplicates: {stats.duplicates}")
    print(f"  sha256:     {stats.sha256}")
    print(f"  output:     {stats.output_path}")
    if stats.missing or stats.off_grid:
        print("WARNING: dataset is incomplete; icbt validate will report gaps.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    # Retain backward compat for anyone importing timezone.utc directly.
    sys.exit(main())
