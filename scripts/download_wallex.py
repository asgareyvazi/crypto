"""Wallex market-data downloader (UDF/TradingView-compatible history endpoint).

Wallex public API: GET https://api.wallex.ir/v1/udf/history

Parameters (per official docs https://api-docs.wallex.ir):
    symbol      e.g. BTCUSDT, BTCTMN
    resolution  one of: 1, 5, 15, 30, 60, 120, 180, 240, 360, 720, "D", "W", "M"
                (numeric values are MINUTES; D/W/M are daily/weekly/monthly)
    from        Unix epoch seconds, inclusive (open time of the first candle)
    to          Unix epoch seconds, inclusive (open time of the last candle)

Response: TradingView UDF format
    { "s": "ok"|"error"|"no_data", "t":[...], "o":[...], "h":[...], "l":[...],
      "c":[...], "v":[...] }   (t is seconds, open-time of each candle).

The script paginates requests in chunks to avoid request-size limits, validates
the returned timestamps against the expected grid, reports missing/off-grid
candles, and writes a CSV compatible with icbt's loader. It never fabricates
candles.

If --btc-pair is specified (default BTCUSDT), the same CSV can be passed to
`icbt run ... --btc data/...csv` for strategies that need a BTC informative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_URL = "https://api.wallex.ir/v1/udf/history"
USER_AGENT = "icbt/0.1 (+https://github.com/asgareyvazi/crypto)"

# Wallex resolution codes (minutes; or D/W/M strings).
RESOLUTION_MAP = {
    "1m": "1",
    "5m": "5",
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


def resolution_to_seconds(res: str) -> int:
    if res == "D":
        return 86400
    if res == "W":
        return 7 * 86400
    if res == "M":
        return 30 * 86400  # approximate for validation only
    return int(res) * 60


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_chunk(symbol: str, resolution: str, start_s: int, end_s: int, timeout: int = 20) -> dict:
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


def download(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    chunk_size: int = 1000,
) -> DownloadStats:
    if timeframe not in RESOLUTION_MAP:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Supported: {sorted(RESOLUTION_MAP)}")
    res = RESOLUTION_MAP[timeframe]
    step_s = resolution_to_seconds(res)

    start_s = int(start.timestamp())
    end_s = int(end.timestamp())

    # Align start to the timeframe boundary on/after the requested start.
    start_s = start_s - (start_s % step_s)
    # end_s must be an exact candle open within the range (inclusive).
    end_s = end_s - (end_s % step_s)

    expected = (end_s - start_s) // step_s + 1
    records: dict[int, dict] = {}
    cursor = start_s
    while cursor <= end_s:
        chunk_end = min(cursor + chunk_size * step_s - step_s, end_s)
        print(f"[wallex] fetching {symbol} {timeframe} {datetime.fromtimestamp(cursor, tz=timezone.utc).isoformat()} -> {datetime.fromtimestamp(chunk_end, tz=timezone.utc).isoformat()}", file=sys.stderr)
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
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                "open": float(o_arr[i]),
                "high": float(h_arr[i]),
                "low": float(l_arr[i]),
                "close": float(c_arr[i]),
                "volume": float(v_arr[i]),
            }
        cursor = chunk_end + step_s

    # Build dataframe.
    df = pd.DataFrame.from_records(list(records.values()))
    duplicates = len(df) - len(df["timestamp"].unique()) if len(df) else 0
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC")
    df.index.name = "date"

    # Determine missing/off-grid timestamps vs expected grid.
    expected_index = pd.date_range(
        start=datetime.fromtimestamp(start_s, tz=timezone.utc),
        end=datetime.fromtimestamp(end_s, tz=timezone.utc),
        freq=pd.Timedelta(seconds=step_s),
        tz="UTC",
    )
    missing = expected_index.difference(df.index)
    # Off-grid: timestamps in response that aren't in expected_index
    off_grid_mask = ~df.index.isin(expected_index)
    off_grid = int(off_grid_mask.sum())

    # Persist CSV (only candles within the requested range; never fabricate missing ones).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = out["timestamp"].map(lambda t: t.isoformat())
    out.to_csv(output_path, index=False)
    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()

    return DownloadStats(
        symbol=symbol,
        resolution=timeframe,
        start_utc=datetime.fromtimestamp(start_s, tz=timezone.utc),
        end_utc=datetime.fromtimestamp(end_s, tz=timezone.utc),
        expected=expected,
        received=len(df),
        missing=len(missing),
        off_grid=off_grid,
        duplicates=duplicates,
        sha256=sha,
        output_path=output_path,
        first_ts=df.index[0].to_pydatetime() if len(df) else None,
        last_ts=df.index[-1].to_pydatetime() if len(df) else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Wallex OHLCV candles.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="5m", choices=sorted(RESOLUTION_MAP.keys()))
    parser.add_argument("--start", required=True, help="ISO UTC start, e.g. 2025-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO UTC end (candle open time, inclusive), e.g. 2025-12-31T23:55:00Z")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (defaults to data/candles_wallex_<symbol>_<tf>.csv)")
    args = parser.parse_args(argv)

    start = parse_iso(args.start)
    end = parse_iso(args.end)
    if end <= start:
        print("--end must be after --start", file=sys.stderr)
        return 2
    out_path = Path(args.output) if args.output else Path(f"data/candles_wallex_{args.symbol.lower()}_{args.timeframe}.csv")

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
    sys.exit(main())
