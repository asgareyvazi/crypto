# CLI reference notes

## `python -m icbt run <csv>` — window semantics

* `--start` and `--end` accept ISO-8601 UTC strings (e.g. `2025-01-01T00:00:00Z`).
* They filter on candle **open timestamps** and are **inclusive on both ends**.
* Because signals emit at candle close and fills occur on the next candle's open
  (`--fill-model next_open`, default), an `--end` of `2025-12-31T23:55:00Z` will
  include the signal evaluated from that candle but no fill will occur for a
  signal generated on the final candle (there is no `2026-01-01T00:00:00Z`
  candle to fill on). Any still-open position at `--end` is force-closed and
  reported as a warning.
* `--btc <path>` loads BTC OHLCV for BTC informative pairs. `--btc-timeframe`
  defaults to the strategy's first BTC informative timeframe (`4h` for NFI-X8)
  and is resampled from the supplied CSV using strict semantics (partial
  buckets dropped, gaps rejected). The informative merge enforces
  no-lookahead: BTC values attached to a 5m candle at time *t* come from the
  BTC 4h candle that closed strictly before *t*.

## Resampling strictness (`resample_ohlcv`)

* Source data must lie on a regular grid at the declared source timeframe;
  any off-grid, duplicate, or timezone-misaligned row raises `ResamplingError`.
* Buckets that do not contain the expected number of source candles are
  dropped. The first and last buckets are expected to be partial and are
  dropped silently; internal missing-bucket gaps raise `ResamplingError`.
* No candles are fabricated. Missing source candles → that target bucket is
  absent from the output rather than forward-filled.

## Wallex downloader (`scripts/download_wallex.py`)

* Uses the public UDF endpoint `https://api.wallex.ir/v1/udf/history` with
  minute-based resolutions.
* Paginates in 1000-candle chunks with 3 retries per chunk.
* Writes an icbt-compatible CSV (`date` index, ISO-8601 UTC timestamps,
  `open/high/low/close/volume` columns) and reports first/last timestamps,
  missing/off-grid/duplicate counts, and SHA-256 of the output.
* Exit code 3 when the dataset is incomplete (missing candles detected).
