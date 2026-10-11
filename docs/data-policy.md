# Data-quality policy

ICBT applies a strict validation policy before any backtest runs. The
philosophy is: **fail loudly rather than fabricate data**. The goal is that
reported results are only as good as the underlying candles.

## Required columns and types

A valid OHLCV CSV must contain a header row with at least these columns
(case-insensitive):

```
timestamp, open, high, low, close, volume
```

* `timestamp` is parsed as integer milliseconds, integer seconds, or
  ISO-8601 strings, and normalised to UTC.
* `open`, `high`, `low`, `close`, `volume` must be numeric (no empty strings,
  no units suffixes).

## OHLC invariants

For every candle:

* `high >= open` and `high >= close`
* `low  <= open` and `low  <= close`
* `high >= low`
* `open > 0`, `high > 0`, `low > 0`, `close > 0`
* `volume >= 0`

Any row violating these is reported as an `INVALID_OHLC` error. A dataset
with such errors is reported as **not OK**.

## Timestamp ordering and duplicates

Timestamps must be timezone-aware (UTC) and strictly increasing. Duplicate
timestamps are an error. Out-of-order timestamps are an error.

## Candle interval

The median interval between consecutive timestamps must match the declared
timeframe within ±5%. Any interval smaller than 50% of the nominal timeframe
is flagged as `IRREGULAR_INTERVAL` (likely duplicate or erroneous rows).

## Missing candles / gaps

The validator constructs the expected grid of timestamps at the nominal
frequency from the first to the last valid candle, and reports any missing
slots as gap blocks. A gap has a start, end, and count of missing candles.

By default gaps emit a **WARNING** so that users can inspect the dataset.
Passing `--fail-on-gaps` to `icbt run`, or enabling the equivalent toggle in
the dashboard, promotes gaps (along with other validation errors) to hard
failures that abort the backtest.

**ICBT never forward-fills, interpolates, or copies prices to cover gaps.**
If a dataset has gaps, the user must either:

1. Provide a different, complete dataset covering the period;
2. Narrow the backtest window with `--start`/`--end` to exclude the gap; or
3. Explicitly proceed knowing that trades spanning the gap are reported with
   full visibility of the gap in the validation report.

## Provenance

Every loaded file is hashed with SHA-256; the digest is stored in the run
record and report alongside the exchange, symbol, timeframe, row count, and
UTC start/end. Synthetic fixtures are clearly marked in metadata as
`SYNTHETIC:...` so they cannot be confused with exchange data.

## Known data issues and historical note

The Wallex September 2026 five-minute dataset was previously reported to have
8,623 valid candles out of 8,640 expected slots (17 missing). That figure is
historical diagnostic information; ICBT independently validates any file it
ingests and reports the actual counts per file. A dataset with gaps is never
described as complete.
