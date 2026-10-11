"""Market data ingestion, validation, normalisation and timeframe aggregation.

Public API:
  - :class:`OHLCV`           : pandas-DataFrame-based candle container with metadata
  - :func:`load_csv`         : load an OHLCV CSV into a validated :class:`OHLCV`
  - :func:`resample_ohlcv`   : aggregate candles to a higher timeframe
  - :func:`merge_informative`: merge an informative-timeframe frame onto base frame
  - :class:`DataValidationReport` / :func:`validate_ohlcv`
  - :func:`manifest_hash`    : SHA-256 of file bytes (for provenance)
"""

from __future__ import annotations

from .ohlcv import OHLCV, load_csv, manifest_hash, write_csv
from .timeframes import (
    merge_informative,
    resample_ohlcv,
    timeframe_to_minutes,
    timeframe_to_timedelta,
)
from .validation import (
    DataValidationReport,
    Severity,
    ValidationIssue,
    validate_ohlcv,
)

__all__ = [
    "OHLCV",
    "load_csv",
    "write_csv",
    "manifest_hash",
    "DataValidationReport",
    "ValidationIssue",
    "Severity",
    "validate_ohlcv",
    "resample_ohlcv",
    "merge_informative",
    "timeframe_to_minutes",
    "timeframe_to_timedelta",
]
