"""Validation of OHLCV datasets.

Implements strict policies:
  * Required columns present and numeric.
  * High >= Open/Close/Low; Low <= Open/Close/High; Volume >= 0.
  * Timestamps UTC-normalised, strictly increasing, no duplicates.
  * Candle interval must be within 5% of the nominal timeframe delta.
  * Missing-candle / gap detection against the expected grid.
  * Reports severity (ERROR/WARNING/INFO) and never silently fakes data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from .ohlcv import OHLCV, REQUIRED_COLUMNS
from .timeframes import timeframe_to_timedelta


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    at: str | None = None  # ISO timestamp if applicable

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "at": self.at,
        }


@dataclass
class DataValidationReport:
    """Structured result of :func:`validate_ohlcv`."""

    exchange: str
    symbol: str
    timeframe: str
    source_path: str | None
    rows_total: int
    rows_valid: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)  # {start,end,missing}
    expected_candles: int = 0
    invalid_ohlc_rows: int = 0
    duplicates: int = 0
    out_of_order: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        lines = [
            f"Validation for {self.exchange}/{self.symbol} {self.timeframe}",
            f"  rows_total       = {self.rows_total}",
            f"  rows_valid       = {self.rows_valid}",
            f"  expected_candles = {self.expected_candles}",
            f"  invalid_ohlc     = {self.invalid_ohlc_rows}",
            f"  duplicates       = {self.duplicates}",
            f"  out_of_order     = {self.out_of_order}",
            f"  gaps             = {len(self.gaps)}",
            f"  errors           = {self.error_count}",
            f"  warnings         = {self.warning_count}",
            f"  verdict          = {'OK' if self.ok else 'FAIL'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source_path": self.source_path,
            "rows_total": self.rows_total,
            "rows_valid": self.rows_valid,
            "expected_candles": self.expected_candles,
            "invalid_ohlc_rows": self.invalid_ohlc_rows,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "gaps": self.gaps,
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }


def _ohlc_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows satisfying OHLC invariants (True == valid)."""
    h = df["high"]
    lo = df["low"]
    o = df["open"]
    c = df["close"]
    v = df["volume"]
    return (
        (h >= np.maximum(o, c))
        & (lo <= np.minimum(o, c))
        & (h >= lo)
        & (v >= 0)
        & (o > 0)
        & (c > 0)
        & (h > 0)
        & (lo > 0)
    )


def validate_ohlcv(
    data: OHLCV, expected_interval_delta_seconds: float | None = None
) -> DataValidationReport:
    """Run the full strict-validation policy against an :class:`OHLCV`."""
    df = data.df
    report = DataValidationReport(
        exchange=data.exchange,
        symbol=data.symbol,
        timeframe=data.timeframe,
        source_path=data.metadata.source_path,
        rows_total=len(df),
    )

    if len(df) == 0:
        report.issues.append(
            ValidationIssue(Severity.ERROR, "EMPTY_DATASET", "Dataset has zero rows")
        )
        return report

    # Columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            report.issues.append(
                ValidationIssue(Severity.ERROR, "MISSING_COLUMN", f"Missing column: {col}")
            )
    if report.error_count > 0 and any(i.code == "MISSING_COLUMN" for i in report.issues):
        return report

    # Timezone check
    if df.index.tz is None:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR, "NO_TZ", "Timestamps are not timezone-aware (expected UTC)"
            )
        )

    # OHLC invariants
    valid = _ohlc_mask(df)
    invalid_n = int((~valid).sum())
    report.invalid_ohlc_rows = invalid_n
    if invalid_n:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_OHLC",
                f"{invalid_n} rows violate OHLC invariants (high>=open/close/low; low<=...; positive prices; volume>=0)",
            )
        )

    # Sorting / duplicates
    ts = df.index
    if not ts.is_monotonic_increasing:
        ooo = int((ts.to_series().diff().dt.total_seconds() < 0).sum())
        report.out_of_order = ooo
        report.issues.append(
            ValidationIssue(Severity.ERROR, "OUT_OF_ORDER", f"{ooo} timestamps are out of order")
        )
    dup_mask = ts.duplicated(keep="first")
    dup_n = int(dup_mask.sum())
    report.duplicates = dup_n
    if dup_n:
        report.issues.append(
            ValidationIssue(Severity.ERROR, "DUPLICATE_TS", f"{dup_n} duplicate timestamps")
        )

    # Interval checks & gap detection on valid (sorted, no-dup) frame
    clean = df[valid].sort_index()
    if len(clean) < 2:
        report.rows_valid = len(clean)
        if len(clean) < 2:
            report.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "TOO_SHORT",
                    "Fewer than 2 valid candles; cannot evaluate intervals",
                )
            )
        return report

    diffs = clean.index.to_series().diff().dt.total_seconds().dropna()
    nominal = timeframe_to_timedelta(data.timeframe).total_seconds()
    if expected_interval_delta_seconds is not None:
        nominal = float(expected_interval_delta_seconds)

    # Median diff as empirical interval (tolerance check)
    median_diff = float(diffs.median())
    if abs(median_diff - nominal) / nominal > 0.05:
        report.issues.append(
            ValidationIssue(
                Severity.WARNING,
                "INTERVAL_MISMATCH",
                f"Median interval {median_diff:.1f}s differs from expected {nominal:.1f}s by "
                f"{abs(median_diff - nominal) / nominal * 100:.1f}%",
            )
        )

    # Expected grid: from first to last valid index on nominal grid.
    start = clean.index[0]
    end = clean.index[-1]
    expected_index = pd.date_range(start=start, end=end, freq=f"{int(nominal)}s", tz="UTC")
    report.expected_candles = len(expected_index)
    report.rows_valid = len(clean)
    missing = expected_index.difference(clean.index)
    if len(missing):
        # Group consecutive missing into gap blocks
        if len(missing) == 1:
            report.gaps.append(
                {"start": missing[0].isoformat(), "end": missing[0].isoformat(), "missing": 1}
            )
        else:
            gaps_blocks = []
            block_start = missing[0]
            prev = missing[0]
            for t in missing[1:]:
                if (t - prev).total_seconds() == nominal:
                    prev = t
                else:
                    gaps_blocks.append((block_start, prev))
                    block_start = t
                    prev = t
            gaps_blocks.append((block_start, prev))
            for s, e in gaps_blocks:
                cnt = int(((e - s).total_seconds() / nominal) + 1)
                report.gaps.append({"start": s.isoformat(), "end": e.isoformat(), "missing": cnt})
        total_missing = int(sum(g["missing"] for g in report.gaps))
        report.issues.append(
            ValidationIssue(
                Severity.WARNING,
                "GAPS",
                f"Detected {len(report.gaps)} gap(s) with {total_missing} missing candle(s) "
                f"out of {report.expected_candles} expected slots",
            )
        )

    # Tolerance for small interval jitter: any diff > 1.5 * nominal that's not aligned to grid = gap already counted above.
    # Any diff < 0.5 * nominal = irregular (should have been caught as duplicate or OoO but flag anyway).
    too_small = int((diffs < 0.5 * nominal).sum())
    if too_small:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "IRREGULAR_INTERVAL",
                f"{too_small} consecutive-candle intervals are <50% of nominal ({nominal:.0f}s) - suspicious duplicate/overlap",
            )
        )

    return report
