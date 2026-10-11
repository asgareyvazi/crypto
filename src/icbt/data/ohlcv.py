"""OHLCV candle container and CSV I/O."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class OHLCVMetadata:
    """Provenance & metadata for an OHLCV dataset."""

    exchange: str
    symbol: str
    timeframe: str
    source_path: str | None = None
    source_sha256: str | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    n_rows: int = 0
    columns: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.start_utc:
            d["start_utc"] = self.start_utc.isoformat()
        if self.end_utc:
            d["end_utc"] = self.end_utc.isoformat()
        return d


@dataclass
class OHLCV:
    """In-memory OHLCV DataFrame plus metadata.

    ``df`` is guaranteed (after construction) to have columns
    ``timestamp, open, high, low, close, volume`` with a UTC ``DatetimeIndex``
    named ``date``. ``df`` is sorted ascending by timestamp and has no duplicate
    timestamps (``load_csv`` will error on duplicates; use ``validation`` to
    report them).
    """

    df: pd.DataFrame = field(repr=False)
    metadata: OHLCVMetadata

    # Convenience accessors -------------------------------------------------
    @property
    def exchange(self) -> str:
        return self.metadata.exchange

    @property
    def symbol(self) -> str:
        return self.metadata.symbol

    @property
    def timeframe(self) -> str:
        return self.metadata.timeframe

    def __len__(self) -> int:
        return len(self.df)

    def slice(self, start: datetime | None = None, end: datetime | None = None) -> OHLCV:
        df = self.df
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        md = OHLCVMetadata(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_path=self.metadata.source_path,
            source_sha256=self.metadata.source_sha256,
            start_utc=df.index[0].to_pydatetime() if len(df) else None,
            end_utc=df.index[-1].to_pydatetime() if len(df) else None,
            n_rows=len(df),
            columns=tuple(df.columns),
            notes=self.metadata.notes,
        )
        return OHLCV(df=df.copy(), metadata=md)


def manifest_hash(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file (for provenance manifests)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_TF_SUFFIX_RE = __import__("re").compile(r"^(\d+)([mhdw])$", __import__("re").IGNORECASE)


def _infer_metadata_from_filename(path: Path) -> tuple[str, str, str]:
    """Best-effort heuristic: ``candles_<exchange>_<symbol>_<timeframe>.csv``.

    The timeframe is the final token if it matches ``<N><unit>``; the symbol
    is the preceding token (with possible ``-`` / ``/`` separators normalised
    to ``/``). Anything between exchange and the (symbol, timeframe) tail is
    ignored, so ``candles_wallex_btcusdt_5m_clean.csv`` resolves to
    ``wallex / BTCUSDT / 5m``.
    """
    stem = path.stem
    parts = stem.split("_")
    exchange = "unknown"
    symbol = "UNKNOWN"
    timeframe = "5m"
    if parts and parts[0] == "candles":
        rest = parts[1:]
        # Strip trailing tokens that don't look like timeframe/symbol (e.g. "clean", "repaired")
        # by searching from the end for a timeframe.
        tf_idx = -1
        for i in range(len(rest) - 1, -1, -1):
            if _TF_SUFFIX_RE.match(rest[i]):
                tf_idx = i
                timeframe = rest[i]
                break
        if tf_idx >= 1:
            symbol = rest[tf_idx - 1].upper()
            exchange = rest[0]
        elif tf_idx == 0 and len(rest) >= 2:
            # candles_5m_xxx (unusual) - skip
            pass
        elif len(rest) >= 1:
            exchange = rest[0]
            if len(rest) >= 2:
                symbol = rest[-1].upper()
    return exchange, symbol, timeframe


def load_csv(
    path: str | Path,
    exchange: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    timestamp_column: str = "timestamp",
    tz: str = "UTC",
) -> OHLCV:
    """Load an OHLCV CSV into a sorted, normalised :class:`OHLCV`.

    The CSV must contain columns for ``timestamp``, ``open``, ``high``, ``low``,
    ``close`` and ``volume`` (case-insensitive). Timestamps can be integer ms,
    integer s, or ISO-8601 strings.

    Raises
    ------
    ValueError
        When required columns are missing or numeric conversion fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {path}")

    sha = manifest_hash(path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise timestamp column name.
    if timestamp_column.lower() != "timestamp" and timestamp_column.lower() in df.columns:
        df = df.rename(columns={timestamp_column.lower(): "timestamp"})
    elif "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing required columns {missing}; got {list(df.columns)}")

    # Coerce timestamp.
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # Heuristic: values > 1e12 are ms, otherwise seconds.
        sample = ts.dropna().iloc[0] if len(ts.dropna()) else 0
        unit = "ms" if float(sample) > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(f"CSV {path} has {bad} unparseable timestamps")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = int(df[col].isna().sum())
            raise ValueError(f"CSV {path} column {col} has {bad} non-numeric values")

    df = (
        df[list(REQUIRED_COLUMNS)]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
    )
    df = df.set_index(pd.DatetimeIndex(df["timestamp"], tz="UTC"))
    df.index.name = "date"

    ex, sym, tf = _infer_metadata_from_filename(path)
    if exchange is None:
        exchange = ex
    if symbol is None:
        symbol = sym
    if timeframe is None:
        timeframe = tf

    md = OHLCVMetadata(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        source_path=str(path.resolve()),
        source_sha256=sha,
        start_utc=df.index[0].to_pydatetime() if len(df) else None,
        end_utc=df.index[-1].to_pydatetime() if len(df) else None,
        n_rows=len(df),
        columns=tuple(df.columns),
    )
    return OHLCV(df=df, metadata=md)


def write_csv(data: OHLCV, path: str | Path) -> Path:
    """Persist the OHLCV frame to CSV (UTC timestamps as ISO-8601)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = data.df.copy()
    out["timestamp"] = out.index.map(lambda t: t.isoformat())
    out[list(REQUIRED_COLUMNS)].to_csv(path, index=False)
    return path
