"""Run-history persistence: every backtest run is recorded under ./runs/<run_id>/."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = "runs"


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    strategy_name: str
    strategy_version: str
    strategy_source_hash: str
    dataset_path: str
    dataset_hash: str
    exchange: str
    symbol: str
    timeframe: str
    start_utc: str | None
    end_utc: str | None
    engine_version: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    validation_ok: bool
    warnings: list[str]
    errors: list[str]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def ensure_runs_dir(base: str | Path) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    (base / "_index.jsonl").touch(exist_ok=True)
    return base


def append_record(base: str | Path, record: RunRecord) -> Path:
    base = ensure_runs_dir(base)
    index_path = base / "_index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps(record.to_dict(), default=str) + "\n")
    return index_path


def load_records(base: str | Path) -> list[dict[str, Any]]:
    base = ensure_runs_dir(base)
    records: list[dict[str, Any]] = []
    idx = base / "_index.jsonl"
    if not idx.exists():
        return records
    with open(idx) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
