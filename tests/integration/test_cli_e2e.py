"""End-to-end tests exercising the real CLI and FastAPI GUI against a synthetic fixture."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from icbt.api.app import create_app
from icbt.data.synthetic import generate_trend_with_cycles, write_fixture_csv


@pytest.fixture()
def fixture_dataset(tmp_path):
    d = generate_trend_with_cycles(n_candles=2000, seed=42)
    p = tmp_path / "candles_synth_BTCUSDT_5m.csv"
    write_fixture_csv(d, p)
    return p


def test_cli_validate_and_run_produce_artifacts(tmp_path, fixture_dataset):
    os.environ.copy()
    # Validate
    v = subprocess.run(
        [sys.executable, "-m", "icbt", "validate", str(fixture_dataset), "--json"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert v.returncode == 0, v.stderr
    payload = json.loads(v.stdout)
    assert payload["ok"]

    # Run
    out = tmp_path / "runs"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "icbt",
            "run",
            str(fixture_dataset),
            "--strategy",
            "SmaCross",
            "--cash",
            "10000",
            "--runs-dir",
            str(out),
            "-e",
            "synth",
            "-s",
            "BTC/USDT",
            "-t",
            "5m",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    # Check artifacts exist
    items = list(out.glob("run_*/*"))
    names = [p.name for p in items]
    assert "summary.txt" in names
    assert "trades.csv" in names
    assert "equity.csv" in names
    assert "result.json" in names
    trades_csv = next(p for p in items if p.name == "trades.csv")
    df = pd.read_csv(trades_csv)
    assert len(df) > 0
    assert "enter_price" in df.columns
    assert "exit_price" in df.columns
    assert "pnl" in df.columns


def test_gui_dashboard_endpoints(tmp_path, fixture_dataset):
    runs = tmp_path / "runs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    import shutil

    shutil.copy(fixture_dataset, data_dir / fixture_dataset.name)
    app = create_app(runs_dir=runs, datasets_dir=data_dir)
    client = TestClient(app)

    # Index
    r = client.get("/")
    assert r.status_code == 200
    assert b"ICBT" in r.content

    # Datasets listing
    r = client.get("/api/datasets")
    assert r.status_code == 200
    j = r.json()
    assert any(ds["name"] == fixture_dataset.name for ds in j["datasets"])

    # Validate API
    r = client.post("/api/validate", json={"file": fixture_dataset.name})
    assert r.status_code == 200
    assert r.json()["report"]["ok"]

    # Run API
    r = client.post(
        "/api/run",
        json={
            "dataset": fixture_dataset.name,
            "strategy": "SmaCross",
            "cash": 10000,
            "stake_pct": 0.1,
            "fee": 0.001,
            "slippage_bps": 0,
            "fill_model": "next_open",
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ok"]
    assert j["metrics"]["closed_trades"] > 0
    assert len(j["equity"]) > 0
    run_id = j["run_id"]

    # Run history
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert any(x["run_id"] == run_id for x in r.json()["runs"])

    # Downloads
    for kind in ("summary", "trades", "equity", "json", "config"):
        r = client.get(f"/runs/{run_id}/download/{kind}")
        assert r.status_code == 200, kind
