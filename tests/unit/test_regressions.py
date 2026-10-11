"""Regression tests for issues discovered on real Windows deployments."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from icbt.analytics import result_summary_text, safe_print
from icbt.api.app import create_app
from icbt.data.synthetic import generate_trend_with_cycles, write_fixture_csv


# ---------------------------------------------------------------------- template response
def test_dashboard_root_returns_200_with_new_style_template_response():
    """Regression for Starlette 0.39+ TemplateResponse signature.

    Using the old ``TemplateResponse(name, {...})`` positional form triggered
    ``TypeError: unhashable type: 'dict'`` in at least one Starlette build.
    The ``index`` route must return HTTP 200 and render the dashboard HTML.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data").mkdir()
        (d / "runs").mkdir()
        app = create_app(runs_dir=d / "runs", datasets_dir=d / "data")
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 200
            assert b"ICBT" in r.content
            assert b"Backtest config" in r.content


def test_dashboard_docs_endpoint_ok():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data").mkdir()
        (d / "runs").mkdir()
        app = create_app(runs_dir=d / "runs", datasets_dir=d / "data")
        with TestClient(app) as client:
            r = client.get("/api/datasets")
            assert r.status_code == 200
            assert "datasets" in r.json()
            r2 = client.get("/docs")
            assert r2.status_code == 200


# ---------------------------------------------------------------------- CLI encoding
def test_result_summary_ascii_only_produces_encodable_output():
    """``ascii_only=True`` must produce output encodable as cp1252."""
    from icbt.engine import BacktestEngine, EngineConfig
    from icbt.strategy.sma_cross import SmaCrossStrategy

    data = generate_trend_with_cycles(n_candles=1000, seed=1)
    r = BacktestEngine(EngineConfig()).run(data, SmaCrossStrategy())
    text = result_summary_text(r, ascii_only=True)
    # Must encode without error on cp1252, cp437 and ASCII.
    text.encode("cp1252")
    text.encode("ascii", errors="strict")
    assert "Run ID:" in text
    assert "Profit factor" in text
    # No box-drawing unicode
    assert "\u2500" not in text
    assert "\u221e" not in text or "inf" in text


def test_safe_print_to_cp1252_stream_does_not_raise():
    """safe_print must not raise UnicodeEncodeError even when the output stream
    is restricted to cp1252 and the text contains box-drawing characters."""
    from rich.console import Console

    from icbt.engine import BacktestEngine, EngineConfig
    from icbt.strategy.sma_cross import SmaCrossStrategy

    data = generate_trend_with_cycles(n_candles=600, seed=2)
    r = BacktestEngine(EngineConfig()).run(data, SmaCrossStrategy())
    text = result_summary_text(r, ascii_only=False)

    # Simulate a cp1252 stdout buffer (as on Windows PowerShell).
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
    console = Console(file=stream, force_terminal=False, highlight=False)
    safe_print(text, console=console)
    stream.flush()
    out = buf.getvalue()
    assert len(out) > 0
    # Must decode back and contain recognisable content.
    decoded = out.decode("cp1252", errors="replace")
    assert "Run ID:" in decoded
    assert "Final equity" in decoded


def test_cli_run_under_restricted_encoding(tmp_path):
    """Run the CLI as a subprocess with PYTHONIOENCODING=cp1252:replace to
    simulate a Windows cp1252 environment. It must exit 0 and produce artifacts."""
    d = generate_trend_with_cycles(n_candles=800, seed=3)
    csv_path = tmp_path / "candles_synth_BTCUSDT_5m.csv"
    write_fixture_csv(d, csv_path)
    runs_dir = tmp_path / "runs"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252:replace"
    cp = subprocess.run(
        [
            sys.executable, "-m", "icbt", "run",
            str(csv_path),
            "--strategy", "SmaCross",
            "--cash", "10000",
            "-e", "synth", "-s", "BTCUSDT", "-t", "5m",
            "--runs-dir", str(runs_dir),
        ],
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )
    assert cp.returncode == 0, (cp.returncode, cp.stderr.decode("utf-8", errors="replace"))
    # stdout must contain key words (in cp1252 after substitution).
    stdout = cp.stdout.decode("cp1252", errors="replace")
    assert "Final equity" in stdout
    assert "Closed trades" in stdout
    # Artifacts must exist
    run_dirs = list(runs_dir.glob("run_*"))
    assert run_dirs, "expected run directory with artifacts"
    rd = run_dirs[0]
    for name in ("summary.txt", "trades.csv", "equity.csv", "result.json", "config.json"):
        assert (rd / name).exists(), f"missing {name}"
    trades = pd.read_csv(rd / "trades.csv")
    assert len(trades) >= 1
    # Ledger must have a record.
    assert (runs_dir / "_index.jsonl").exists()
