"""ICBT command-line interface.

Usage examples::

    icbt validate data/candles_wallex_5m_clean.csv
    icbt info data/candles_wallex_5m_clean.csv
    icbt run data/candles_wallex_5m_clean.csv --strategy SmaCross --cash 10000
    icbt run data/candles_wallex_5m_clean.csv --strategy NFI-X8 --btc data/btc_5m.csv
    icbt list-runs
    icbt gui            # launch local web dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .. import __version__
from ..analytics import result_summary_text, write_run_artifacts
from ..data import OHLCV, load_csv, validate_ohlcv
from ..engine import BacktestEngine, EngineConfig, FillModel
from ..engine.persistence import RunRecord, append_record, ensure_runs_dir, file_hash, load_records
from ..strategy.base import Strategy
from ..strategy.nfi_x8 import NostalgiaForInfinityX8
from ..strategy.sma_cross import SmaCrossStrategy

app = typer.Typer(add_completion=False, help="Independent Crypto Backtester (ICBT) CLI")
console = Console()


def _get_strategy(name: str) -> type[Strategy]:
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("smacross", "sma", "simple"):
        return SmaCrossStrategy
    if name in ("nfi", "nfix8", "nostalgiaforinfinityx8", "nfinex8", "x8"):
        return NostalgiaForInfinityX8
    raise typer.BadParameter(f"Unknown strategy: {name}")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"icbt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Independent Crypto Backtester."""


@app.command("validate")
def validate_cmd(
    csv_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="OHLCV CSV file to validate"
    ),
    exchange: str | None = typer.Option(None, "--exchange", "-e"),
    symbol: str | None = typer.Option(None, "--symbol", "-s"),
    timeframe: str | None = typer.Option(None, "--timeframe", "-t"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON report"),
) -> None:
    """Validate an OHLCV CSV file against the strict data-quality policy."""
    try:
        data = load_csv(csv_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    except Exception as e:
        console.print(f"[red]Failed to load {csv_path}: {e}[/red]")
        raise typer.Exit(code=2)
    report = validate_ohlcv(data)
    if json_out:
        console.print_json(json.dumps(report.to_dict(), default=str))
    else:
        console.print(report.summary())
        for issue in report.issues:
            color = {"info": "cyan", "warning": "yellow", "error": "red"}[issue.severity.value]
            console.print(
                f"  [{color}]{issue.severity.value.upper():7}[/{color}] {issue.code}: {issue.message}"
            )
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("info")
def info_cmd(
    csv_path: Path = typer.Argument(..., exists=True, readable=True),
    exchange: str | None = typer.Option(None, "--exchange", "-e"),
    symbol: str | None = typer.Option(None, "--symbol", "-s"),
    timeframe: str | None = typer.Option(None, "--timeframe", "-t"),
) -> None:
    """Print dataset metadata and validation summary."""
    data = load_csv(csv_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    report = validate_ohlcv(data)
    table = Table(title=f"Dataset: {csv_path}")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Exchange", data.exchange)
    table.add_row("Symbol", data.symbol)
    table.add_row("Timeframe", data.timeframe)
    table.add_row("Rows", str(len(data)))
    table.add_row("Start (UTC)", str(data.metadata.start_utc))
    table.add_row("End (UTC)", str(data.metadata.end_utc))
    table.add_row("SHA-256", data.metadata.source_sha256 or "")
    table.add_row("Valid?", "YES" if report.ok else "NO")
    table.add_row("Gaps", str(len(report.gaps)))
    console.print(table)


@app.command("list-strategies")
def list_strategies() -> None:
    """List bundled strategies."""
    table = Table(title="Bundled strategies")
    table.add_column("Name")
    table.add_column("Timeframe")
    table.add_column("Informatives")
    table.add_column("BTC informatives")
    table.add_column("Startup")
    table.add_column("Version")
    for cls in (SmaCrossStrategy, NostalgiaForInfinityX8):
        s = cls()
        table.add_row(
            s.name,
            s.timeframe,
            ",".join(s.inf_timeframes) or "—",
            ",".join(s.btc_inf_timeframes) or "—",
            str(s.startup_candle_count),
            s.version,
        )
    console.print(table)


@app.command("run")
def run_cmd(
    csv_path: Path = typer.Argument(..., exists=True, readable=True),
    strategy_name: str = typer.Option(
        "SmaCross", "--strategy", "-S", help="Strategy name (list via list-strategies)"
    ),
    cash: float = typer.Option(10_000.0, "--cash", "-c"),
    stake_pct: float = typer.Option(0.10, "--stake-pct"),
    stake_amount: float | None = typer.Option(None, "--stake-amount"),
    fee: float = typer.Option(0.001, "--fee", help="Taker/maker fee rate (default 0.001 = 10bps)"),
    slippage_bps: float = typer.Option(5.0, "--slippage-bps"),
    stoploss: float | None = typer.Option(None, "--stoploss"),
    max_open: int = typer.Option(1, "--max-open"),
    fill_model: str = typer.Option(
        "next_open", "--fill-model", help="next_open|next_close|same_close"
    ),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    btc_csv: Path | None = typer.Option(
        None, "--btc", exists=True, readable=True, help="BTC OHLCV CSV for BTC informatives"
    ),
    btc_timeframe: str | None = typer.Option(None, "--btc-timeframe"),
    exchange: str | None = typer.Option(None, "--exchange", "-e"),
    symbol: str | None = typer.Option(None, "--symbol", "-s"),
    timeframe: str | None = typer.Option(None, "--timeframe", "-t"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    no_buyhold: bool = typer.Option(False, "--no-buyhold"),
    fail_on_gaps: bool = typer.Option(False, "--fail-on-gaps"),
) -> None:
    """Run a backtest against a CSV dataset."""
    strat_cls = _get_strategy(strategy_name)
    strategy = strat_cls()

    data = load_csv(csv_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    btc_data: OHLCV | None = None
    if btc_csv:
        btc_data = load_csv(
            btc_csv, exchange=data.exchange, symbol="BTC/USDT", timeframe=btc_timeframe or "5m"
        )
    elif strategy.btc_inf_timeframes:
        console.print(
            f"[yellow]Strategy {strategy.name} requires BTC informatives "
            f"{strategy.btc_inf_timeframes} but no --btc CSV supplied. Run will fail closed.[/yellow]"
        )

    cfg = EngineConfig(
        initial_cash=cash,
        stake_amount=stake_amount,
        stake_pct=stake_pct,
        max_open_trades=max_open,
        fee_rate=fee,
        slippage_bps=slippage_bps,
        fill_model=FillModel(fill_model),
        stoploss=stoploss,
        start=start,
        end=end,
        enable_buy_and_hold=not no_buyhold,
        fail_on_gaps=fail_on_gaps,
    )
    engine = BacktestEngine(cfg)
    result = engine.run(data, strategy, btc_data=btc_data)

    console.print(result_summary_text(result))

    runs_dir = ensure_runs_dir(runs_dir)
    paths = write_run_artifacts(result, runs_dir)
    console.print(f"\nArtifacts written under: {Path(paths['summary']).parent}")

    # Append run record
    strategy_source_file = sys.modules[strat_cls.__module__].__file__
    strategy_hash = file_hash(strategy_source_file) if strategy_source_file else ""
    rec = RunRecord(
        run_id=result.run_id,
        created_at=__import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        strategy_source_hash=strategy_hash,
        dataset_path=str(Path(csv_path).resolve()),
        dataset_hash=data.metadata.source_sha256 or "",
        exchange=data.exchange,
        symbol=data.symbol,
        timeframe=data.timeframe,
        start_utc=data.metadata.start_utc.isoformat() if data.metadata.start_utc else None,
        end_utc=data.metadata.end_utc.isoformat() if data.metadata.end_utc else None,
        engine_version=__version__,
        config=cfg.to_dict(),
        metrics=result.metrics.to_dict(),
        validation_ok=bool(result.validation.get("ok")),
        warnings=result.warnings,
        errors=result.errors,
        output_dir=str(Path(paths["summary"]).parent),
    )
    append_record(runs_dir, rec)

    if result.errors:
        raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs(runs_dir: Path = typer.Option(Path("runs"), "--runs-dir")) -> None:
    records = load_records(runs_dir)
    table = Table(title=f"Run history ({len(records)} runs) at {runs_dir}")
    table.add_column("Run ID")
    table.add_column("Strategy")
    table.add_column("Symbol")
    table.add_column("Return %")
    table.add_column("Trades")
    table.add_column("Valid")
    table.add_column("Created")
    for r in records[::-1][:50]:
        m = r.get("metrics", {})
        table.add_row(
            r.get("run_id", ""),
            f"{r.get('strategy_name', '')} {r.get('strategy_version', '')}",
            f"{r.get('exchange', '')}/{r.get('symbol', '')} {r.get('timeframe', '')}",
            f"{m.get('return_pct', 0):.2f}%",
            str(m.get("closed_trades", 0)),
            "OK" if r.get("validation_ok") else "FAIL",
            r.get("created_at", ""),
        )
    console.print(table)


@app.command("gui")
def gui_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    datasets_dir: Path = typer.Option(Path("data"), "--datasets-dir"),
) -> None:
    """Launch the local web dashboard."""
    import uvicorn

    from ..api.app import create_app

    app_ = create_app(runs_dir=runs_dir, datasets_dir=datasets_dir)
    console.print(f"[green]ICBT dashboard starting at http://{host}:{port}[/green]")
    console.print("Press Ctrl+C to stop.")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
