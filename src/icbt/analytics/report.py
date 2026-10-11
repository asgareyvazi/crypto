"""Result export helpers (CSV / JSON / text summary)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..engine.engine import BacktestResult
from .metrics import metrics_to_json_safe


def trades_to_csv(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result.trades)
    df.to_csv(path, index=False)
    return path


def equity_to_csv(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(path)
    return path


def metrics_to_json(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["metrics"] = metrics_to_json_safe(result.metrics)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, float):
        import math

        if math.isinf(o) or math.isnan(o):
            return None
        return o
    return str(o)


def result_summary_text(result: BacktestResult) -> str:
    m = result.metrics
    lines = [
        f"Run ID:        {result.run_id}",
        f"Strategy:      {result.strategy_info.get('name')} v{result.strategy_info.get('version')}",
        f"Dataset:       {result.dataset_info.get('exchange')}/{result.dataset_info.get('symbol')} "
        f"{result.dataset_info.get('timeframe')}",
        f"Source SHA256: {result.dataset_info.get('source_sha256')}",
        f"Period:        {result.dataset_info.get('start')} -> {result.dataset_info.get('end')}",
        "─" * 64,
        f"Initial cash:   {m.initial_capital:,.2f}",
        f"Final equity:   {m.final_equity:,.2f}",
        f"Net profit:     {m.net_profit:,.2f}  ({m.return_pct:.2f}%)",
        "Buy & Hold:     "
        + (f"{m.buy_and_hold_return_pct:.2f}%" if m.buy_and_hold_return_pct is not None else "n/a"),
        f"Total fees:     {m.total_fees:,.4f}",
        f"Closed trades:  {m.closed_trades}",
        f"Win rate:       {m.win_rate:.2f}%",
        f"Profit factor:  {'∞ (no losses)' if m.profit_factor == float('inf') else f'{m.profit_factor:.3f}'}",
        f"Max drawdown:   {m.max_drawdown:,.2f}  ({m.max_drawdown_pct:.2f}%)",
        f"Avg trade:      {m.avg_trade_return:.3f}%   median {m.median_trade_return:.3f}%",
        f"Avg win / loss: {m.avg_win:,.2f} / {m.avg_loss:,.2f}",
        f"Time in market: {m.time_in_market_pct:.1f}%",
        "Sharpe (ann.):  " + (f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "n/a"),
    ]
    if result.warnings:
        lines.append("─" * 64)
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  ! {w}")
    if result.errors:
        lines.append("─" * 64)
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  * {e}")
    return "\n".join(lines)


def write_run_artifacts(result: BacktestResult, out_dir: str | Path) -> dict[str, str]:
    """Persist the canonical set of run artifacts. Returns a map of name->path."""
    out_dir = Path(out_dir)
    run_dir = out_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": str(run_dir / "summary.txt"),
        "trades_csv": str(run_dir / "trades.csv"),
        "equity_csv": str(run_dir / "equity.csv"),
        "result_json": str(run_dir / "result.json"),
        "config_json": str(run_dir / "config.json"),
    }
    with open(paths["summary"], "w") as f:
        f.write(result_summary_text(result))
    trades_to_csv(result, paths["trades_csv"])
    equity_to_csv(result, paths["equity_csv"])
    metrics_to_json(result, paths["result_json"])
    with open(paths["config_json"], "w") as f:
        json.dump(
            {
                "run_id": result.run_id,
                "strategy": result.strategy_info,
                "config": result.config,
                "dataset": result.dataset_info,
                "validation_ok": result.validation.get("ok"),
                "warnings": result.warnings,
                "errors": result.errors,
            },
            f,
            indent=2,
            default=str,
        )
    return paths
