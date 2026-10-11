"""Result export helpers (CSV / JSON / text summary)."""

from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..engine.engine import BacktestResult
from .metrics import metrics_to_json_safe

# Unicode glyphs used in the human-readable summary, with ASCII fallbacks for
# environments whose stdout encoding can't represent them (legacy Windows
# codepages like cp1252 in particular).
_UNICODE_TO_ASCII = {
    "\u2500": "-",  # ─
    "\u2502": "|",  # │
    "\u256d": "+",  # ╭
    "\u256e": "+",  # ╮
    "\u256f": "+",  # ╯
    "\u2570": "+",  # ╰
    "\u250f": "+",  # ┏
    "\u2513": "+",  # ┓
    "\u2517": "+",  # ┗
    "\u251b": "+",  # ┛
    "\u2501": "-",  # ━
    "\u2503": "|",  # ┃
    "\u250c": "+",  # ┌
    "\u2510": "+",  # ┐
    "\u2514": "+",  # └
    "\u2518": "+",  # ┘
    "\u2026": "...",  # …
    "\u2192": "->",  # →
    "\u2713": "OK",  # ✓
    "\u2014": "-",  # —
    "\u221e": "inf",  # ∞
}


def _ascii_safe(text: str) -> str:
    out = text
    for src, dst in _UNICODE_TO_ASCII.items():
        out = out.replace(src, dst)
    return out


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


def _json_default(o: Any) -> Any:
    if isinstance(o, float):
        if math.isinf(o) or math.isnan(o):
            return None
        return o
    return str(o)


def result_summary_text(result: BacktestResult, ascii_only: bool = False) -> str:
    """Produce a human-readable text summary.

    Parameters
    ----------
    ascii_only:
        When True, substitute unicode box-drawing / infinity characters with
        7-bit ASCII substitutes ("--", "|", "inf", ...) so the string is safe
        to print on legacy Windows codepages (cp1252, cp437, etc.) or into
        logs that don't support UTF-8.
    """
    m = result.metrics
    inf = float("inf")
    pf_str = "inf (no losses)" if m.profit_factor == inf else f"{m.profit_factor:.3f}"
    bh_str = f"{m.buy_and_hold_return_pct:.2f}%" if m.buy_and_hold_return_pct is not None else "n/a"
    sharpe_str = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "n/a"
    sep = "-" * 64 if ascii_only else "\u2500" * 64
    lines = [
        f"Run ID:        {result.run_id}",
        f"Strategy:      {result.strategy_info.get('name')} v{result.strategy_info.get('version')}",
        f"Dataset:       {result.dataset_info.get('exchange')}/{result.dataset_info.get('symbol')} "
        f"{result.dataset_info.get('timeframe')}",
        f"Source SHA256: {result.dataset_info.get('source_sha256')}",
        f"Period:        {result.dataset_info.get('start')} -> {result.dataset_info.get('end')}",
        sep,
        f"Initial cash:   {m.initial_capital:,.2f}",
        f"Final equity:   {m.final_equity:,.2f}",
        f"Net profit:     {m.net_profit:,.2f}  ({m.return_pct:.2f}%)",
        f"Buy & Hold:     {bh_str}",
        f"Total fees:     {m.total_fees:,.4f}",
        f"Closed trades:  {m.closed_trades}",
        f"Win rate:       {m.win_rate:.2f}%",
        f"Profit factor:  {pf_str}",
        f"Max drawdown:   {m.max_drawdown:,.2f}  ({m.max_drawdown_pct:.2f}%)",
        f"Avg trade:      {m.avg_trade_return:.3f}%   median {m.median_trade_return:.3f}%",
        f"Avg win / loss: {m.avg_win:,.2f} / {m.avg_loss:,.2f}",
        f"Time in market: {m.time_in_market_pct:.1f}%",
        f"Sharpe (ann.):  {sharpe_str}",
    ]
    if result.warnings:
        lines.append(sep)
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  ! {w}")
    if result.errors:
        lines.append(sep)
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  * {e}")
    text = "\n".join(lines)
    if ascii_only:
        text = _ascii_safe(text)
    return text


def _stream_encoding(stream: Any) -> str:
    return getattr(stream, "encoding", None) or "utf-8"


def _stream_can_print(text: str, stream: Any) -> bool:
    enc = _stream_encoding(stream)
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def safe_print(text: str, console: Any = None) -> None:
    """Print ``text`` in an encoding-safe manner.

    Uses Rich ``console.print`` when the console's output stream can encode the
    characters we use; otherwise emits an ASCII-substituted version directly to
    the stream buffer using ``errors='replace'`` so subprocess captures and
    legacy Windows codepages never raise ``UnicodeEncodeError``.
    """
    target_stream = getattr(console, "file", None) if console is not None else sys.stdout
    if target_stream is None:
        target_stream = sys.stdout
    if _stream_can_print(text, target_stream) and console is not None:
        try:
            console.print(text)
            return
        except UnicodeEncodeError:
            pass
    safe_text = _ascii_safe(text)
    buf = getattr(target_stream, "buffer", None)
    enc = _stream_encoding(target_stream)
    data = safe_text.encode(enc, errors="replace") + b"\n"
    if buf is not None:
        buf.write(data)
        with contextlib.suppress(Exception):
            buf.flush()
    else:
        target_stream.write(safe_text + "\n")


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
    # Write the summary with unicode glyphs preserved (file is UTF-8).
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write(result_summary_text(result, ascii_only=False))
    trades_to_csv(result, paths["trades_csv"])
    equity_to_csv(result, paths["equity_csv"])
    metrics_to_json(result, paths["result_json"])
    with open(paths["config_json"], "w", encoding="utf-8") as f:
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
            ensure_ascii=False,
        )
    return paths
