"""Performance metric calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..engine.orders import Trade


@dataclass
class MetricsSummary:
    initial_capital: float
    final_equity: float
    net_profit: float
    return_pct: float
    total_fees: float
    slippage_total: float
    closed_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_return: float
    median_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    gross_profit: float
    gross_loss: float
    time_in_market_pct: float
    buy_and_hold_return_pct: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (max_drawdown_abs, max_drawdown_pct)."""
    if len(equity) == 0:
        return 0.0, 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    dd_pct = dd / running_max.replace(0, np.nan)
    mdd_abs = float(dd.min()) if len(dd) else 0.0
    mdd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0
    return mdd_abs, mdd_pct


def _sharpe(equity: pd.Series, periods_per_year: int = 365 * 24 * 12) -> float | None:
    """Annualised Sharpe ratio (rf=0) from per-period equity returns."""
    if len(equity) < 2:
        return None
    rets = equity.pct_change().dropna()
    if rets.std(ddof=0) == 0:
        return None
    return float(np.sqrt(periods_per_year) * rets.mean() / rets.std(ddof=0))


def _sortino(equity: pd.Series, periods_per_year: int = 365 * 24 * 12) -> float | None:
    if len(equity) < 2:
        return None
    rets = equity.pct_change().dropna()
    downside = rets[rets < 0]
    if len(downside) == 0 or downside.std(ddof=0) == 0:
        return None
    return float(np.sqrt(periods_per_year) * rets.mean() / downside.std(ddof=0))


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    initial_capital: float,
    fee_rate: float,
    buy_and_hold: dict | None = None,
    exposure_bars: pd.Series | None = None,
    slippage_total: float = 0.0,
) -> MetricsSummary:
    closed = [t for t in trades if not t.is_open and t.exit_price is not None]
    n = len(closed)
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]

    gross_profit = float(sum(t.pnl for t in wins)) if wins else 0.0
    gross_loss = float(sum(-t.pnl for t in losses)) if losses else 0.0
    net_profit = gross_profit - gross_loss
    total_fees = float(sum(t.enter_fee + t.exit_fee for t in closed)) if closed else 0.0

    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital
    return_pct = (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    profit_factor = (
        float("inf")
        if (gross_loss == 0 and gross_profit > 0)
        else (gross_profit / gross_loss)
        if gross_loss > 0
        else 0.0
    )

    [t.pnl for t in closed]
    pct_rets = [t.pnl_pct * 100.0 for t in closed]
    avg_trade = float(np.mean(pct_rets)) if pct_rets else 0.0
    med_trade = float(np.median(pct_rets)) if pct_rets else 0.0
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    largest_win = float(max((t.pnl for t in wins), default=0.0))
    largest_loss = float(min((t.pnl for t in losses), default=0.0))

    mdd_abs, mdd_pct = _max_drawdown(equity_curve)
    sharpe = _sharpe(equity_curve)
    sortino = _sortino(equity_curve)

    time_in_market = 0.0
    if exposure_bars is not None and len(exposure_bars):
        time_in_market = float(exposure_bars.sum() / len(exposure_bars)) * 100.0

    bh_ret = buy_and_hold["return_pct"] if buy_and_hold else None

    return MetricsSummary(
        initial_capital=float(initial_capital),
        final_equity=final_equity,
        net_profit=float(net_profit),
        return_pct=float(return_pct),
        total_fees=total_fees,
        slippage_total=float(slippage_total),
        closed_trades=int(n),
        win_rate=float(win_rate),
        profit_factor=_safe_pf(profit_factor),
        max_drawdown=float(mdd_abs),
        max_drawdown_pct=float(mdd_pct) * 100.0,
        avg_trade_return=avg_trade,
        median_trade_return=med_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        time_in_market_pct=time_in_market,
        buy_and_hold_return_pct=bh_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    )


def _json_safe(v):
    """Convert inf/nan to None for JSON output."""
    if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
        return None
    return v


def metrics_to_json_safe(m: MetricsSummary) -> dict[str, Any]:
    d = m.to_dict()
    return {k: _json_safe(v) for k, v in d.items()}


def _safe_pf(pf: float) -> float:
    if np.isinf(pf):
        return float("inf")
    return float(pf)
