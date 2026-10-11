"""Event-driven backtesting engine.

Fill semantics (documented, conservative):

* Signals are evaluated at the **close** of candle T using data known at or
  before T.
* If :class:`FillModel.NEXT_OPEN` is selected (default), the resulting market
  order fills at the **open** of candle T+1 at price ``open_(T+1) * (1 +/-
  slippage)`` and with fee ``notional * fee_rate``.
* Stop-loss orders are checked against the next candle's **low/high**: if
  ``low_(T+1) <= stop_price`` the stop triggers at ``max(open_(T+1),
  stop_price) - slippage`` for sells (i.e. we assume the open is hit first
  only if open <= stop; otherwise the stop is hit during the candle at the
  stop price). This is a conservative intrabar model that never assumes the
  stop was hit when the candle low did not reach it.
* ROI exits, trailing stops, and custom exits are evaluated similarly at
  each candle's close; they fill at the next open.
* Equity is mark-to-market at each candle close.
* No lookahead: indicators, signals and informative joins are computed using
  only data with timestamp <= current candle's close. (see
  :func:`icbt.data.timeframes.merge_informative`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..analytics.metrics import MetricsSummary

from ..data import (
    OHLCV,
    DataValidationReport,
    merge_informative,
    resample_ohlcv,
    timeframe_to_timedelta,
    validate_ohlcv,
)
from ..strategy.base import Strategy
from ..strategy.indicators import ATR
from .config import EngineConfig, FillModel, SlippageModel
from .orders import Order, OrderSide, OrderType, Trade


@dataclass
class BacktestResult:
    run_id: str
    strategy_info: dict[str, Any]
    config: dict[str, Any]
    dataset_info: dict[str, Any]
    validation: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: pd.DataFrame = field(repr=False)
    metrics: MetricsSummary
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def to_dict(self) -> dict[str, Any]:
        from ..analytics.metrics import metrics_to_json_safe

        return {
            "run_id": self.run_id,
            "strategy": self.strategy_info,
            "config": self.config,
            "dataset": self.dataset_info,
            "validation": self.validation,
            "metrics": metrics_to_json_safe(self.metrics),
            "warnings": self.warnings,
            "errors": self.errors,
            "trade_count": len(self.trades),
        }


class BacktestEngine:
    """Event-driven single-pair spot backtester."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # --------------------------------------------------------------------- run
    def run(
        self,
        data: OHLCV,
        strategy: Strategy,
        btc_data: OHLCV | None = None,
        extra_informative: dict[str, OHLCV] | None = None,
    ) -> BacktestResult:
        run_id = _new_run_id()
        warnings: list[str] = []
        errors: list[str] = []

        # 1. Validate data
        validation = validate_ohlcv(data)
        if not validation.ok:
            msg = f"Dataset failed validation with {validation.error_count} errors"
            if self.config.fail_on_gaps:
                errors.append(msg)
                return _empty_result(
                    run_id, strategy, self.config, data, validation, errors, warnings
                )
            warnings.append(msg + " (proceeding because fail_on_gaps=False)")
        if validation.gaps:
            warnings.append(
                f"Dataset has {len(validation.gaps)} gap block(s) totaling "
                f"{sum(g['missing'] for g in validation.gaps)} missing candles"
            )

        # 2. Build enriched dataframe (informative timeframes)
        df = data.df.copy()
        available_inf: dict[str, pd.DataFrame] = {}
        available_inf[strategy.timeframe] = df
        for tf in strategy.inf_timeframes:
            if tf == strategy.timeframe:
                continue
            try:
                available_inf[tf] = resample_ohlcv(
                    df[["open", "high", "low", "close", "volume"]], strategy.timeframe, tf
                )
            except Exception as e:
                errors.append(f"Failed to resample to {tf}: {e}")
                return _empty_result(
                    run_id, strategy, self.config, data, validation, errors, warnings
                )
        # BTC informative
        btc_inf: dict[str, pd.DataFrame] = {}
        if strategy.btc_inf_timeframes:
            if btc_data is None:
                # If BTC 4h requested but no BTC data, generate a synthetic BTC price? NO - fail closed.
                errors.append(
                    f"Strategy requires BTC informative(s) {strategy.btc_inf_timeframes} but no BTC data supplied"
                )
                return _empty_result(
                    run_id, strategy, self.config, data, validation, errors, warnings
                )
            for tf in strategy.btc_inf_timeframes:
                if tf == btc_data.timeframe:
                    btc_inf[tf] = btc_data.df
                else:
                    btc_inf[tf] = resample_ohlcv(
                        btc_data.df[["open", "high", "low", "close", "volume"]],
                        btc_data.timeframe,
                        tf,
                    )

        # 3. Merge informatives onto the base frame
        enriched = df.copy()
        for tf in strategy.inf_timeframes:
            if tf == strategy.timeframe:
                continue
            suffix = f"_{tf}"
            enriched = merge_informative(
                enriched, available_inf[tf], suffix, strategy.timeframe, tf
            )
        for tf, bdf in btc_inf.items():
            suffix = f"_btc_{tf}"
            enriched = merge_informative(enriched, bdf, suffix, strategy.timeframe, tf)

        # 4. Compute indicators and signals
        try:
            enriched = strategy.populate_indicators(enriched, {"pair": data.symbol})
            enriched = strategy.populate_entry_signals(enriched)
            enriched = strategy.populate_exit_signals(enriched)
        except Exception as e:
            errors.append(f"Strategy indicator/signal failure: {e}")
            return _empty_result(run_id, strategy, self.config, data, validation, errors, warnings)

        if "enter_long" not in enriched.columns or "exit_long" not in enriched.columns:
            errors.append("Strategy did not produce enter_long/exit_long columns")
            return _empty_result(run_id, strategy, self.config, data, validation, errors, warnings)

        # 5. Trim warm-up
        startup = int(strategy.startup_candle_count)
        if len(enriched) <= startup:
            errors.append(
                f"Insufficient data: {len(enriched)} candles <= startup_candle_count={startup}"
            )
            return _empty_result(run_id, strategy, self.config, data, validation, errors, warnings)
        # Keep warm-up for indicator computation but only start trading after it.
        # The indicator methods already return NaN until they have enough history
        # (signals are gated on sma_fast.notna() etc.), but additionally enforce
        # that no trades open before ``startup`` candles.
        first_tradable_idx = enriched.index[startup]

        # 6. Windowing
        if self.config.start:
            start_ts = pd.Timestamp(self.config.start, tz="UTC")
            enriched = enriched[enriched.index >= start_ts]
        if self.config.end:
            end_ts = pd.Timestamp(self.config.end, tz="UTC")
            enriched = enriched[enriched.index <= end_ts]
        if len(enriched) == 0:
            errors.append("Backtest window is empty after applying start/end filters")
            return _empty_result(run_id, strategy, self.config, data, validation, errors, warnings)

        # 7. Main event loop
        cash = float(self.config.initial_cash)
        trades: list[Trade] = []
        open_trade: Trade | None = None
        # Allow multiple positions? Strategy specifies max_open_trades; for simplicity
        # single-pair spot only => max 1 open trade at a time.
        max_open = max(1, int(self.config.max_open_trades or 1))
        if max_open > 1:
            warnings.append("Single-pair engine enforces max_open_trades=1")
            max_open = 1

        # Per-step bookkeeping
        equity_records = []
        pending_orders: list[Order] = []  # orders to fill at the next candle's open

        # Pre-compute ATR for volatility slippage if requested.
        atr_series: pd.Series | None = None
        if self.config.slippage_model == SlippageModel.VOLATILITY:
            atr_series = ATR(enriched["high"], enriched["low"], enriched["close"], 14)

        stoploss = (
            self.config.stoploss if self.config.stoploss is not None else float(strategy.stoploss)
        )
        trailing = bool(self.config.trailing_stop or strategy.trailing_stop)
        trailing_offset = float(
            self.config.trailing_stop_positive_offset or strategy.trailing_stop_positive_offset
        )
        trailing_pos = float(self.config.trailing_stop_positive or strategy.trailing_stop_positive)

        # ROI table: keys are str minutes, values are target profit ratios.
        roi_table: dict[int, float] = {}
        for k, v in (self.config.minimal_roi or strategy.minimal_roi or {}).items():
            try:
                roi_table[int(k)] = float(v)
            except (ValueError, TypeError):
                warnings.append(f"Invalid ROI entry {k}:{v} ignored")

        indexes = list(enriched.index)

        def _apply_slippage(price: float, side: OrderSide, i: int) -> float:
            if self.config.slippage_model == SlippageModel.NONE:
                return price
            if self.config.slippage_model == SlippageModel.FIXED_BPS:
                slip = price * self.config.slippage_bps / 10_000.0
                return price + slip if side == OrderSide.BUY else price - slip
            if self.config.slippage_model == SlippageModel.VOLATILITY and atr_series is not None:
                atr_val = (
                    float(atr_series.iloc[i])
                    if i < len(atr_series) and not np.isnan(atr_series.iloc[i])
                    else 0.0
                )
                slip = atr_val * self.config.slippage_atr_fraction
                return price + slip if side == OrderSide.BUY else price - slip
            return price

        def _buy_notional(curr_cash: float) -> float:
            if self.config.stake_amount is not None:
                return min(float(self.config.stake_amount), curr_cash)
            return curr_cash * float(self.config.stake_pct)

        def _fill_buy(order: Order, price: float, at: pd.Timestamp) -> None:
            nonlocal cash
            fee = price * order.quantity * self.config.fee_rate
            cost = price * order.quantity + fee
            if cost > cash + 1e-9:
                # Reduce quantity to fit available cash
                order.quantity = (cash) / (price * (1.0 + self.config.fee_rate))
                fee = price * order.quantity * self.config.fee_rate
                cost = price * order.quantity + fee
            cash -= cost
            order.fill(price, fee, filled_at=at)

        def _fill_sell(order: Order, price: float, at: pd.Timestamp) -> None:
            nonlocal cash
            fee = price * order.quantity * self.config.fee_rate
            proceeds = price * order.quantity - fee
            cash += proceeds
            order.fill(price, fee, filled_at=at)

        def _mark_to_market(close: float) -> float:
            eq = cash
            if open_trade is not None:
                # Unrealised PnL using current close, subtract estimated exit fee.
                eq += close * open_trade.quantity * (1.0 - self.config.fee_rate)
            return eq

        # Iterate over candles; at candle i we observe the close and decide what
        # orders to place at candle i+1.
        float(self.config.initial_cash)
        timeframe_to_timedelta(data.timeframe)

        for i, ts in enumerate(indexes):
            row = enriched.iloc[i]
            is_warmup = ts < first_tradable_idx
            # Prices available at this candle: ohlcv are fully known now (close at ts).
            # Any pending market orders from previous candle close fill at THIS
            # candle's open (or same close / next close per fill model).
            if self.config.fill_model == FillModel.NEXT_OPEN:
                fill_price_source = "open"
            elif self.config.fill_model == FillModel.SAME_CLOSE:
                fill_price_source = "close"
            else:
                fill_price_source = "open"  # NEXT_CLOSE handled below

            if i > 0 and pending_orders and not is_warmup:
                exec_ts = ts
                if self.config.fill_model == FillModel.NEXT_CLOSE:
                    # defer to close of this candle (handled below)
                    pass
                else:
                    fill_price_base = float(row["open"])
                    if fill_price_source == "close":
                        fill_price_base = float(row["close"])
                    # Process stop orders first (if any), then market orders.
                    remaining: list[Order] = []
                    for ord_ in pending_orders:
                        if open_trade is None and ord_.side == OrderSide.SELL:
                            continue  # nothing to sell (shouldn't happen)
                        if ord_.type == OrderType.STOP and open_trade is not None:
                            # Check if stop would have been hit intrabar.
                            lo = float(row["low"])
                            op = float(row["open"])
                            stop = float(ord_.stop_price or open_trade.enter_price)
                            if lo <= stop:
                                # Stop triggered: price is stop if open > stop else open.
                                fillp = op if op <= stop else stop
                                fillp = _apply_slippage(fillp, OrderSide.SELL, i)
                                _fill_sell(ord_, fillp, exec_ts)
                                open_trade.close(ord_)
                                trades.append(open_trade)
                                open_trade = None
                            else:
                                remaining.append(ord_)
                            continue
                        if ord_.side == OrderSide.BUY:
                            fillp = _apply_slippage(fill_price_base, OrderSide.BUY, i)
                            qty = ord_.quantity
                            notional = _buy_notional(cash)
                            qty = notional / fillp
                            ord_.quantity = qty
                            _fill_buy(ord_, fillp, exec_ts)
                            open_trade = Trade.open_new(data.symbol, ord_)
                            static_stop = (
                                open_trade.enter_price * (1.0 + stoploss) if stoploss < 0 else None
                            )
                            open_trade.stop_price = static_stop
                            # Immediately check for intra-candle stop on the fill candle.
                            # If the current candle's open already gaps below the stop, close at open.
                            if static_stop is not None:
                                lo = float(row["low"])
                                op = float(row["open"])
                                if op <= static_stop or lo <= static_stop:
                                    stop_fillp = op if op <= static_stop else static_stop
                                    stop_fillp = _apply_slippage(stop_fillp, OrderSide.SELL, i)
                                    sell = Order.new_market_sell(
                                        exec_ts, open_trade.quantity, reason="stoploss"
                                    )
                                    _fill_sell(sell, stop_fillp, exec_ts)
                                    open_trade.close(sell)
                                    trades.append(open_trade)
                                    open_trade = None
                                else:
                                    # Queue a stop for subsequent candles.
                                    pending_orders.append(
                                        Order.new_stop_sell(
                                            exec_ts,
                                            ord_.quantity,
                                            float(static_stop),
                                            reason="stoploss",
                                        )
                                    )
                        elif ord_.side == OrderSide.SELL and open_trade is not None:
                            fillp = _apply_slippage(fill_price_base, OrderSide.SELL, i)
                            ord_.quantity = open_trade.quantity
                            _fill_sell(ord_, fillp, exec_ts)
                            open_trade.close(ord_)
                            trades.append(open_trade)
                            open_trade = None
                    pending_orders = remaining

            # NEXT_CLOSE model: pending orders from previous step fill at this candle close,
            # before we emit new signals.
            if (
                self.config.fill_model == FillModel.NEXT_CLOSE
                and i > 0
                and pending_orders
                and not is_warmup
            ):
                fill_price_base = float(row["close"])
                for ord_ in pending_orders:
                    if ord_.side == OrderSide.BUY:
                        fillp = _apply_slippage(fill_price_base, OrderSide.BUY, i)
                        notional = _buy_notional(cash)
                        ord_.quantity = notional / fillp
                        _fill_buy(ord_, fillp, ts)
                        open_trade = Trade.open_new(data.symbol, ord_)
                        static_stop = (
                            open_trade.enter_price * (1.0 + stoploss) if stoploss < 0 else None
                        )
                        open_trade.stop_price = static_stop
                        if static_stop is not None:
                            lo = float(row["low"])
                            op = float(row["open"])
                            if op <= static_stop or lo <= static_stop:
                                stop_fillp = op if op <= static_stop else static_stop
                                stop_fillp = _apply_slippage(stop_fillp, OrderSide.SELL, i)
                                sell = Order.new_market_sell(
                                    ts, open_trade.quantity, reason="stoploss"
                                )
                                _fill_sell(sell, stop_fillp, ts)
                                open_trade.close(sell)
                                trades.append(open_trade)
                                open_trade = None
                            else:
                                pending_orders.append(
                                    Order.new_stop_sell(
                                        ts, ord_.quantity, float(static_stop), reason="stoploss"
                                    )
                                )
                    elif ord_.side == OrderSide.SELL and open_trade is not None:
                        fillp = _apply_slippage(fill_price_base, OrderSide.SELL, i)
                        ord_.quantity = open_trade.quantity
                        _fill_sell(ord_, fillp, ts)
                        open_trade.close(ord_)
                        trades.append(open_trade)
                        open_trade = None
                pending_orders = []

            # Evaluate existing open trade for stoploss / trailing / ROI / exit signal.
            if open_trade is not None and not is_warmup:
                current_rate = float(row["close"])
                current_profit = current_rate / open_trade.enter_price - 1.0
                open_trade.max_profit_pct = max(open_trade.max_profit_pct, current_profit)

                # Trailing stop update
                if trailing:
                    open_trade.update_trailing(current_rate, trailing_offset, trailing_pos)

                # Effective stop price (static stoploss or trailing)
                effective_stop: float | None = None
                if open_trade.stop_price is not None:
                    effective_stop = open_trade.stop_price
                elif stoploss < 0:
                    effective_stop = open_trade.enter_price * (1.0 + stoploss)

                # ROI check
                roi_hit = False
                if self.config.use_roi and roi_table:
                    hold_min = (ts - open_trade.enter_time).total_seconds() / 60.0
                    target = _roi_target(roi_table, hold_min)
                    if target is not None and current_profit >= target:
                        pending_orders.append(
                            Order.new_market_sell(
                                ts, open_trade.quantity, reason=f"roi@+{target:.3f}"
                            )
                        )
                        roi_hit = True

                # Custom exit hook
                custom_exit_hit = False
                try:
                    ce = strategy.custom_exit(open_trade, ts, current_rate, current_profit)
                    if ce:
                        pending_orders.append(
                            Order.new_market_sell(ts, open_trade.quantity, reason="custom_exit")
                        )
                        custom_exit_hit = True
                except Exception:
                    pass

                # Exit signal from dataframe
                exit_signal = bool(row.get("exit_long", False))
                if not roi_hit and not custom_exit_hit and exit_signal:
                    pending_orders.append(
                        Order.new_market_sell(ts, open_trade.quantity, reason="exit_signal")
                    )

                # Place/update stop if no market sell already pending
                pending_sell = any(o.side == OrderSide.SELL for o in pending_orders)
                if not pending_sell and effective_stop is not None:
                    # Check if stop would be hit on the NEXT candle (i+1) - but we can't know that now.
                    # Register a stop order; it will be checked next iteration.
                    # To keep it simple and avoid accumulating multiple stop orders, maintain at most one.
                    pending_orders = [o for o in pending_orders if o.type != OrderType.STOP]
                    pending_orders.append(
                        Order.new_stop_sell(
                            ts, open_trade.quantity, float(effective_stop), reason="stoploss"
                        )
                    )

            # Entry signal
            if open_trade is None and not is_warmup and not pending_orders:
                enter = bool(row.get("enter_long", False))
                if enter:
                    # Stake sizing
                    notional = _buy_notional(cash)
                    if notional > 0:
                        # We don't know fill price yet; placeholder quantity, corrected on fill.
                        qty = notional / float(row["close"])  # provisional
                        pending_orders.append(Order.new_market_buy(ts, qty, reason="entry_signal"))

            # Mark-to-market
            eq = _mark_to_market(float(row["close"]))
            equity_records.append(
                {
                    "timestamp": ts,
                    "cash": cash,
                    "close": float(row["close"]),
                    "equity": eq,
                    "open_trade": 1 if open_trade is not None else 0,
                }
            )

        # Close open trade at end of dataset at final close for mark-to-market metrics,
        # but tag it as forced liquidation.
        if open_trade is not None:
            final_row = enriched.iloc[-1]
            fillp = float(final_row["close"])
            sell = Order.new_market_sell(
                final_row.name, open_trade.quantity, reason="end_of_backtest"
            )
            _fill_sell(sell, fillp, final_row.name)
            open_trade.close(sell)
            trades.append(open_trade)
            open_trade = None
            warnings.append("An open position was force-closed at the end of the backtest window")
            # Update last equity record with final cash
            if equity_records:
                equity_records[-1]["cash"] = cash
                equity_records[-1]["equity"] = cash

        eq_df = pd.DataFrame(equity_records).set_index("timestamp")

        # Buy & hold benchmark
        bh_metrics = None
        if self.config.enable_buy_and_hold and len(eq_df) > 1:
            bh_qty = float(self.config.initial_cash) / float(enriched["close"].iloc[0])
            bh_fee_open = float(self.config.initial_cash) * self.config.fee_rate
            bh_qty = (float(self.config.initial_cash) - bh_fee_open) / float(
                enriched["close"].iloc[0]
            )
            bh_proceeds_final = (
                bh_qty * float(enriched["close"].iloc[-1]) * (1.0 - self.config.fee_rate)
            )
            bh_return = bh_proceeds_final / float(self.config.initial_cash) - 1.0
            bh_qty * enriched["close"] * (1.0 - self.config.fee_rate)
            bh_metrics = {"return_pct": bh_return * 100.0, "final_value": bh_proceeds_final}

        from ..analytics.metrics import compute_metrics  # noqa: WPS433 (lazy to break cycle)

        metrics = compute_metrics(
            equity_curve=eq_df["equity"],
            trades=[t for t in trades if not t.is_open],
            initial_capital=float(self.config.initial_cash),
            fee_rate=self.config.fee_rate,
            buy_and_hold=bh_metrics,
            exposure_bars=eq_df["open_trade"],
        )

        # Replace metrics dict later when exporting; we keep the typed object here.
        result = BacktestResult(
            run_id=run_id,
            strategy_info=strategy.info(),
            config=self.config.to_dict(),
            dataset_info={
                "exchange": data.exchange,
                "symbol": data.symbol,
                "timeframe": data.timeframe,
                "source_path": data.metadata.source_path,
                "source_sha256": data.metadata.source_sha256,
                "start": data.metadata.start_utc.isoformat() if data.metadata.start_utc else None,
                "end": data.metadata.end_utc.isoformat() if data.metadata.end_utc else None,
                "rows": len(data),
            },
            validation=validation.to_dict(),
            trades=[t.to_dict() for t in trades],
            equity_curve=eq_df,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
        )
        return result


def _new_run_id() -> str:
    import time
    import uuid

    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _roi_target(roi_table: dict[int, float], hold_minutes: float) -> float | None:
    """Return the applicable ROI target for the given holding time (minutes)."""
    if not roi_table:
        return None
    # roi_table keys are minutes; find the largest key <= hold_minutes
    candidates = [(m, t) for m, t in roi_table.items() if float(m) <= hold_minutes]
    if not candidates:
        return None
    # Freqtrade semantics: most-recently-crossed threshold applies i.e. largest m.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return float(candidates[0][1])


def _empty_result(
    run_id: str,
    strategy: Strategy,
    cfg: EngineConfig,
    data: OHLCV,
    validation: DataValidationReport,
    errors: list[str],
    warnings: list[str],
) -> BacktestResult:
    from ..analytics.metrics import MetricsSummary as _MS  # noqa: WPS433

    empty_eq = pd.DataFrame(columns=["cash", "close", "equity", "open_trade"])
    metrics = _MS(
        initial_capital=float(cfg.initial_cash),
        final_equity=float(cfg.initial_cash),
        net_profit=0.0,
        return_pct=0.0,
        total_fees=0.0,
        slippage_total=0.0,
        closed_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        avg_trade_return=0.0,
        median_trade_return=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        largest_win=0.0,
        largest_loss=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        time_in_market_pct=0.0,
        buy_and_hold_return_pct=None,
    )
    return BacktestResult(
        run_id=run_id,
        strategy_info=strategy.info(),
        config=cfg.to_dict(),
        dataset_info={
            "exchange": data.exchange,
            "symbol": data.symbol,
            "timeframe": data.timeframe,
            "source_path": data.metadata.source_path,
            "source_sha256": data.metadata.source_sha256,
            "rows": len(data),
        },
        validation=validation.to_dict(),
        trades=[],
        equity_curve=empty_eq,
        metrics=metrics,
        warnings=warnings,
        errors=errors,
    )
