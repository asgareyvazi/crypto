# Architecture & data flow

ICBT is layered as follows:

```
  ┌──────────────────────────────┐
  │  CLI (Typer)   GUI (FastAPI) │   User-facing surfaces
  └───────────┬──────────────────┘
              │
  ┌───────────▼──────────────────┐
  │    Service / API layer       │   Validation requests, run orchestration
  └───────────┬──────────────────┘
              │
  ┌───────────▼──────────────────┐
  │     BacktestEngine           │   Event-driven replay, order routing,
  │                              │   position accounting, risk
  └───────┬─────────┬────────────┘
          │         │
  ┌───────▼──┐  ┌───▼──────────┐    ┌──────────────────┐
  │ Data     │  │ Strategy     │    │ Analytics        │
  │ load /   │  │ interface +  │    │ metrics, reports │
  │ validate │  │ indicators   │    └──────────────────┘
  │ resample │  │ SmaCross,    │
  │ merge    │  │ NFI-X8       │
  └──────────┘  └──────────────┘
          │         │
          └────┬────┘
               ▼
          Persistence
          (runs/<id>/ artifacts,
           _index.jsonl ledger)
```

## Key modules

* `icbt.data.ohlcv` — `OHLCV` dataclass, CSV I/O, SHA-256 provenance,
  filename metadata inference.
* `icbt.data.validation` — strict validator producing
  `DataValidationReport` with issues categorised INFO/WARNING/ERROR.
* `icbt.data.timeframes` — timeframe delta parsing, OHLCV resampling
  (open=first, high=max, low=min, close=last, volume=sum), and the
  `merge_informative` helper that implements no-lookahead informative joins
  via `pandas.merge_asof` with direction='backward' on the shifted
  (right-edge-aligned) informative index.
* `icbt.strategy.base` — abstract `Strategy` base, `StrategyConfig`,
  `SignalType`.
* `icbt.strategy.indicators` — TA-Lib-free indicator library (SMA, EMA, RSI,
  MACD, BBands, ATR, StochRSI, MFI, CCI, Williams %R, CMF, ADX, ROC, EWO,
  cross detection).
* `icbt.engine.config` — `EngineConfig` (capital, fees, slippage, fill
  model, risk controls, window).
* `icbt.engine.orders` — `Order`, `Trade`, side/type/status enumerations.
* `icbt.engine.engine.BacktestEngine` — the event loop (see below).
* `icbt.analytics.metrics` — return, win rate, profit factor, max drawdown,
  Sharpe, Sortino, buy-and-hold benchmark, exposure.
* `icbt.analytics.report` — CSV/JSON/text writers and `write_run_artifacts`.
* `icbt.engine.persistence` — run ledger (`_index.jsonl`) and helpers to
  compute file hashes.
* `icbt.cli.main` — Typer application.
* `icbt.api.app` — FastAPI application factory with REST endpoints and
  Jinja2 dashboard.

## Event loop (pseudocode)

```
load + validate OHLCV
derive informative timeframes from base tf
merge informatives (no-lookahead)
strategy.populate_indicators(df)
strategy.populate_entry_signals(df); populate_exit_signals(df)
trim warm-up (startup_candle_count)
for each candle i with timestamp ts:
    # 1. Fill pending orders (from previous close) at this candle's open
    for stop orders in pending:
        if low <= stop_price: fill at max(open, stop_price) - slip; close trade
    for buy orders in pending:
        fill at open + slip; open new Trade; attach static stop
        if open gaps below stop on this candle: immediately close at open
    for market sell orders:
        fill at open - slip; close trade
    # 2. Evaluate open trade (on current close)
    update trailing stop
    if ROI target reached or custom_exit() or exit_long:
        queue market sell
    elif no market sell pending:
        place/update stop order
    # 3. Evaluate entry (on current close)
    if no open trade and enter_long:
        queue market buy (qty derived from stake sizing)
    # 4. Mark-to-market
    record equity at close
force-close any remaining open trade at final close (tagged end_of_backtest)
compute metrics and return BacktestResult
persist artifacts
```

## Determinism

Given the same dataset (same file hash), same engine config, same strategy
version, and same software version, the backtest produces byte-identical
equity curves and trade lists. There is no randomness in the engine; all
RPNG use is confined to synthetic data generation (which takes an explicit
seed).
