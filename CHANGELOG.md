# Changelog

All notable changes to ICBT are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-11

### Added
- Initial release of ICBT, an independent cryptocurrency spot backtesting
  engine.
- OHLCV CSV loader with strict validation (OHLC invariants, ordering,
  duplicates, gaps, UTC normalisation, SHA-256 provenance).
- Timeframe utilities: pandas-freq conversion, OHLCV resampling,
  no-lookahead `merge_informative` helper.
- Strategy interface (`icbt.strategy.base.Strategy`) with support for
  informative timeframes, BTC informatives, custom exits, and stable
  `StrategyConfig`.
- Pure-pandas/numpy indicator library (SMA, EMA, RSI, MACD, BBands, ATR,
  StochRSI, MFI, CCI, Williams %R, CMF, ADX, ROC, EWO, cross helpers).
- `SmaCross` reference strategy for tests/examples.
- `NostalgiaForInfinityX8` ported strategy with documented compatibility
  (multi-indicator long-normal entry/exit, multi-TF and BTC informative
  support); see `docs/strategy-port.md`.
- Event-driven single-pair spot backtesting engine with deterministic
  execution, configurable fill models (next_open / next_close / same_close),
  fixed-bps slippage, fees, stoploss, trailing stop, ROI table, intrabar
  stop-fill semantics, and end-of-window force-close.
- Performance analytics: initial/final equity, net profit, return %,
  gross profit/loss, fees, win rate, profit factor, max drawdown,
  average/median trade return, avg win/loss, time-in-market, buy-and-hold
  benchmark, Sharpe, Sortino, equity curve.
- CLI via Typer (`icbt validate|info|run|list-strategies|list-runs|gui`).
- FastAPI web dashboard with dataset browser, validator, run configuration,
  equity chart, trade table, run history, and CSV/JSON artifact downloads.
- Run persistence under `runs/<run_id>/` with a JSONL ledger (never
  overwrites).
- Windows PowerShell (`scripts/launch.ps1`) and POSIX (`scripts/launch.sh`)
  launch scripts.
- Synthetic deterministic data generator for tests and reproducible
  examples (clearly labeled as synthetic).
- Unit, negative, integration, and e2e test suite (23 tests).
- GitHub Actions CI workflow (lint, type check, tests).
- Documentation (README, architecture, data policy, strategy port,
  changelog).

### Known limitations
- Single-pair spot-long only in v1.
- NFI-X8 port implements the architectural framework and a representative
  long-normal entry/exit rule; many original tags/features are not ported
  (see `docs/strategy-port.md`).
