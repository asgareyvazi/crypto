# ICBT — Independent Cryptocurrency Backtester

A self-contained, independently runnable cryptocurrency spot-trading backtesting
application with a local web dashboard, CLI, reproducible artifacts, and a
strategy interface that ships two reference strategies:

* `SmaCross` — a simple SMA-crossover reference strategy (fast on tests/examples).
* `NostalgiaForInfinityX8 (NFI-X8)` — an **architectural port** of the public
  `iterativv/NostalgiaForInfinity` NFI-X8 strategy. See
  [`docs/strategy-port.md`](docs/strategy-port.md) for the compatibility matrix
  and the traceability notes.

The engine is event-driven, deterministic, and enforces strict no-lookahead
semantics (signals emit at candle close, fills occur on the next candle's open
by default, and informative-timeframe joins are carefully aligned to avoid
leaking future data).

---

## Quick start (Windows)

1. Install Python 3.11 (or newer) from https://www.python.org/downloads/.
2. Open PowerShell, change into the project directory.
3. Run:

   ```powershell
   .\scripts\launch.ps1
   ```

   That creates a virtual environment, installs dependencies, and opens the
   dashboard at http://127.0.0.1:8765.
4. To stop the dashboard press `Ctrl+C`.

To run the CLI instead:

```powershell
.\scripts\launch.ps1 -Cli              # prints icbt --help
.\scripts\launch.ps1 -Cli run data\synthetic_btc_5m.csv --strategy SmaCross
```

## Quick start (Linux / macOS)

```bash
./scripts/launch.sh             # launches GUI on http://127.0.0.1:8765
./scripts/launch.sh cli run data/synthetic_btc_5m.csv --strategy SmaCross
```

## Manual installation

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e '.[dev]'

icbt --help
icbt gui                      # start dashboard
```

## CLI reference

```bash
icbt list-strategies                                       # list bundled strategies
icbt validate path/to/candles.csv                          # strict data validation
icbt info path/to/candles.csv                              # metadata + validation summary
icbt run path/to/candles.csv --strategy SmaCross --cash 10000
icbt run path/to/candles.csv --strategy NFI-X8 --btc btc_5m.csv
icbt list-runs                                             # show historical runs
icbt gui                                                   # launch web dashboard
```

Key flags for `icbt run`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--cash` | 10000 | Initial quote-currency capital |
| `--stake-pct` | 0.10 | Fraction of free cash per trade |
| `--stake-amount` | None | Fixed absolute stake (overrides stake-pct) |
| `--fee` | 0.001 | Flat fee rate (10 bps default) |
| `--slippage-bps` | 5 | Fixed adverse slippage in basis points |
| `--fill-model` | `next_open` | `next_open`, `next_close` or `same_close` (see docs) |
| `--stoploss` | (strategy) | Override stoploss fraction (e.g. -0.05) |
| `--btc` | None | BTC OHLCV CSV for strategies that need BTC informatives |
| `--start` / `--end` | None | ISO-8601 UTC window bounds |
| `--fail-on-gaps` | false | Abort the run on validation errors |

## Dashboard

The FastAPI dashboard is the primary UI. It lets you:

* Browse, upload, and validate OHLCV CSV datasets.
* Configure strategy, capital, fees, slippage, fill model, and window.
* Launch a backtest and watch equity/drawdown/trades render live.
* Inspect trade history, download CSV/JSON artifacts.
* Browse and compare previous runs.

## Data format and data-quality policy

ICBT accepts CSV files with the columns:

```
timestamp, open, high, low, close, volume
```

`timestamp` is parsed as integer ms, integer seconds, or ISO-8601 strings and
normalised to UTC. File names of the form `candles_<exchange>_<symbol>_<tf>.csv`
populate metadata automatically.

The validator enforces a strict policy (see
[`docs/data-policy.md`](docs/data-policy.md)):

* Required columns present; numeric values parseable.
* `high >= max(open, close) >= min(open, close) >= low > 0`; `volume >= 0`.
* Timestamps strictly increasing; no duplicates.
* Candle interval within 5% of the nominal timeframe.
* Missing-candle gaps detected against the expected grid; reported as
  warnings by default, or hard errors with `--fail-on-gaps`.

**ICBT never fabricates missing candles or forward-fills prices to make a
backtest "pass".** A dataset with gaps is reported as such; the user must
either adjust the window or explicitly acknowledge the gaps.

## Fill & execution semantics (documented)

* Signals are evaluated at candle close `T` using data available at or before
  `T` (no lookahead).
* Default fill model **NEXT_OPEN**: market orders are filled at the open of
  candle `T+1`, at `open_(T+1) * (1 +/- slippage)` with `notional * fee`
  charged.
* Stop-loss: if the next candle's low breaches the stop, the fill price is
  `max(open, stop) - slippage` (conservative: the open is taken first if it
  gaps through the stop; otherwise the stop triggers at the stop price).
* Trailing stop follows the Freqtrade convention: activates after reaching
  `trailing_stop_positive_offset` profit, then trails at `trailing_stop_positive`.
* ROI table is keyed by minutes held (same schema as Freqtrade
  `minimal_roi`).
* Open positions at the end of the test window are force-closed at the final
  close and tagged `end_of_backtest`.
* Single-pair only (spot-long); `max_open_trades > 1` is coerced to 1 on a
  single-pair dataset.

## Outputs and reproducibility

Every run produces a unique `runs/<run_id>/` directory containing:

* `summary.txt` — human-readable result summary.
* `trades.csv` — closed trades with entry/exit prices, fees, pnl, hold time,
  reasons.
* `equity.csv` — per-candle mark-to-market equity curve.
* `result.json` — full metrics + trade/equity payload.
* `config.json` — configuration, dataset hash, validation outcome, warnings.

A master index (`runs/_index.jsonl`) records every run with strategy, dataset,
config hash, output paths, and metrics. Runs are never overwritten.

## Strategy API

Strategies subclass `icbt.strategy.base.Strategy` and implement
`populate_indicators`, `populate_entry_signals`, and `populate_exit_signals`.
Optional hooks include `custom_exit` and `custom_stoploss`. See
[`docs/architecture.md`](docs/architecture.md) and the bundled
`SmaCrossStrategy` / `NostalgiaForInfinityX8` for examples.

## Testing

```bash
pytest tests -v
ruff check src tests
```

The test suite covers unit, negative, integration, and end-to-end tests
(including real CLI invocations and the FastAPI GUI exercised through
`starlette.testclient`). See `tests/`.

## Layout

```
src/icbt/
    data/             CSV ingestion, validation, timeframe resampling, synthetic fixtures
    strategy/         base interface + indicators + bundled strategies (SmaCross, NFI-X8)
    engine/           event-driven backtester, orders/trades, metrics wiring, persistence
    analytics/        metric calculations, reporting, CSV/JSON export
    cli/              Typer CLI
    api/              FastAPI app factory
    ui/               Jinja2 templates and static assets for the dashboard
tests/                unit / negative / integration / e2e tests
scripts/              Windows PowerShell and POSIX launchers
docs/                 architecture, strategy port notes, data policy, examples
```

## Known limitations

* **NFI-X8 parity** — the full original strategy is ~61,000 lines of
  Freqtrade-specific Python relying on Freqtrade runtime hooks, DCA
  (`adjust_trade_position`), bad-trade controller, 20+ entry-tag families,
  protective pairs, blacklists, and the `pandas_ta`/`TA-Lib` indicator
  catalogue. This repository ships an independent port of the strategy
  *architecture* (metadata, timeframes, startup, indicator categories,
  multi-factor entry/exit framework) and a representative multi-factor
  long-normal entry rule — not a line-for-line reproduction. See
  [`docs/strategy-port.md`](docs/strategy-port.md) for the exact compatibility
  matrix and unsupported features.
* **Single-pair, spot-long only** in v1. Multi-pair portfolio simulation and
  shorting are not yet implemented.
* **No real exchange adapters** — ICBT consumes CSV files. A separate data
  collection tool can write compatible CSVs.
* **Synthetic fixture data is clearly labelled**; never treat it as an
  exchange observation.

## License

MIT
