# Troubleshooting

## GUI doesn't open in browser

The PowerShell script opens the server at http://127.0.0.1:8765; navigate to that
URL manually if the browser doesn't auto-launch. If port 8765 is busy, pass a
different port:

```powershell
.\scripts\launch.ps1   # uses 8765
$env:PORT=9000; .\scripts\launch.ps1
```

## `ModuleNotFoundError` for `icbt`

Always activate the virtual env or use the launch scripts. A fresh install:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'
```

## CSV fails validation with `INVALID_OHLC`, `DUPLICATE_TS`, or `GAPS`

Open the summary from `icbt validate <file>` and check the issue list. Common
causes:

* Exchanges sometimes emit zero-volume or zero-price candles on maintenance
  boundaries. Strip those rows before running.
* Adjacent files concatenated out of order will produce duplicate or
  out-of-order timestamps. Sort and de-duplicate (the validator will tell you
  exactly where).
* Gaps larger than one candle interval indicate missing data. Either narrow the
  backtest window with `--start`/`--end` or run with `--fail-on-gaps` to abort
  loudly.

## "Strategy requires BTC informative(s) ... but no BTC data supplied"

NFI-X8 needs BTC/USDT candles for the BTC 4h informative. Pass `--btc
path/to/btc_5m.csv`. The engine fails closed instead of silently dropping the
BTC filter.

## Results look too good (or too bad)

Check:

* Did you use the synthetic fixture (filename starts with `synthetic_`)?
  Synthetic data is for software testing and does not represent real exchange
  behaviour.
* Default slippage and fees are conservative (5 bps slip, 10 bps fee). For
  high-fee exchanges raise `--fee`.

## Reproducing a previous run

Each `runs/<run_id>/config.json` captures the full configuration. You can
re-run with the same CLI flags; results are deterministic for the same dataset
hash + config + engine version.
