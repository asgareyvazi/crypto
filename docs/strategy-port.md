# NFI-X8 strategy port: traceability and compatibility

This document describes what was ported from the public
[iterativv/NostalgiaForInfinity](https://github.com/iterativv/NostalgiaForInfinity)
`NostalgiaForInfinityX8.py` strategy (`v18.0.124`, commit tip of `main`) and
what remains **unsupported** in this independent engine.

## Metadata ported verbatim

| Attribute | Value in source | ICBT |
|-----------|-----------------|------|
| `timeframe` | `"5m"` | `5m` |
| `info_timeframes` | `["15m","1h","4h","1d"]` | `["15m","1h","4h","1d"]` |
| `btc_info_timeframes` | `["4h"]` | `["4h"]` |
| `startup_candle_count` | `800` (adjusted per informative) | `800` |
| `stoploss` | `-0.99` (effectively disabled; custom_exit drives exits) | `-0.99` default |
| `trailing_stop` | `False` | `False` default |
| `use_exit_signal` | `True` | `True` |
| `ignore_roi_if_entry_signal` | `True` | not yet implemented (ROI still respected) |

## Indicator framework ported

ICBT re-implements the following indicator families used across NFI-X8's
informative functions:

| Indicator | Source reference | ICBT function |
|-----------|------------------|---------------|
| Simple moving average | `ta.SMA` | `SMA` |
| Exponential moving average | `ta.EMA` | `EMA` |
| Relative Strength Index (Wilder) | `ta.RSI` | `RSI` |
| Moving Average Convergence/Divergence | `ta.MACD` | `MACD` |
| Bollinger Bands | `ta.BBANDS` | `BBANDS` |
| Average True Range | `ta.ATR` | `ATR` |
| Stochastic RSI | custom | `STOCHRSI` |
| Money Flow Index | `ta.MFI` | `MFI` |
| Commodity Channel Index | `ta.CCI` | `CCI` |
| Williams %R | `ta.WILLR` | `WILLR` |
| Chaikin Money Flow | custom | `CMF` |
| Average Directional Index | `ta.ADX`, `ta.PLUS_DI/MINUS_DI` | `ADX` |
| Rate of Change | `ta.ROC` | `ROC` |
| Elliott Wave Oscillator | `close.rolling(5).mean() - close.rolling(35).mean()` | `EWO` |
| Volume moving average | `volume.rolling(n).mean()` | `SMA(volume, n)` |
| Cross-up / cross-down helpers | `(a.shift(1) < b.shift(1)) & (a > b)` | `cross_up`, `cross_down` |

These are implemented in pure pandas/numpy (no TA-Lib required), which keeps
ICBT easy to install on Windows. Numerical parity with TA-Lib is "close but
not guaranteed to be bit-identical" due to minor differences in Wilder
smoothing initialisation; for validation this is documented and treated as
an expected (small) deviation, not a bug.

## Entry rules ported

The original NFI-X8 has 50+ long entry tags grouped into "normal", "pump",
"quick", "rebuy", "high profit", "rapid", "grind", "btc", "top coins", and
"scalp" families. Each tag has bespoke multi-factor conditions referencing
dozens of indicators across four informative pairs plus BTC, volume filters,
protection guards, and per-tag cooldowns.

This port implements a **representative long-normal style entry rule** that
encodes the same *categories* of conditions used across tags `1`–`13`:

1. **Trend alignment** — price above EMA(50), EMA(50) above EMA(200) on the
   base 5m frame; higher-TF EMA(50) > EMA(200) on 1h and 4h when those
   informatives are present.
2. **Oversold retracement** — RSI below the entry threshold, MFI oversold,
   CCI deeply negative (multi-indicator confluence).
3. **Reversal confirmation** — StochRSI K crosses above D in oversold
   territory and EWO ticks upward.
4. **Volume confirmation** — current volume at least 80% of its recent
   average.
5. **Trend strength filter** — ADX above minimum.
6. **BTC 4h bearish protection** — no entries when BTC is trading below its
   4h EMA(50) with 4h RSI below 45 (when BTC data is supplied).

This entry rule is **explicitly not** a line-for-line copy of any single tag,
and it will not reproduce NFI-X8's exact historical trades. It is a
conservative, architecturally-faithful representative that exercises the same
indicator / timeframe / risk-control infrastructure used by the real
strategy, so that the engine, informative joins, risk controls and analytics
all run against a genuine multi-timeframe strategy rather than a toy example.

## Exit rules ported

NFI-X8 exits are largely driven by its `custom_exit` logic (profit targets,
bad-trade controller, EWO rollover, multi-indicator overbought) and uses
ROI/minus_stoploss only as broad guards. This port implements:

1. **Overbought exit** — RSI > threshold AND price tags the upper Bollinger
   Band (bb_pct > 0.9).
2. **EMA rollover exit** — fast EMA(20) crosses below mid EMA(50).
3. **Momentum exhaustion exit** — Williams %R > -10 (extreme overbought)
   while EWO is rolling down and RSI is elevated.

The protective `custom_stoploss`, trailing, and bad-trade controller of the
original are *not* ported; ICBT uses the static stoploss / trailing-stop /
ROI framework of `EngineConfig` instead. If you want exact NFI-X8 exits you
must run the original Freqtrade strategy in Freqtrade.

## Known unsupported features

The following features of NFI-X8 are **not implemented** in ICBT. They are
called out here rather than silently stubbed:

* DCA / position scaling (`adjust_trade_position`, rebuy tags).
* Bad-trade controller (`_bad_trade_controller_*`).
* Per-tag ROI scaling and `custom_exit`'s profit-target logic (
  `mark_profit_target`, `exit_profit_target`).
* Downtime protection, age filter, hold/trades.json support.
* Multi-pair entry selection / volume pairlist logic.
* Short entries/tags (ICBT is spot-long only in v1).
* Per-trade `custom_stoploss` dynamic schedules.
* Protective pairs and blacklists.
* All tag IDs 21–173 (pump/quick/rebuy/rapid/grind/btc/top-coins/scalp).

If you attempt to run the ported strategy without supplying a BTC 4h
informative, the engine **fails closed** with a clear error rather than
silently dropping the BTC filter.

## Traceability references

* Source: `iterativv/NostalgiaForInfinity/NostalgiaForInfinityX8.py`
  (https://github.com/iterativv/NostalgiaForInfinity).
* ICBT port: `src/icbt/strategy/nfi_x8.py`.
* Indicator implementations: `src/icbt/strategy/indicators.py`.
* Informative join semantics: `src/icbt/data/timeframes.py::merge_informative`
  and tests in `tests/unit/test_data.py::test_merge_informative_no_lookahead`.

If you need exact 1:1 parity with a specific NFI-X8 release, run the original
`.py` strategy inside Freqtrade. ICBT is an independent engine and does not
bundle Freqtrade runtime components; the port is provided so that users can
experiment with the engine architecture, multi-timeframe joins, strict data
validation, and the dashboard on a real multi-indicator strategy.
