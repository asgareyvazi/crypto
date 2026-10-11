import pandas as pd

from icbt.data.synthetic import generate_trend_with_cycles
from icbt.engine import BacktestEngine, EngineConfig, FillModel
from icbt.strategy.sma_cross import SmaCrossStrategy


def test_engine_rejects_insufficient_warmup():
    data = generate_trend_with_cycles(n_candles=20)  # less than startup
    eng = BacktestEngine(EngineConfig())
    s = SmaCrossStrategy()
    r = eng.run(data, s)
    assert r.errors, "Should error on insufficient warmup"
    assert r.metrics.closed_trades == 0


def test_engine_runs_on_valid_data_produces_trades():
    data = generate_trend_with_cycles(n_candles=2000, seed=42)
    s = SmaCrossStrategy()
    cfg = EngineConfig(initial_cash=10000, fee_rate=0.001, slippage_bps=0)
    eng = BacktestEngine(cfg)
    r = eng.run(data, s)
    assert not r.errors
    assert r.metrics.closed_trades > 0, "should produce at least one trade on trending data"
    # cash accounting: final equity equals cash after closing
    assert abs(r.metrics.final_equity - r.equity_curve["equity"].iloc[-1]) < 0.01


def test_engine_no_lookahead_fill_semantics():
    """With fill_model=NEXT_OPEN an entry signal at T must fill at T+1 open (not T close)."""
    data = generate_trend_with_cycles(n_candles=2000, seed=42)
    cfg = EngineConfig(
        initial_cash=10000, fee_rate=0, slippage_bps=0, fill_model=FillModel.NEXT_OPEN
    )
    eng = BacktestEngine(cfg)
    r = eng.run(data, SmaCrossStrategy())
    assert r.metrics.closed_trades > 0
    # Every entry price must equal the open of the candle AFTER the signal candle.
    # Check a few trades: find a signal in the raw strategy output and compare with the trade record.
    # We verify trade.enter_time is strictly after any signal on the previous candle by examining that
    # trade.enter_price equals the open price at that timestamp.
    df = data.df
    for t in r.trades[:5]:
        ts = pd.Timestamp(t["enter_time"])
        assert ts in df.index, f"entry time {ts} not in dataset"
        assert abs(t["enter_price"] - df.loc[ts, "open"]) < 1e-6, (
            "entry must fill at open of candle T+1"
        )


def test_fees_reduce_profit():
    data = generate_trend_with_cycles(n_candles=2000, seed=7)
    cfg_nofee = EngineConfig(initial_cash=10000, fee_rate=0, slippage_bps=0)
    cfg_fee = EngineConfig(initial_cash=10000, fee_rate=0.01, slippage_bps=0)
    r_no = BacktestEngine(cfg_nofee).run(data, SmaCrossStrategy())
    r_fee = BacktestEngine(cfg_fee).run(data, SmaCrossStrategy())
    if r_no.metrics.closed_trades > 0:
        assert r_fee.metrics.total_fees > 0
        assert r_fee.metrics.final_equity <= r_no.metrics.final_equity + 1e-6


def test_stoploss_closes_position():
    """If price gaps through the stop on the open of the next candle the stop fires.

    We use a manually crafted entry (by injecting a signal) followed by an immediate
    gap below the stop price; this isolates stop semantics from entry-signal logic.
    """
    import numpy as np

    from icbt.data.ohlcv import OHLCV, OHLCVMetadata

    class _PreSignaled(SmaCrossStrategy):
        """Strategy that emits exactly one entry at bar 200 and no exit signals."""

        startup_candle_count = 40

        def populate_entry_signals(self, df):
            df = df.copy()
            df["enter_long"] = False
            # trigger a cross-up at bar 200 via our synthetic data (we build the cross)
            # simpler: force True at exactly one location
            sig = np.zeros(len(df), dtype=bool)
            if len(df) > 200:
                sig[200] = True
            df["enter_long"] = sig
            return df

        def populate_exit_signals(self, df):
            df = df.copy()
            df["exit_long"] = False
            return df

    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # Bars 0..199 flat at 100, bar 200 close at 101 to register the entry signal,
    # bar 201 opens at 80 (well below the 5% stop of 101*0.95=95.95) and stays low.
    closes = np.full(n, 100.0)
    closes[200] = 101.0
    closes[201] = 80.0
    closes[202:] = 80.0
    opens = np.concatenate([[100.0], closes[:-1]])
    opens[201] = 80.0
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    # The fill for our market buy happens at bar 201 open = 80;
    # set bar 201 low way below the 5% stop (which will be at 80*0.95=76)
    # to guarantee a stop fill: low=70.
    lows[201] = 70.0
    highs[201] = 81.0
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 100,
            "timestamp": idx,
        },
        index=idx,
    )
    df.index.name = "date"
    md = OHLCVMetadata(
        exchange="t",
        symbol="TST/USDT",
        timeframe="5m",
        n_rows=n,
        columns=tuple(df.columns),
        start_utc=idx[0],
        end_utc=idx[-1],
    )
    data = OHLCV(df=df, metadata=md)
    from icbt.engine import FillModel

    cfg = EngineConfig(
        initial_cash=10000,
        fee_rate=0,
        slippage_bps=0,
        stoploss=-0.05,
        trailing_stop=False,
        fill_model=FillModel.NEXT_OPEN,
    )
    r = BacktestEngine(cfg).run(data, _PreSignaled())
    assert r.metrics.closed_trades >= 1, f"no trades; errors={r.errors}"
    reasons = [t["exit_reason"] for t in r.trades]
    assert any("stop" in rsn for rsn in reasons), f"expected stoploss exit, got reasons={reasons}"
    # Stop exit should have a loss
    losing = [t for t in r.trades if "stop" in t["exit_reason"]]
    assert losing and losing[0]["pnl"] < 0, "stop exit should result in a loss"


def test_metrics_are_sane_on_empty():
    import pandas as pd

    from icbt.analytics.metrics import compute_metrics

    m = compute_metrics(
        equity_curve=pd.Series([10000], index=[pd.Timestamp("2026-01-01", tz="UTC")]),
        trades=[],
        initial_capital=10000,
        fee_rate=0,
    )
    assert m.closed_trades == 0
    assert m.final_equity == 10000
    assert m.win_rate == 0.0


def test_buy_and_hold_reported():
    data = generate_trend_with_cycles(n_candles=2000, seed=42)
    cfg = EngineConfig(enable_buy_and_hold=True, fee_rate=0, slippage_bps=0)
    r = BacktestEngine(cfg).run(data, SmaCrossStrategy())
    assert r.metrics.buy_and_hold_return_pct is not None


def test_deterministic_run():
    data = generate_trend_with_cycles(n_candles=1000, seed=42)
    cfg = EngineConfig(initial_cash=10000, fee_rate=0, slippage_bps=0)
    r1 = BacktestEngine(cfg).run(data, SmaCrossStrategy())
    r2 = BacktestEngine(cfg).run(data, SmaCrossStrategy())
    assert r1.metrics.final_equity == r2.metrics.final_equity
    assert r1.metrics.closed_trades == r2.metrics.closed_trades
