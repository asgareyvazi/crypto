"""Pure-pandas/numpy technical indicators (no TA-Lib required).

These implementations cover the subset used by NostalgiaForInfinityX8 and
provide commonly needed primitives (SMA/EMA/RSI/MACD/BBands/ATR/StochRSI,
etc.). They are not intended as a drop-in replacement for TA-Lib's entire
catalogue; every indicator used by any bundled strategy must be implemented
here or explicitly raise ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def SMA(series: pd.Series, period: int) -> pd.Series:  # noqa: N802
    return series.rolling(window=period, min_periods=period).mean()


def EMA(series: pd.Series, period: int) -> pd.Series:  # noqa: N802
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def RSI(series: pd.Series, period: int = 14) -> pd.Series:  # noqa: N802
    """Wilder RSI (matches TA-Lib timeperiod=period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)  # convention when loss==0 and we haven't had data yet? Use NaN.


def MACD(  # noqa: N802
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_sig = EMA(macd, signal)
    hist = macd - macd_sig
    return macd, macd_sig, hist


def BBANDS(  # noqa: N802
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = SMA(series, period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:  # noqa: N802
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def STOCHRSI(  # noqa: N802
    series: pd.Series,
    period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    rsi = RSI(series, period)
    rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100.0
    k = stoch.rolling(window=k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(window=d_smooth, min_periods=d_smooth).mean()
    return k, d


def TRUE_RANGE(df: pd.DataFrame) -> pd.Series:  # noqa: N802
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )


def EWO(series: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:  # noqa: N802
    """Elliott Wave Oscillator (SMA5 - SMA35)."""
    return SMA(series, fast) - SMA(series, slow)


def ROC(series: pd.Series, period: int) -> pd.Series:  # noqa: N802
    return (series - series.shift(period)) / series.shift(period).replace(0, np.nan) * 100.0


def MFI(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:  # noqa: N802
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta = tp.diff()
    pos = mf.where(delta > 0, 0.0)
    neg = mf.where(delta < 0, 0.0)
    pos_sum = pos.rolling(period, min_periods=period).sum()
    neg_sum = neg.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:  # noqa: N802
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def WILLR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:  # noqa: N802
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100.0 * (hh - close) / (hh - ll).replace(0, np.nan)


def CMF(  # noqa: N802
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    ad = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    return (
        ad.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum()
    )


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:  # noqa: N802
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = TRUE_RANGE(pd.DataFrame({"high": high, "low": low, "close": close}))
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).min()


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).max()


def stdev(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).std(ddof=0)


def cross_up(a: pd.Series, b: pd.Series | float | int) -> pd.Series:
    b_series = pd.Series(b, index=a.index) if isinstance(b, (int, float)) else b
    return (a.shift(1) <= b_series.shift(1)) & (a > b_series)


def cross_down(a: pd.Series, b: pd.Series | float | int) -> pd.Series:
    b_series = pd.Series(b, index=a.index) if isinstance(b, (int, float)) else b
    return (a.shift(1) >= b_series.shift(1)) & (a < b_series)
