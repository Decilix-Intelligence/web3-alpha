"""Market data, and the guarantee that none of it comes from the future.

    feed.py        slices each symbol and timeframe at a cursor, resamples
                   higher timeframes, and computes indicators on the slice
    indicators.py  EMA / MACD / RSI / ATR over a closed-bar prefix
"""
from sentinels.backtest.data.feed import (
    Bar,
    DataFeed,
    SymbolSnapshot,
    TimeframeSnapshot,
    resample,
)
from sentinels.backtest.data.indicators import atr, ema, ema_series, macd, rsi

__all__ = [
    "Bar", "DataFeed", "SymbolSnapshot", "TimeframeSnapshot",
    "atr", "ema", "ema_series", "macd", "resample", "rsi",
]
