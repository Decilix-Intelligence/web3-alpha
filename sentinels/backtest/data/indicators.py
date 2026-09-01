"""Technical indicators, computed on a closed-bar prefix only.

Every function takes values ordered oldest -> newest and returns the value at
the LAST element. They only ever read the list handed to them, so a caller that
slices its series at ts cannot leak a future bar in through here; enforcing that
slice is datafeed.py's job.

Standard parameterisations: EMA20/50, MACD(12,26,9), RSI7/14 and ATR14 both with
Wilder smoothing.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def ema_series(values: Sequence[float], period: int) -> List[float]:
    if period <= 0 or not values:
        return []
    k = 2.0 / (period + 1.0)
    out: List[float] = []
    prev: Optional[float] = None
    for i, v in enumerate(values):
        if i + 1 < period:
            out.append(float("nan"))
            continue
        if prev is None:
            seed = sum(values[i + 1 - period : i + 1]) / period
            prev = seed
        else:
            prev = v * k + prev * (1.0 - k)
        out.append(prev)
    return out


def ema(values: Sequence[float], period: int) -> Optional[float]:
    series = ema_series(values, period)
    if not series:
        return None
    last = series[-1]
    return None if last != last else last  # NaN check


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram) at the last bar."""
    if len(values) < slow:
        return None, None, None
    fast_s = ema_series(values, fast)
    slow_s = ema_series(values, slow)
    diff: List[float] = []
    for f, s in zip(fast_s, slow_s):
        if f != f or s != s:
            continue
        diff.append(f - s)
    if not diff:
        return None, None, None
    macd_line = diff[-1]
    if len(diff) < signal:
        return macd_line, None, None
    sig_s = ema_series(diff, signal)
    signal_line = sig_s[-1]
    if signal_line != signal_line:
        return macd_line, None, None
    return macd_line, signal_line, macd_line - signal_line


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Wilder RSI at the last bar."""
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> Optional[float]:
    """Wilder ATR at the last bar."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value
