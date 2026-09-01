"""Execution price policies.

The default is `next_open`: a decision taken on the bar closing at ts is filled
at the open of the following bar. Filling at the signal bar's own close is the
classic optimistic bias — the decision would be using a price it could not have
traded at — so it is available only as the explicit `mark` policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sentinels.backtest.core.config import FILL_BAR_VWAP, FILL_MARK, FILL_MID, FILL_NEXT_OPEN
from sentinels.backtest.data.feed import Bar


@dataclass
class ExecutionPrice:
    price: float
    source: str
    fell_back: bool = False


def execution_price(
    policy: str,
    signal_bar: Optional[Bar],
    next_bar: Optional[Bar],
    mark_price: float,
) -> ExecutionPrice:
    """Resolve the price an order decided on `signal_bar` actually fills at."""
    if policy == FILL_NEXT_OPEN:
        if next_bar is not None and next_bar.open > 0:
            return ExecutionPrice(next_bar.open, FILL_NEXT_OPEN)
        # last bar of the run: nothing left to fill against
        return ExecutionPrice(mark_price, FILL_NEXT_OPEN, fell_back=True)

    if policy == FILL_BAR_VWAP:
        if signal_bar is not None:
            avg = _ohlc4(signal_bar)
            if avg > 0:
                return ExecutionPrice(avg, FILL_BAR_VWAP)
        return ExecutionPrice(mark_price, FILL_BAR_VWAP, fell_back=True)

    if policy == FILL_MID:
        if signal_bar is not None and signal_bar.high > 0 and signal_bar.low > 0:
            return ExecutionPrice((signal_bar.high + signal_bar.low) / 2.0, FILL_MID)
        return ExecutionPrice(mark_price, FILL_MID, fell_back=True)

    return ExecutionPrice(mark_price, FILL_MARK)


def _ohlc4(bar: Bar) -> float:
    """Typical-price stand-in for VWAP; sample OHLCV carries no per-trade prints."""
    values = [v for v in (bar.open, bar.high, bar.low, bar.close) if v > 0]
    return sum(values) / len(values) if values else 0.0
