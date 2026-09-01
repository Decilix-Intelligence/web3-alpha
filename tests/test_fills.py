"""Execution price policies — chiefly that a signal never fills at its own price."""

from __future__ import annotations

import pytest

from sentinels.backtest.core.config import FILL_BAR_VWAP, FILL_MARK, FILL_MID, FILL_NEXT_OPEN
from sentinels.backtest.execution.fills import execution_price
from tests.conftest import make_bars

SIGNAL, NEXT = make_bars([(100.0, 106.0, 94.0, 104.0), (110.0, 115.0, 108.0, 112.0)])


def test_next_open_does_not_fill_on_the_signal_bar():
    resolved = execution_price(FILL_NEXT_OPEN, SIGNAL, NEXT, mark_price=SIGNAL.close)

    assert resolved.price == pytest.approx(NEXT.open)
    assert resolved.price != pytest.approx(
        SIGNAL.close
    ), "filling at the signal bar's close trades at a price the signal was derived from"
    assert not resolved.fell_back


def test_next_open_falls_back_when_the_run_has_no_further_bar():
    resolved = execution_price(FILL_NEXT_OPEN, SIGNAL, None, mark_price=SIGNAL.close)
    assert resolved.price == pytest.approx(SIGNAL.close)
    assert resolved.fell_back, "the fallback must be visible in the decision log"


def test_bar_vwap_uses_the_signal_bar_ohlc_average():
    resolved = execution_price(FILL_BAR_VWAP, SIGNAL, NEXT, mark_price=999.0)
    assert resolved.price == pytest.approx((100.0 + 106.0 + 94.0 + 104.0) / 4)
    assert resolved.source == FILL_BAR_VWAP


def test_mid_uses_the_signal_bar_range_midpoint():
    resolved = execution_price(FILL_MID, SIGNAL, NEXT, mark_price=999.0)
    assert resolved.price == pytest.approx((106.0 + 94.0) / 2)


def test_mark_uses_the_close_and_is_the_only_way_to_get_it():
    resolved = execution_price(FILL_MARK, SIGNAL, NEXT, mark_price=SIGNAL.close)
    assert resolved.price == pytest.approx(SIGNAL.close)
    assert resolved.source == FILL_MARK
