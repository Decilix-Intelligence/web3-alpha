"""Position accounting: averaging, proportional costs, slippage, liquidation."""

from __future__ import annotations

import pytest

from sentinels.backtest.execution.account import (
    Account,
    InsufficientMargin,
    NoSuchPosition,
    apply_slippage,
    liquidation_price,
)


def test_liquidation_price_is_one_over_leverage_away():
    assert liquidation_price(100.0, 5, "long") == pytest.approx(80.0)
    assert liquidation_price(100.0, 5, "short") == pytest.approx(120.0)
    assert liquidation_price(100.0, 10, "long") == pytest.approx(90.0)
    assert liquidation_price(100.0, 2, "short") == pytest.approx(150.0)


def test_slippage_always_moves_against_the_holder():
    rate = 0.01
    assert apply_slippage(100.0, rate, "long", opening=True) == pytest.approx(101.0)
    assert apply_slippage(100.0, rate, "long", opening=False) == pytest.approx(99.0)
    assert apply_slippage(100.0, rate, "short", opening=True) == pytest.approx(99.0)
    assert apply_slippage(100.0, rate, "short", opening=False) == pytest.approx(101.0)


def test_adding_to_a_position_blends_the_entry_by_size():
    acc = Account(10_000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    acc.open("BTCUSDT", "long", 1.0, 5, 200.0)

    pos = acc.get("BTCUSDT", "long")
    assert pos.quantity == pytest.approx(2.0)
    assert pos.entry_price == pytest.approx(150.0)
    assert pos.notional == pytest.approx(300.0)
    assert pos.margin == pytest.approx(60.0)
    # liquidation is recomputed off the blended entry, not the first fill
    assert pos.liquidation_price == pytest.approx(120.0)


def test_uneven_add_weights_the_entry_by_quantity():
    acc = Account(10_000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "long", 3.0, 5, 100.0)
    acc.open("BTCUSDT", "long", 1.0, 5, 200.0)
    assert acc.get("BTCUSDT", "long").entry_price == pytest.approx(125.0)


def test_partial_close_splits_margin_notional_and_entry_fees():
    acc = Account(1000, fee_bps=10, slippage_bps=0)  # 0.10%
    acc.open("BTCUSDT", "long", 2.0, 5, 100.0)

    pos = acc.get("BTCUSDT", "long")
    assert pos.entry_fees == pytest.approx(0.2)
    assert acc.cash == pytest.approx(1000 - 40.0 - 0.2)

    fill = acc.close("BTCUSDT", "long", 1.0, 110.0)

    # half the entry fee follows the half being closed
    assert fill.fee == pytest.approx(0.11 + 0.10)
    assert fill.realized_pnl == pytest.approx(10.0 - 0.21)

    remaining = acc.get("BTCUSDT", "long")
    assert remaining.quantity == pytest.approx(1.0)
    assert remaining.margin == pytest.approx(20.0)
    assert remaining.notional == pytest.approx(100.0)
    assert remaining.entry_fees == pytest.approx(0.1)

    # only the exit fee leaves cash here; the entry fee left when the position opened
    assert acc.cash == pytest.approx(959.8 + 20.0 + 10.0 - 0.11)


def test_entry_fees_are_charged_once_across_a_full_exit_in_two_steps():
    acc = Account(1000, fee_bps=10, slippage_bps=0)
    acc.open("BTCUSDT", "long", 2.0, 5, 100.0)
    first = acc.close("BTCUSDT", "long", 1.0, 100.0)
    second = acc.close("BTCUSDT", "long", 1.0, 100.0)

    entry_fee_charged = (first.fee - 0.1) + (second.fee - 0.1)
    assert entry_fee_charged == pytest.approx(0.2)
    assert acc.get("BTCUSDT", "long") is None


def test_closing_flat_at_the_entry_costs_exactly_the_round_trip_fees():
    acc = Account(1000, fee_bps=10, slippage_bps=0)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    fill = acc.close("BTCUSDT", "long", None, 100.0)
    assert fill.realized_pnl == pytest.approx(-0.2)
    assert acc.cash == pytest.approx(1000 - 0.2)
    assert acc.realized_pnl == pytest.approx(-0.2)


def test_short_pnl_has_the_opposite_sign():
    acc = Account(1000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "short", 1.0, 5, 100.0)
    assert acc.close("BTCUSDT", "short", None, 90.0).realized_pnl == pytest.approx(10.0)


def test_long_and_short_on_one_symbol_are_separate_positions():
    acc = Account(10_000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    acc.open("BTCUSDT", "short", 1.0, 5, 100.0)
    assert acc.position_count() == 2
    assert {p.key for p in acc.positions()} == {"BTCUSDT:long", "BTCUSDT:short"}


def test_equity_is_cash_plus_margin_plus_unrealized():
    acc = Account(1000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    assert acc.equity({"BTCUSDT": 100.0}) == pytest.approx(1000.0)
    assert acc.equity({"BTCUSDT": 110.0}) == pytest.approx(1010.0)
    assert acc.equity({"BTCUSDT": 90.0}) == pytest.approx(990.0)


def test_order_larger_than_cash_is_rejected():
    acc = Account(100, fee_bps=0, slippage_bps=0)
    with pytest.raises(InsufficientMargin):
        acc.open("BTCUSDT", "long", 100.0, 5, 100.0)  # needs 2000 margin
    assert acc.cash == pytest.approx(100.0)
    assert acc.position_count() == 0


def test_closing_a_position_that_is_not_open_is_rejected():
    acc = Account(1000, fee_bps=0, slippage_bps=0)
    with pytest.raises(NoSuchPosition):
        acc.close("BTCUSDT", "long", None, 100.0)


def test_exact_price_close_skips_slippage():
    acc = Account(1000, fee_bps=0, slippage_bps=100)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    fill = acc.close("BTCUSDT", "long", None, 90.0, exact_price=True)
    assert fill.price == pytest.approx(90.0)
    assert fill.slippage == pytest.approx(0.0)


def test_peak_pnl_only_ratchets_up():
    acc = Account(1000, fee_bps=0, slippage_bps=0)
    acc.open("BTCUSDT", "long", 1.0, 5, 100.0)
    acc.mark_to_market({"BTCUSDT": 110.0})
    peak = acc.get("BTCUSDT", "long").peak_pnl_pct
    acc.mark_to_market({"BTCUSDT": 101.0})
    assert acc.get("BTCUSDT", "long").peak_pnl_pct == pytest.approx(peak)


def test_an_order_can_be_sized_to_spend_the_cash_exactly():
    """The affordability limit is derived from margin plus fee, not a guessed buffer."""
    from sentinels.backtest.core.config import BacktestConfig, RiskControlConfig
    from sentinels.backtest.core.schema import Decision
    from sentinels.backtest.execution.broker import Broker
    from datetime import datetime, timezone

    cfg = BacktestConfig(
        run_id="sizing",
        start_ts=datetime(2026, 1, 15, tzinfo=timezone.utc),
        end_ts=datetime(2026, 1, 16, tzinfo=timezone.utc),
        risk=RiskControlConfig(min_position_size_btc_eth=1.0),
    )
    acc = Account(1000, fee_bps=5, slippage_bps=0)
    broker = Broker(cfg, acc, feed=None)

    wants_far_too_much = Decision(
        symbol="BTCUSDT", action="open_long", leverage=5, position_size_usd=1_000_000
    )
    qty, note = broker.size_order(wants_far_too_much, price=100.0)
    assert "trimmed" in note

    # spending the whole trimmed size must leave the account at zero cash,
    # not overdrawn and not holding an arbitrary remainder
    acc.open("BTCUSDT", "long", qty, 5, 100.0)
    assert acc.cash == pytest.approx(0.0, abs=1e-6)


def test_the_derived_size_is_exactly_margin_plus_fee():
    from sentinels.backtest.core.config import BacktestConfig, RiskControlConfig
    from sentinels.backtest.core.schema import Decision
    from sentinels.backtest.execution.broker import Broker
    from datetime import datetime, timezone

    cfg = BacktestConfig(
        run_id="sizing2",
        start_ts=datetime(2026, 1, 15, tzinfo=timezone.utc),
        end_ts=datetime(2026, 1, 16, tzinfo=timezone.utc),
        risk=RiskControlConfig(min_position_size_btc_eth=1.0),
    )
    acc = Account(500, fee_bps=10, slippage_bps=0)
    qty, _ = Broker(cfg, acc, feed=None).size_order(
        Decision(symbol="BTCUSDT", action="open_long", leverage=4, position_size_usd=10**9),
        price=50.0,
    )
    notional = qty * 50.0
    assert notional / 4 + notional * 0.001 == pytest.approx(500.0, abs=1e-6)
