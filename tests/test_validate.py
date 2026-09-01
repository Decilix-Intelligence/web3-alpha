"""Hard risk checks. These run in code, so the prompt cannot be the only guard."""

from __future__ import annotations

import pytest

from sentinels.backtest.core.config import RiskControlConfig
from sentinels.backtest.core.schema import Decision
from sentinels.backtest.decisions.validate import PortfolioState, risk_reward, validate

RISK = RiskControlConfig()
FLAT = PortfolioState(equity=1000.0, cash=1000.0, margin_used=0.0)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def opening(**kwargs) -> Decision:
    base = dict(
        symbol="BTCUSDT",
        action="open_long",
        leverage=5,
        position_size_usd=500.0,
        stop_loss=41_000.0,
        take_profit=45_000.0,
        confidence=80,
    )
    base.update(kwargs)
    return Decision(**base)


def test_excess_leverage_is_capped_and_the_adjustment_is_recorded():
    result = validate(opening(leverage=50), RISK, FLAT, 42_000.0, SYMBOLS)
    assert result.ok
    assert result.decision.leverage == RISK.btc_eth_max_leverage
    assert any("capped" in a for a in result.adjustments)


def test_leverage_beyond_the_altcoin_limit_is_capped_to_the_altcoin_limit():
    risk = RiskControlConfig(altcoin_max_leverage=3, btc_eth_max_leverage=10)
    decision = opening(symbol="SOLUSDT", leverage=10, position_size_usd=50.0)
    result = validate(decision, risk, FLAT, 42_000.0, ["SOLUSDT"])
    assert result.decision.leverage == 3


def test_position_value_above_the_equity_multiple_is_rejected():
    # BTC cap is 5x equity = 5000 USDT
    result = validate(opening(position_size_usd=9_000.0), RISK, FLAT, 42_000.0, SYMBOLS)
    assert not result.ok
    assert "exceeds" in result.reason


def test_position_below_the_symbol_minimum_is_rejected():
    result = validate(opening(position_size_usd=20.0), RISK, FLAT, 42_000.0, SYMBOLS)
    assert not result.ok
    assert "minimum" in result.reason


def test_confidence_below_the_threshold_cannot_open():
    result = validate(opening(confidence=RISK.min_confidence - 1), RISK, FLAT, 42_000.0, SYMBOLS)
    assert not result.ok
    assert "confidence" in result.reason


def test_opening_without_a_stop_or_target_is_rejected():
    assert not validate(opening(stop_loss=None), RISK, FLAT, 42_000.0, SYMBOLS).ok
    assert not validate(opening(take_profit=None), RISK, FLAT, 42_000.0, SYMBOLS).ok


def test_stop_on_the_wrong_side_of_the_target_is_rejected():
    long_bad = validate(opening(stop_loss=46_000.0), RISK, FLAT, 42_000.0, SYMBOLS)
    assert not long_bad.ok and "stop_loss < take_profit" in long_bad.reason

    short_bad = validate(
        opening(action="open_short", stop_loss=41_000.0, take_profit=45_000.0),
        RISK,
        FLAT,
        42_000.0,
        SYMBOLS,
    )
    assert not short_bad.ok and "stop_loss > take_profit" in short_bad.reason


def test_risk_reward_is_measured_against_the_real_entry():
    # entry 42000, stop 41000 (risk 1000), target 45000 (reward 3000) -> 3.0
    assert risk_reward(opening(), 42_000.0) == pytest.approx(3.0)
    # move the entry and the ratio must move with it
    assert risk_reward(opening(), 44_000.0) == pytest.approx(1.0 / 3.0)


def test_a_thin_target_fails_the_risk_reward_floor():
    result = validate(opening(take_profit=43_000.0), RISK, FLAT, 42_000.0, SYMBOLS)
    assert not result.ok
    assert "risk-reward" in result.reason


def test_deriving_the_entry_from_the_stop_and_target_would_never_reject_anything():
    """Why risk_reward() anchors on the entry price rather than the level spread.

    Placing a notional entry a fixed fraction f into the stop-to-target range
    makes reward/risk equal (1-f)/f for *any* stop and target, so the check
    becomes a constant and stops being a check at all.
    """
    for stop, target in ((41_000.0, 45_000.0), (41_900.0, 42_010.0), (10.0, 1_000_000.0)):
        entry = stop + (target - stop) * 0.2
        assert (target - entry) / (entry - stop) == pytest.approx(4.0)


def test_the_position_limit_blocks_a_new_symbol_but_not_an_add():
    risk = RiskControlConfig(max_positions=2)
    full = PortfolioState(
        equity=1000.0,
        cash=1000.0,
        margin_used=0.0,
        open_keys=["BTCUSDT:long", "ETHUSDT:long"],
    )
    blocked = validate(
        opening(symbol="SOLUSDT", position_size_usd=100.0), risk, full, 100.0, ["SOLUSDT"]
    )
    assert not blocked.ok and "limit is 2" in blocked.reason

    adding = validate(opening(), risk, full, 42_000.0, SYMBOLS)
    assert adding.ok, "adding to a position already open must not count as a new slot"


def test_margin_usage_ceiling_is_enforced():
    risk = RiskControlConfig(max_margin_usage=0.5)
    loaded = PortfolioState(equity=1000.0, cash=600.0, margin_used=400.0)
    result = validate(
        opening(position_size_usd=1_000.0, leverage=5), risk, loaded, 42_000.0, SYMBOLS
    )
    assert not result.ok
    assert "margin usage" in result.reason


def test_closing_a_position_that_is_not_open_is_rejected():
    result = validate(
        Decision(symbol="BTCUSDT", action="close_long"), RISK, FLAT, 42_000.0, SYMBOLS
    )
    assert not result.ok
    assert "no open long position" in result.reason


def test_closing_a_position_that_is_open_is_allowed():
    held = PortfolioState(equity=1000.0, cash=800.0, margin_used=200.0, open_keys=["BTCUSDT:long"])
    assert validate(
        Decision(symbol="BTCUSDT", action="close_long"), RISK, held, 42_000.0, SYMBOLS
    ).ok


def test_a_symbol_outside_the_run_universe_is_rejected():
    result = validate(opening(symbol="DOGEUSDT"), RISK, FLAT, 0.12, SYMBOLS)
    assert not result.ok
    assert "not part of this run" in result.reason


def test_hold_and_wait_always_pass():
    for action in ("hold", "wait"):
        assert validate(Decision(symbol="BTCUSDT", action=action), RISK, FLAT, 42_000.0, SYMBOLS).ok
