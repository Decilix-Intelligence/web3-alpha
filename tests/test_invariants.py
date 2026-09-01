"""Properties that must hold for every run, checked across a configuration sweep.

These are the statements the rest of the suite's individual cases are examples
of. Running them over a matrix of fill policies, leverages, fee levels and
agents is what turns "the tests pass" into "the accounting closes".
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from sentinels.backtest import (
    BacktestConfig,
    BacktestRunner,
    InMemoryStore,
    RiskControlConfig,
)
from sentinels.backtest.agents import BuyAndHoldAgent, EmaCrossAgent

UTC = timezone.utc
DAY = datetime(2026, 1, 15, tzinfo=UTC)

MATRIX = list(
    itertools.product(
        ["next_open", "bar_vwap", "mid", "mark"],
        [1, 5, 20],
        [BuyAndHoldAgent, EmaCrossAgent],
        [0.0, 5.0],
    )
)
IDS = [f"{p}-lev{l}-{a.name}-fee{f:g}" for p, l, a, f in MATRIX]


# The sweep is read-only, so each configuration is simulated once and the three
# property tests below share the result. Reloading the feed per assertion would
# multiply the suite's runtime for no extra coverage.
_FEED = None
_RESULTS: dict = {}


def _shared_feed():
    global _FEED
    if _FEED is None:
        from sentinels.backtest.data import DataFeed

        _FEED = DataFeed(
            BacktestConfig(
                run_id="feed",
                start_ts=DAY,
                end_ts=DAY + timedelta(days=1),
            ).validate()
        )
    return _FEED


def run_one(policy, leverage, agent_cls, fee_bps):
    key = (policy, leverage, agent_cls.name, fee_bps)
    if key not in _RESULTS:
        risk = RiskControlConfig(btc_eth_max_leverage=leverage, altcoin_max_leverage=leverage)
        cfg = BacktestConfig(
            run_id="invariants",
            start_ts=DAY,
            end_ts=DAY + timedelta(days=1),
            fill_policy=policy,
            fee_bps=fee_bps,
            slippage_bps=2.0,
            decision_cadence_nbars=10,
            risk=risk,
        ).validate()
        runner = BacktestRunner(
            cfg, agent=agent_cls(risk), feed=_shared_feed(), store=InMemoryStore()
        )
        _RESULTS[key] = (runner, runner.run())
    return _RESULTS[key]


@pytest.mark.parametrize("policy,leverage,agent_cls,fee_bps", MATRIX, ids=IDS)
def test_the_accounting_closes(policy, leverage, agent_cls, fee_bps):
    runner, metrics = run_one(policy, leverage, agent_cls, fee_bps)
    account = runner.account
    prices = runner.feed.price_map(runner._current_ts())

    # equity is cash plus posted margin plus mark-to-market, with nothing else
    identity = account.cash + account.margin_used() + account.unrealized(prices)
    assert identity == pytest.approx(metrics["FinalEquity"], abs=1e-6)

    # a simulated account cannot spend money it does not have
    assert account.cash >= -1e-9

    # every fill is a real trade
    for fill in runner._fills:
        assert fill["price"] > 0
        assert fill["fee"] >= -1e-12
        assert fill["qty"] > 0


@pytest.mark.parametrize("policy,leverage,agent_cls,fee_bps", MATRIX, ids=IDS)
def test_the_equity_curve_is_well_formed(policy, leverage, agent_cls, fee_bps):
    runner, metrics = run_one(policy, leverage, agent_cls, fee_bps)
    stamps: List[datetime] = [p.ts for p in runner._equity_points]

    assert len(stamps) == runner.feed.decision_bar_count(), "one point per bar"
    assert len(set(stamps)) == len(stamps), "no repeated timestamps"
    assert stamps == sorted(stamps), "points must be in time order"
    assert 0.0 <= metrics["MaxDrawdownPct"] <= 100.0 + 1e-9


@pytest.mark.parametrize("policy,leverage,agent_cls,fee_bps", MATRIX, ids=IDS)
def test_a_flattened_run_reports_only_realised_money(policy, leverage, agent_cls, fee_bps):
    runner, metrics = run_one(policy, leverage, agent_cls, fee_bps)
    if runner.account.position_count():
        pytest.skip("run ended holding positions")

    realised = sum(f["realized_pnl"] for f in runner._fills)
    assert metrics["TotalReturnPct"] == pytest.approx(
        realised / runner.cfg.initial_balance * 100.0, abs=1e-6
    )
    assert metrics["FinalEquity"] == pytest.approx(runner.cfg.initial_balance + realised, abs=1e-6)


def test_leverage_high_enough_to_be_liquidated_is_liquidated():
    """The liquidation path fires on the shipped data, not only on fixtures."""
    risk = RiskControlConfig(btc_eth_max_leverage=50, altcoin_max_leverage=50)
    cfg = BacktestConfig(
        run_id="liq",
        start_ts=DAY,
        end_ts=DAY + timedelta(days=1),
        decision_cadence_nbars=8,
        risk=risk,
    ).validate()
    runner = BacktestRunner(cfg, agent=EmaCrossAgent(risk), store=InMemoryStore())
    metrics = runner.run()

    assert metrics["Liquidated"] is True
    assert any(f["liquidation"] for f in runner._fills)
    assert runner.account.position_count() == 0


def test_stops_fire_on_the_shipped_data():
    risk = RiskControlConfig()
    cfg = BacktestConfig(
        run_id="stops",
        start_ts=DAY,
        end_ts=DAY + timedelta(days=1),
        decision_cadence_nbars=8,
        risk=risk,
    ).validate()
    runner = BacktestRunner(cfg, agent=EmaCrossAgent(risk), store=InMemoryStore())
    runner.run()

    assert any(f["action"] == "stop_loss" for f in runner._fills)


def test_a_resumed_run_matches_one_that_was_never_interrupted(tmp_path):
    """Checkpoint and replay have to reconstruct the same account, not a similar one."""
    from sentinels.backtest.storage import RunStore

    risk = RiskControlConfig()

    def build(run_id):
        cfg = BacktestConfig(
            run_id=run_id,
            start_ts=DAY,
            end_ts=DAY + timedelta(days=1),
            decision_cadence_nbars=10,
            checkpoint_interval_bars=5,
            risk=risk,
            output_dir=str(tmp_path),
        ).validate()
        return cfg, BacktestRunner(cfg, agent=EmaCrossAgent(risk), store=RunStore(tmp_path, run_id))

    _, straight = build("straight")
    reference = straight.run()

    cfg, partial = build("interrupted")
    partial.store.truncate_streams()
    for i in range(40):
        partial.step(i, final=False)
        partial.bar_index = i + 1
    partial._checkpoint()

    _, resumed = build("interrupted")
    actual = resumed.run(resume=True)

    for field in ("TotalReturnPct", "MaxDrawdownPct", "FinalEquity", "TotalTrades"):
        assert actual[field] == pytest.approx(reference[field], abs=1e-9), field
    assert len(resumed.store.read_equity()) == resumed.feed.decision_bar_count()
