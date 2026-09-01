"""Agents are interchangeable, and the baselines are held to the same rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sentinels.backtest.agents import BuyAndHoldAgent, EmaCrossAgent, LLMAgent, build_baseline
from sentinels.backtest.agents.client import DecisionClient
from sentinels.backtest.storage.cache import AICache
from sentinels.backtest.core.config import BacktestConfig, RiskControlConfig
from sentinels.backtest.decisions.context import build_context
from sentinels.backtest.data.feed import DataFeed
from sentinels.backtest.execution.account import Account
from sentinels.backtest.manager import BacktestManager
from sentinels.backtest.storage.store import InMemoryStore
from sentinels.backtest.core.protocols import Agent, Storage
from sentinels.backtest.decisions.validate import PortfolioState, validate_all
from tests.conftest import ScriptedLLM, decision_reply

UTC = timezone.utc
RISK = RiskControlConfig()


def trending_frame(n: int, rising: bool) -> pd.DataFrame:
    """A clean trend, so EMA20 and EMA50 are unambiguously ordered."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol, base in (("BTCUSDT", 40_000.0), ("ETHUSDT", 2_000.0)):
        price = base
        for i in range(n):
            step = base * (0.0009 if rising else -0.0009)
            close = price + step
            rows.append(
                {
                    "timestamp": (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbol": symbol,
                    "open": price,
                    "high": max(price, close) * 1.0008,
                    "low": min(price, close) * 0.9992,
                    "close": close,
                    "volume": 100.0,
                }
            )
            price = close
    return pd.DataFrame(rows)


def context_for(frame: pd.DataFrame, account: Account = None):
    cfg = BacktestConfig(
        run_id="agent-test",
        start_ts=datetime(2026, 1, 3, tzinfo=UTC),
        end_ts=datetime(2026, 1, 3, 6, tzinfo=UTC),
        timeframes=["15m"],
        lookback_bars=200,
    ).validate()
    feed = DataFeed(cfg, frame=frame)
    ts = feed.decision_timestamp(0)
    account = account or Account(1000, 5, 2)
    market = feed.build_market_snapshot(ts)
    return cfg, build_context(cfg, account, market, ts, cycle=1), market


def test_every_agent_satisfies_the_protocol():
    client = DecisionClient(ScriptedLLM([]), AICache(None), variant="basic")
    cfg = BacktestConfig(
        run_id="x",
        start_ts=datetime(2026, 1, 15, tzinfo=UTC),
        end_ts=datetime(2026, 1, 16, tzinfo=UTC),
    ).validate()
    for agent in (LLMAgent(client, cfg), BuyAndHoldAgent(RISK), EmaCrossAgent(RISK)):
        assert isinstance(agent, Agent)
        assert agent.name


def test_in_memory_store_satisfies_the_storage_protocol():
    assert isinstance(InMemoryStore(), Storage)


def test_buy_and_hold_opens_once_then_holds():
    frame = trending_frame(400, rising=True)
    cfg, ctx, _ = context_for(frame)
    agent = BuyAndHoldAgent(RISK)

    first = agent.decide(ctx)
    assert {d.action for d in first.decisions} == {"open_long"}
    assert all(d.leverage == 1 for d in first.decisions), "a levered floor can be liquidated"

    ctx.positions = [{"symbol": d.symbol, "side": "long", "quantity": 1.0} for d in first.decisions]
    second = agent.decide(ctx)
    assert {d.action for d in second.decisions} == {"hold"}


def test_ema_cross_goes_long_in_an_uptrend():
    cfg, ctx, _ = context_for(trending_frame(400, rising=True))
    decisions = EmaCrossAgent(RISK).decide(ctx).decisions
    assert {d.action for d in decisions} == {"open_long"}


def test_ema_cross_stands_aside_in_a_downtrend():
    cfg, ctx, _ = context_for(trending_frame(400, rising=False))
    decisions = EmaCrossAgent(RISK).decide(ctx).decisions
    assert {d.action for d in decisions} == {"wait"}


def test_ema_cross_closes_a_long_when_the_trend_turns():
    cfg, ctx, _ = context_for(trending_frame(400, rising=False))
    ctx.positions = [{"symbol": "BTCUSDT", "side": "long", "quantity": 1.0}]
    decisions = {d.symbol: d for d in EmaCrossAgent(RISK).decide(ctx).decisions}
    assert decisions["BTCUSDT"].action == "close_long"


@pytest.mark.parametrize("rising", [True, False])
@pytest.mark.parametrize("name", ["buy_and_hold", "ema_cross"])
def test_baseline_output_passes_the_same_validator_as_the_model(name, rising):
    """Baselines are not waved past the risk limits the model is held to."""
    cfg, ctx, _ = context_for(trending_frame(400, rising=rising))
    agent = build_baseline(name, RISK)

    state = PortfolioState(equity=ctx.account["equity"], cash=ctx.account["cash"], margin_used=0.0)
    price_map = {s: snap.current_price for s, snap in ctx.market.items()}
    verdicts = validate_all(agent.decide(ctx).decisions, RISK, state, price_map, cfg.symbols)

    rejected = [v for v in verdicts if not v.ok]
    assert not rejected, f"{name} produced invalid orders: {[v.reason for v in rejected]}"


def test_an_unknown_baseline_name_is_refused():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("crystal_ball", RISK)


def test_the_llm_agent_records_its_prompts_for_audit():
    frame = trending_frame(400, rising=True)
    cfg, ctx, _ = context_for(frame)
    cfg.custom_prompt = "HIGH DISAGREEMENT MARKET: reduce position sizes."
    client = DecisionClient(
        ScriptedLLM([decision_reply({"symbol": "BTCUSDT", "action": "wait"})]),
        AICache(None),
        variant="basic",
    )
    outcome = LLMAgent(client, cfg).decide(ctx)

    assert outcome.provenance["agent"] == "llm"
    assert outcome.provenance["custom_prompt_in_system"] is True
    assert "Trading Decision Request" in outcome.provenance["user_prompt"]
    assert outcome.provenance["from_cache"] is False


def test_the_manager_scores_every_agent_on_the_same_bars():
    frame = trending_frame(400, rising=True)
    cfg = BacktestConfig(
        run_id="sweep",
        start_ts=datetime(2026, 1, 3, tzinfo=UTC),
        end_ts=datetime(2026, 1, 3, 12, tzinfo=UTC),
        timeframes=["15m"],
        lookback_bars=200,
    ).validate()
    manager = BacktestManager(cfg, feed=DataFeed(cfg, frame=frame))

    report = manager.compare(
        {"buy_and_hold": BuyAndHoldAgent(RISK), "ema_cross": EmaCrossAgent(RISK)},
        persist=False,
    )

    assert [r.label for r in report.rows] == ["buy_and_hold", "ema_cross"]
    assert all(r.error == "" for r in report.rows)
    assert all("TotalReturnPct" in r.metrics for r in report.rows)
    assert "return%" in report.to_table()
    assert report.best() is not None


def test_manager_binds_external_llm_agent_to_the_llm_child_run(tmp_path):
    """Calls, cache stats and cache files must all belong to ``__llm``."""
    frame = trending_frame(400, rising=True)
    cfg = BacktestConfig(
        run_id="sweep",
        start_ts=datetime(2026, 1, 3, tzinfo=UTC),
        end_ts=datetime(2026, 1, 3, 1, tzinfo=UTC),
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["15m"],
        decision_cadence_nbars=1,
        max_decisions=1,
        lookback_bars=200,
        output_dir=str(tmp_path),
    ).validate()
    manager = BacktestManager(cfg, feed=DataFeed(cfg, frame=frame))
    outer_cache_path = tmp_path / "sweep" / "ai_cache.json"
    outer_client = DecisionClient(
        ScriptedLLM([decision_reply({"symbol": "BTCUSDT", "action": "wait"})]),
        AICache(outer_cache_path),
        variant=cfg.prompt_variant,
    )

    row = manager.compare({"llm": LLMAgent(outer_client, cfg)}).rows[0]

    child_cache_path = tmp_path / "sweep__llm" / "ai_cache.json"
    assert row.error == ""
    assert row.run_id == "sweep__llm"
    assert row.metrics["AICalls"] == 1
    assert row.metrics["CacheStats"] == {"entries": 1, "hits": 0, "misses": 1}
    assert child_cache_path.exists()
    assert not outer_cache_path.exists()
    assert outer_client.calls == 0, "the runner owns per-child call accounting"


def test_one_failing_agent_does_not_sink_the_sweep():
    class Broken:
        name = "broken"

        def decide(self, ctx):
            raise RuntimeError("agent exploded")

    frame = trending_frame(400, rising=True)
    cfg = BacktestConfig(
        run_id="sweep2",
        start_ts=datetime(2026, 1, 3, tzinfo=UTC),
        end_ts=datetime(2026, 1, 3, 6, tzinfo=UTC),
        timeframes=["15m"],
        lookback_bars=200,
    ).validate()
    manager = BacktestManager(cfg, feed=DataFeed(cfg, frame=frame))

    report = manager.compare(
        {"broken": Broken(), "buy_and_hold": BuyAndHoldAgent(RISK)}, persist=False
    )
    assert report.rows[0].error
    assert report.rows[1].error == ""
