"""The simulation loop: protective exits, fills, validation records, replay."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sentinels.backtest.storage.cache import CacheMiss
from sentinels.backtest.core.config import BacktestConfig, FILL_MARK, RiskControlConfig
from sentinels.backtest.data.feed import DataFeed
from sentinels.backtest.storage.store import RunStore
from sentinels.backtest.runner import BacktestRunner
from tests.conftest import ScriptedLLM, decision_reply

UTC = timezone.utc
START = datetime(2026, 1, 15, tzinfo=UTC)


def frame_from(rows) -> pd.DataFrame:
    """rows: list of (open, high, low, close) for BTCUSDT on 15m."""
    out = []
    for i, (o, h, l, c) in enumerate(rows):
        out.append(
            {
                "timestamp": (START + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "symbol": "BTCUSDT",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(out)


def build_runner(tmp_path, rows, llm, **overrides) -> BacktestRunner:
    cfg = BacktestConfig(
        run_id=overrides.pop("run_id", "test-run"),
        symbols=["BTCUSDT"],
        timeframes=["15m"],
        decision_timeframe="15m",
        decision_cadence_nbars=overrides.pop("cadence", 1000),  # only bar 0 decides
        start_ts=START,
        end_ts=START + timedelta(minutes=15 * len(rows)),
        initial_balance=overrides.pop("balance", 1000.0),
        fee_bps=overrides.pop("fee_bps", 0.0),
        slippage_bps=overrides.pop("slippage_bps", 0.0),
        lookback_bars=4,
        output_dir=str(tmp_path),
        **overrides,
    )
    feed = DataFeed(cfg.validate(), frame=frame_from(rows))
    return BacktestRunner(cfg, llm, feed=feed, store=RunStore(tmp_path, cfg.run_id))


LONG_WITH_STOP = decision_reply(
    {
        "symbol": "BTCUSDT",
        "action": "open_long",
        "leverage": 5,
        "position_size_usd": 500,
        "stop_loss": 95,
        "take_profit": 200,
        "confidence": 90,
    }
)


def test_stop_loss_fires_when_the_bar_low_reaches_it(tmp_path):
    rows = [
        (100, 101, 99, 100),  # 0 decision -> open long, filled at bar 1 open
        (100, 101, 99, 100),  # 1 fill bar
        (100, 101, 99, 100),  # 2
        (100, 101, 90, 92),  # 3 low 90 <= stop 95
        (92, 93, 91, 92),  # 4
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]))
    runner.run()

    stops = [f for f in runner._fills if f["action"] == "stop_loss"]
    assert len(stops) == 1, "a bar trading through the stop must close the position"
    assert stops[0]["price"] == pytest.approx(95.0), "the stop level is the fill"
    assert stops[0]["realized_pnl"] < 0
    assert runner.account.position_count() == 0
    assert not runner.liquidated


def test_take_profit_fires_when_the_bar_high_reaches_it(tmp_path):
    reply = decision_reply(
        {
            "symbol": "BTCUSDT",
            "action": "open_long",
            "leverage": 5,
            "position_size_usd": 500,
            "stop_loss": 95,
            "take_profit": 115,
            "confidence": 90,
        }
    )
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 120, 99, 118),  # high 120 >= target 115
        (118, 119, 117, 118),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([reply]))
    runner.run()

    targets = [f for f in runner._fills if f["action"] == "take_profit"]
    assert len(targets) == 1
    assert targets[0]["price"] == pytest.approx(115.0)
    assert targets[0]["realized_pnl"] > 0


def test_price_reaching_the_liquidation_level_wipes_the_position(tmp_path):
    reply = decision_reply(
        {
            "symbol": "BTCUSDT",
            "action": "open_long",
            "leverage": 5,
            "position_size_usd": 500,
            "stop_loss": 75,
            "take_profit": 200,
            "confidence": 90,
        }
    )
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),  # filled at 100 -> liquidation at 80
        (100, 101, 78, 79),  # low 78 <= 80
        (79, 80, 78, 79),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([reply]))
    metrics = runner.run()

    liquidations = [f for f in runner._fills if f["liquidation"]]
    assert len(liquidations) == 1
    assert liquidations[0]["price"] == pytest.approx(80.0)
    assert runner.liquidated
    assert metrics["Liquidated"] is True
    assert runner.account.position_count() == 0


def test_liquidation_takes_precedence_over_a_stop_below_it(tmp_path):
    """A stop parked past the liquidation level can never be the exit."""
    reply = decision_reply(
        {
            "symbol": "BTCUSDT",
            "action": "open_long",
            "leverage": 5,
            "position_size_usd": 500,
            "stop_loss": 70,
            "take_profit": 200,
            "confidence": 90,
        }
    )
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 65, 68),  # blows through liquidation (80) and the stop (70)
        (68, 69, 67, 68),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([reply]))
    runner.run()

    exits = [f for f in runner._fills if f["action"] in ("liquidated", "stop_loss")]
    assert len(exits) == 1
    assert exits[0]["action"] == "liquidated"
    assert exits[0]["price"] == pytest.approx(80.0)


def test_a_short_is_liquidated_on_the_bar_high(tmp_path):
    reply = decision_reply(
        {
            # stop parked above the liquidation level so liquidation is what fires;
            # risk 25 against reward 75 clears the 3:1 floor from a 100 entry
            "symbol": "BTCUSDT",
            "action": "open_short",
            "leverage": 5,
            "position_size_usd": 500,
            "stop_loss": 125,
            "take_profit": 25,
            "confidence": 90,
        }
    )
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),  # short filled at 100 -> liquidation at 120
        (100, 125, 99, 124),  # high 125 >= 120
        (124, 125, 123, 124),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([reply]))
    runner.run()

    liquidations = [f for f in runner._fills if f["liquidation"]]
    assert len(liquidations) == 1
    assert liquidations[0]["price"] == pytest.approx(120.0)


def test_the_order_fills_at_the_next_bar_open_not_the_signal_close(tmp_path):
    rows = [
        (100, 101, 99, 100),  # 0 decision bar, close 100
        (110, 111, 109, 110),  # 1 opens at 110 — a gap the signal could not trade
        (110, 111, 109, 110),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]))
    runner.run()

    opens = [f for f in runner._fills if f["action"] == "open_long"]
    assert opens[0]["price"] == pytest.approx(110.0)
    assert opens[0]["price"] != pytest.approx(100.0)


def test_the_mark_policy_is_the_only_way_to_fill_at_the_signal_close(tmp_path):
    rows = [
        (100, 101, 99, 100),
        (110, 111, 109, 110),
        (110, 111, 109, 110),
    ]
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]), fill_policy=FILL_MARK)
    runner.run()
    opens = [f for f in runner._fills if f["action"] == "open_long"]
    assert opens[0]["price"] == pytest.approx(100.0)


def test_a_rejected_decision_is_recorded_with_its_reason(tmp_path):
    over_leveraged_and_oversized = decision_reply(
        {
            "symbol": "BTCUSDT",
            "action": "open_long",
            "leverage": 5,
            "position_size_usd": 999_999,
            "stop_loss": 95,
            "take_profit": 200,
            "confidence": 90,
        }
    )
    rows = [(100, 101, 99, 100)] * 4
    runner = build_runner(tmp_path, rows, ScriptedLLM([over_leveraged_and_oversized]))
    runner.run()

    record = json.loads((runner.store.decisions_dir / "0001.json").read_text())
    action = record["actions"][0]
    assert action["validation"]["ok"] is False
    assert "exceeds" in action["validation"]["reason"]
    assert action["executed"] is False
    assert runner.account.position_count() == 0, "a rejected decision must not reach the account"


def test_the_decision_record_captures_the_full_exchange(tmp_path):
    rows = [(100, 101, 99, 100)] * 4
    memo = "## News Sentiment Analysis Report\nHIGH DISAGREEMENT MARKET detected."
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]), custom_prompt=memo)
    runner.run()

    record = json.loads((runner.store.decisions_dir / "0001.json").read_text())
    assert record["system_prompt_chars"] > 0
    assert record["custom_prompt_in_system"] is True, "the news memo must reach the system prompt"
    assert record["custom_prompt_chars"] == len(memo)
    assert "Trading Decision Request" in record["user_prompt"]
    assert "<decision>" in record["raw_response"]
    assert record["from_cache"] is False
    assert record["risk_limits"]["min_confidence"] == RiskControlConfig().min_confidence


def test_the_news_memo_reaches_the_system_prompt_and_not_the_user_turn(tmp_path):
    memo = "HIGH DISAGREEMENT MARKET: reduce position sizes by 30-50%."
    llm = ScriptedLLM([LONG_WITH_STOP])
    runner = build_runner(tmp_path, [(100, 101, 99, 100)] * 4, llm, custom_prompt=memo)
    runner.run()

    system, user = llm.calls[0]["system"], llm.calls[0]["user"]
    assert memo in system
    assert memo not in user, "the memo frames the session; it is not resent with every price update"


def test_replay_only_refuses_to_call_the_model_on_a_cache_miss(tmp_path):
    llm = ScriptedLLM([LONG_WITH_STOP])
    runner = build_runner(tmp_path, [(100, 101, 99, 100)] * 4, llm, replay_only=True, cache_ai=True)
    with pytest.raises(CacheMiss):
        runner.run()
    assert llm.calls == [], "replay_only must never reach the provider"


def test_a_resumed_run_preserves_its_cache_without_calling_the_model(tmp_path):
    rows = [(100, 101, 99, 100)] * 4
    first = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]), run_id="cached")
    first_metrics = first.run()

    replay_llm = ScriptedLLM([])
    second = build_runner(tmp_path, rows, replay_llm, run_id="cached", replay_only=True)
    assert second.cache.stats["entries"] == 1, "construction should load the saved cache"
    second_metrics = second.run(resume=True)

    assert replay_llm.calls == []
    assert second_metrics["TotalReturnPct"] == pytest.approx(first_metrics["TotalReturnPct"])
    assert second_metrics["TotalTrades"] == first_metrics["TotalTrades"]


def test_a_fresh_run_with_the_same_id_clears_every_previous_artifact(tmp_path):
    rows = [(100, 101, 99, 100)] * 4
    first = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]), run_id="replace")
    first.run()

    stale_decision = first.store.decisions_dir / "9999.json"
    stale_decision.write_text('{"stale": true}', encoding="utf-8")
    for path in (
        first.store.metrics_path,
        first.store.checkpoint_path,
        first.store.ai_cache_path,
        stale_decision,
    ):
        assert path.exists()

    # This runner has already loaded the old on-disk cache.  A genuinely fresh
    # run must clear both that in-memory copy and all previous run documents.
    replay_llm = ScriptedLLM([])
    fresh = build_runner(tmp_path, rows, replay_llm, run_id="replace", replay_only=True)
    assert fresh.cache.stats["entries"] == 1

    with pytest.raises(CacheMiss):
        fresh.run()

    assert replay_llm.calls == []
    assert not stale_decision.exists()
    assert not fresh.store.metrics_path.exists()
    assert not fresh.store.checkpoint_path.exists()
    assert not fresh.store.ai_cache_path.exists()
    failed = json.loads(fresh.store.run_path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed", "run.json must describe this attempt, not the old one"


def test_a_flaky_provider_is_retried_rather_than_failing_the_run(tmp_path):
    llm = ScriptedLLM([LONG_WITH_STOP], fail_times=2)
    runner = build_runner(tmp_path, [(100, 101, 99, 100)] * 4, llm, ai_retry_base_delay=0.0)
    runner.run()

    assert len(llm.calls) == 3, "two failures then a success"
    assert any(f["action"] == "open_long" for f in runner._fills)


def test_a_provider_that_never_answers_degrades_to_wait(tmp_path):
    llm = ScriptedLLM([], fail_times=99)
    runner = build_runner(tmp_path, [(100, 101, 99, 100)] * 4, llm, ai_retry_base_delay=0.0)
    metrics = runner.run()

    record = json.loads((runner.store.decisions_dir / "0001.json").read_text())
    assert "model call failed" in record["error"]
    assert record["parsed"]["decisions"][0]["action"] == "wait"
    assert metrics["TotalTrades"] == 0, "a dead provider must not invent trades"


def test_the_run_writes_the_documented_output_layout(tmp_path):
    rows = [(100, 101, 99, 100)] * 6
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]))
    metrics = runner.run()

    root = runner.store.root
    for name in ("run.json", "metrics.json", "equity.jsonl", "trades.jsonl", "checkpoint.json"):
        assert (root / name).exists(), f"{name} missing"
    assert list((root / "decisions").glob("*.json"))

    on_disk = json.loads((root / "metrics.json").read_text())
    for field in (
        "TotalReturnPct",
        "MaxDrawdownPct",
        "SharpeRatio",
        "WinRate",
        "ProfitFactor",
        "TotalTrades",
        "Liquidated",
    ):
        assert field in on_disk, f"{field} missing from metrics.json"
    assert on_disk["TotalReturnPct"] == pytest.approx(metrics["TotalReturnPct"])

    equity = [json.loads(line) for line in (root / "equity.jsonl").read_text().splitlines() if line]
    assert len(equity) == runner.feed.decision_bar_count(), "exactly one point per bar"
    assert len({e["ts"] for e in equity}) == len(equity), "timestamps must be unique"


def test_positions_left_open_are_flattened_so_metrics_are_realised(tmp_path):
    rows = [(100, 101, 99, 100)] * 5
    runner = build_runner(tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]))
    metrics = runner.run()

    assert runner.account.position_count() == 0
    assert any(f.get("note") == "end of run" for f in runner._fills)
    assert metrics["TotalTrades"] >= 1


def test_the_reported_return_equals_what_was_actually_realised(tmp_path):
    """Marking to market at the last bar would omit the exit costs of the flatten."""
    rows = [(100, 101, 99, 100)] * 4 + [(104, 105, 103, 104)] * 2
    runner = build_runner(
        tmp_path, rows, ScriptedLLM([LONG_WITH_STOP]), fee_bps=5.0, slippage_bps=2.0
    )
    metrics = runner.run()

    realised = sum(f["realized_pnl"] for f in runner._fills)
    assert runner.account.position_count() == 0
    assert metrics["TotalReturnPct"] == pytest.approx(
        realised / runner.cfg.initial_balance * 100.0, abs=1e-9
    )
    assert metrics["FinalEquity"] == pytest.approx(runner.cfg.initial_balance + realised, abs=1e-9)
