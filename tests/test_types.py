"""Typed records survive the disk round trip, and curve reshaping is honest."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sentinels.backtest.analysis import equity as eq
from sentinels.backtest.storage.store import RunStore
from sentinels.backtest.core.types import (
    Checkpoint,
    EquityPoint,
    RunMetadata,
    RunState,
    RunSummary,
)

UTC = timezone.utc
START = datetime(2026, 1, 15, tzinfo=UTC)


def point(i: int, equity: float) -> EquityPoint:
    return EquityPoint(
        ts=START + timedelta(minutes=15 * i),
        bar=i,
        cycle=i // 20,
        equity=equity,
        cash=equity,
        margin_used=0.0,
        unrealized_pnl=0.0,
        realized_pnl=equity - 1000.0,
        drawdown_pct=0.0,
        positions=0,
        initial_balance=1000.0,
    )


def test_equity_point_survives_a_round_trip():
    original = point(7, 1042.5)
    restored = EquityPoint.from_dict(original.to_dict())

    assert restored.ts == original.ts
    assert restored.bar == original.bar
    assert restored.equity == pytest.approx(original.equity)
    assert restored.initial_balance == pytest.approx(1000.0)
    assert restored.pnl_pct == pytest.approx(4.25)


def test_checkpoint_survives_a_round_trip_with_positions():
    original = Checkpoint(
        bar_index=40,
        cycle=2,
        ai_calls=2,
        cash=899.7,
        realized_pnl=-3.2,
        max_equity=1005.0,
        bar_ts=START,
        liquidated=False,
        positions=[{"symbol": "BTCUSDT", "side": "long", "quantity": 0.02}],
    )
    restored = Checkpoint.from_dict(original.to_dict())

    assert restored.bar_index == 40
    assert restored.ai_calls == 2
    assert restored.bar_ts == START
    assert restored.positions[0]["symbol"] == "BTCUSDT"


def test_run_metadata_carries_only_the_headline_metrics():
    meta = RunMetadata(
        run_id="r1",
        state=RunState.COMPLETED,
        config={"symbols": ["BTCUSDT"]},
        summary=RunSummary(["BTCUSDT"], "15m", 97, 97, 5, 5),
        metrics={"TotalReturnPct": 1.5, "SharpeRatio": 0.2, "SymbolStats": {"noise": 1}},
    )
    payload = meta.to_dict()

    assert payload["state"] == "completed"
    assert payload["metrics"]["TotalReturnPct"] == 1.5
    assert "SymbolStats" not in payload["metrics"], "run.json stays a summary"
    assert payload["summary"]["progress_pct"] == pytest.approx(100.0)


def test_the_store_reads_back_what_it_wrote(tmp_path):
    store = RunStore(tmp_path, "roundtrip")
    written = [point(i, 1000.0 + i) for i in range(5)]
    for p in written:
        store.append_equity(p)

    read = store.read_equity()
    assert len(read) == 5
    assert [p.equity for p in read] == pytest.approx([p.equity for p in written])
    assert read[0].ts == written[0].ts


def test_a_torn_final_line_does_not_break_the_reader(tmp_path):
    store = RunStore(tmp_path, "torn")
    store.append_equity(point(0, 1000.0))
    with store.equity_path.open("a") as fh:
        fh.write('{"ts": "2026-01-15T00:15:00+00:00", "equ')  # killed mid-write

    assert len(store.read_equity()) == 1


def test_reset_run_removes_only_the_named_run_artifacts(tmp_path):
    store = RunStore(tmp_path, "replace-me")
    other = RunStore(tmp_path, "keep-me")

    generated = (
        store.run_path,
        store.equity_path,
        store.trades_path,
        store.metrics_path,
        store.checkpoint_path,
        store.ai_cache_path,
    )
    for path in generated:
        path.write_text("{}", encoding="utf-8")
    for path in (
        store.run_path,
        store.metrics_path,
        store.checkpoint_path,
        store.ai_cache_path,
    ):
        path.with_suffix(path.suffix + ".tmp").write_text("torn", encoding="utf-8")

    numeric_decisions = (
        store.decisions_dir / "0001.json",
        store.decisions_dir / "12345.json",
        store.decisions_dir / "0002.json.tmp",
    )
    for path in numeric_decisions:
        path.write_text("{}", encoding="utf-8")

    operator_note = store.root / "notes.txt"
    named_decision_note = store.decisions_dir / "review.json"
    operator_note.write_text("keep", encoding="utf-8")
    named_decision_note.write_text("keep", encoding="utf-8")
    other.metrics_path.write_text('{"keep": true}', encoding="utf-8")

    store.reset_run()

    assert not any(path.exists() for path in generated)
    assert not any(path.exists() for path in numeric_decisions)
    assert operator_note.read_text(encoding="utf-8") == "keep"
    assert named_decision_note.read_text(encoding="utf-8") == "keep"
    assert other.metrics_path.exists(), "resetting one run id must not touch its sibling"


def test_resampling_keeps_each_bucket_closing_value():
    points = [point(i, 1000.0 + i) for i in range(32)]  # 32 x 15m = two 4h buckets
    resampled = eq.resample(points, "4h")

    assert len(resampled) == 2
    assert resampled[0].equity == pytest.approx(1015.0)  # bar 15 closes bucket one
    assert resampled[1].equity == pytest.approx(1031.0)


def test_limiting_a_curve_keeps_the_endpoints():
    points = [point(i, 1000.0 + i) for i in range(96)]
    limited = eq.limit(points, 10)

    assert len(limited) == 10
    assert limited[0].equity == pytest.approx(points[0].equity)
    assert limited[-1].equity == pytest.approx(points[-1].equity), "the final value must survive"


def test_limiting_is_a_no_op_when_the_curve_is_already_short():
    points = [point(i, 1000.0 + i) for i in range(4)]
    assert len(eq.limit(points, 10)) == 4


def test_drawdown_series_is_recomputed_not_trusted():
    points = [point(0, 1000.0), point(1, 1200.0), point(2, 900.0), point(3, 1100.0)]
    for p in points:
        p.drawdown_pct = 0.0  # deliberately wrong on the record

    series = eq.drawdown_series(points)
    assert series[1] == pytest.approx(0.0)
    assert series[2] == pytest.approx(25.0)


def test_the_sharpe_note_explains_every_zero():
    """A Sharpe of 0.0 must never be readable as a measured 0.0."""
    from sentinels.backtest.analysis.metrics import MIN_SHARPE_POINTS, compute_metrics

    too_few = compute_metrics([point(i, 1000.0 + i) for i in range(4)], [], 1000.0, 900)
    assert too_few["SharpeRatio"] == 0.0
    assert "need" in too_few["SharpeNote"]

    flat = compute_metrics(
        [point(i, 1000.0) for i in range(MIN_SHARPE_POINTS + 5)], [], 1000.0, 900
    )
    assert flat["SharpeRatio"] == 0.0
    assert "no variance" in flat["SharpeNote"]

    varied = compute_metrics([point(i, 1000.0 + (i % 3) * 7) for i in range(40)], [], 1000.0, 900)
    assert varied["SharpeRatio"] != 0.0
    assert varied["SharpeNote"] == ""
