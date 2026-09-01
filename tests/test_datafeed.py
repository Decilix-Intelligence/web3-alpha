"""The no-look-ahead guarantee, and higher-timeframe aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.data.feed import DataFeed, resample
from tests.conftest import make_bars

UTC = timezone.utc


def synthetic_frame(n_bars: int, symbols=("BTCUSDT", "ETHUSDT"), start=None) -> pd.DataFrame:
    """A rising sawtooth — the exact shape does not matter, only that it varies."""
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for symbol in symbols:
        price = 100.0 if symbol == "BTCUSDT" else 50.0
        for i in range(n_bars):
            ts = start + timedelta(minutes=15 * i)
            drift = 0.35 if (i // 7) % 2 == 0 else -0.2
            close = price + drift
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbol": symbol,
                    "open": price,
                    "high": max(price, close) + 0.4,
                    "low": min(price, close) - 0.4,
                    "close": close,
                    "volume": 10.0 + i,
                }
            )
            price = close
    return pd.DataFrame(rows)


def feed_for(frame: pd.DataFrame, start: datetime, end: datetime, **kwargs) -> DataFeed:
    cfg = BacktestConfig(
        run_id="test",
        start_ts=start,
        end_ts=end,
        lookback_bars=kwargs.pop("lookback_bars", 20),
        **kwargs,
    ).validate()
    return DataFeed(cfg, frame=frame)


def test_indicators_ignore_bars_that_have_not_closed_yet():
    """The headline invariant: appending future data must not move the past."""
    start = datetime(2026, 1, 2, tzinfo=UTC)
    end = datetime(2026, 1, 2, 6, tzinfo=UTC)

    truncated = synthetic_frame(400)
    extended = synthetic_frame(600)  # same generator, 200 extra bars on the end

    feed_a = feed_for(truncated, start, end)
    feed_b = feed_for(extended, start, end)

    ts = feed_a.decision_timestamp(4)
    snap_a = feed_a.build_market_snapshot(ts)
    snap_b = feed_b.build_market_snapshot(ts)

    for symbol in ("BTCUSDT", "ETHUSDT"):
        for tf in ("15m", "4h"):
            a = snap_a[symbol].by_tf[tf].to_dict()
            b = snap_b[symbol].by_tf[tf].to_dict()
            assert a == b, f"{symbol} {tf} indicators changed when future bars were added"


def test_slice_never_returns_a_bar_that_closes_after_the_cursor():
    start = datetime(2026, 1, 2, tzinfo=UTC)
    feed = feed_for(synthetic_frame(400), start, datetime(2026, 1, 2, 6, tzinfo=UTC))

    for index in range(feed.decision_bar_count()):
        ts = feed.decision_timestamp(index)
        for tf in ("15m", "4h"):
            bars = feed.slice_up_to("BTCUSDT", tf, ts)
            assert bars, f"no {tf} data at {ts}"
            assert bars[-1].close_ts <= ts
            assert all(b.close_ts <= ts for b in bars)


def test_the_bar_closing_exactly_at_the_cursor_is_included():
    start = datetime(2026, 1, 2, tzinfo=UTC)
    feed = feed_for(synthetic_frame(400), start, datetime(2026, 1, 2, 6, tzinfo=UTC))
    ts = feed.decision_timestamp(3)
    assert feed.slice_up_to("BTCUSDT", "15m", ts)[-1].close_ts == ts


def test_decision_bar_returns_the_signal_bar_and_its_successor():
    start = datetime(2026, 1, 2, tzinfo=UTC)
    feed = feed_for(synthetic_frame(400), start, datetime(2026, 1, 2, 6, tzinfo=UTC))
    ts = feed.decision_timestamp(2)

    signal, nxt = feed.decision_bar("BTCUSDT", ts)
    assert signal.close_ts == ts
    assert nxt.open_ts == ts, "the next bar must open exactly when the signal bar closed"
    assert nxt.close_ts > ts


def test_resample_drops_an_incomplete_trailing_bucket():
    bars = make_bars([(100, 101, 99, 100)] * 20)  # 20 x 15m = one 4h bucket plus 4 bars
    four_hour = resample(bars, "4h")
    assert len(four_hour) == 1, "a partial bucket would invent a bar that never closed"
    assert four_hour[0].open_ts == bars[0].open_ts
    assert four_hour[0].close_ts == bars[15].close_ts


def test_resample_aggregates_ohlcv_correctly():
    prices = [(100 + i, 105 + i, 95 + i, 101 + i) for i in range(16)]
    bars = make_bars(prices)
    (candle,) = resample(bars, "4h")

    assert candle.open == pytest.approx(bars[0].open)
    assert candle.close == pytest.approx(bars[-1].close)
    assert candle.high == pytest.approx(max(b.high for b in bars))
    assert candle.low == pytest.approx(min(b.low for b in bars))
    assert candle.volume == pytest.approx(sum(b.volume for b in bars))


def test_a_symbol_missing_decision_bars_is_rejected_up_front():
    frame = synthetic_frame(400)
    gap = (frame["symbol"] == "ETHUSDT") & (frame["timestamp"] > "2026-01-02T02:00:00Z")
    with pytest.raises(ValueError, match="missing"):
        feed_for(frame[~gap], datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 2, 6, tzinfo=UTC))


def test_shipped_sample_data_covers_the_documented_run():
    """The command in the README must work against the committed sample data."""
    cfg = BacktestConfig(
        run_id="sample",
        start_ts=datetime(2026, 1, 15, tzinfo=UTC),
        end_ts=datetime(2026, 1, 16, tzinfo=UTC),
    ).validate()
    feed = DataFeed(cfg)

    assert feed.decision_bar_count() >= 96
    for symbol in ("BTCUSDT", "ETHUSDT"):
        warm = feed.slice_up_to(symbol, "15m", cfg.start_ts)
        assert len(warm) >= 200, f"{symbol} needs 200+ warmup bars, has {len(warm)}"
        snap = feed.build_market_snapshot(feed.decision_timestamp(0))[symbol]
        assert snap.by_tf["15m"].ema50 is not None
        assert snap.by_tf["4h"].ema50 is not None
        assert snap.by_tf["15m"].rsi14 is not None
        assert snap.by_tf["15m"].atr14 is not None


def test_a_thousands_separator_is_repaired_only_in_object_values():
    """Fusing [1,234] into [1234] would silently rewrite the model's output."""
    from sentinels.backtest.core.schema import parse_response

    repaired = parse_response(
        '<decision>[{"symbol": "BTCUSDT", "action": "open_long", "leverage": 5, '
        '"position_size_usd": 1,200, "stop_loss": 41000, "take_profit": 45000, '
        '"confidence": 80}]</decision>'
    )
    assert repaired.decisions[0].position_size_usd == pytest.approx(1200.0)
    assert not repaired.fallback

    untouched = parse_response(
        '<decision>[{"symbol": "BTCUSDT", "action": "wait", '
        '"reasoning": "support at [1,234] intact"}]</decision>'
    )
    assert untouched.decisions[0].reasoning == "support at [1,234] intact"
