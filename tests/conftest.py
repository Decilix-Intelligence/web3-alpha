"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinels.backtest.data.feed import Bar  # noqa: E402

UTC = timezone.utc


class ScriptedLLM:
    """Stands in for a model: returns queued replies, records what it was asked."""

    DEFAULT = '<reasoning>no setup</reasoning><decision>[{"symbol": "BTCUSDT", "action": "wait"}]</decision>'

    def __init__(self, replies: Optional[List[str]] = None, fail_times: int = 0):
        self.replies = list(replies or [])
        self.fail_times = fail_times
        self.calls: List[dict] = []

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.calls.append({"system": system_prompt or "", "user": prompt})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated provider outage")
        return self.replies.pop(0) if self.replies else self.DEFAULT


@pytest.fixture
def scripted_llm():
    return ScriptedLLM


def make_bars(
    prices,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    start: datetime = datetime(2026, 1, 15, tzinfo=UTC),
    step_seconds: int = 900,
) -> List[Bar]:
    """Build bars from (open, high, low, close) tuples."""
    bars: List[Bar] = []
    for i, row in enumerate(prices):
        o, h, l, c = row
        open_ts = start + timedelta(seconds=step_seconds * i)
        bars.append(
            Bar(
                open_ts=open_ts,
                close_ts=open_ts + timedelta(seconds=step_seconds),
                symbol=symbol,
                timeframe=timeframe,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=100.0,
            )
        )
    return bars


def decision_reply(*decisions) -> str:
    import json

    return (
        "<reasoning>scripted</reasoning>"
        f"<decision>```json\n{json.dumps(list(decisions))}\n```</decision>"
    )
