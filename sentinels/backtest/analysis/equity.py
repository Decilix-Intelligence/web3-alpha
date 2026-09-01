"""Equity curve utilities.

A run on a 15m decision timeframe produces one point per bar, which is more than
a chart or a report needs. These reshape a curve without pretending to smooth it:
resampling keeps the last point in each bucket, and limiting samples uniformly.
"""

from __future__ import annotations

from typing import List, Sequence

from sentinels.backtest.core.config import tf_seconds
from sentinels.backtest.core.types import EquityPoint


def sort_by_time(points: Sequence[EquityPoint]) -> List[EquityPoint]:
    return sorted(points, key=lambda p: p.ts)


def resample(points: Sequence[EquityPoint], timeframe: str) -> List[EquityPoint]:
    """Keep the last point in each `timeframe` bucket."""
    if not points or not timeframe:
        return list(points)
    bucket_seconds = tf_seconds(timeframe)

    buckets = {}
    for point in sort_by_time(points):
        key = int(point.ts.timestamp()) // bucket_seconds
        buckets[key] = point  # later points overwrite, leaving the bucket's close
    return [buckets[key] for key in sorted(buckets)]


def limit(points: Sequence[EquityPoint], max_points: int) -> List[EquityPoint]:
    """Uniformly sample down to `max_points`, always keeping the last one."""
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    step = (len(points) - 1) / (max_points - 1)
    picked = [points[min(round(i * step), len(points) - 1)] for i in range(max_points)]
    picked[-1] = points[-1]
    return picked


def drawdown_series(points: Sequence[EquityPoint]) -> List[float]:
    """Running drawdown in percent, recomputed from equity rather than trusted."""
    out: List[float] = []
    peak = 0.0
    for point in points:
        peak = max(peak, point.equity)
        out.append(((peak - point.equity) / peak * 100.0) if peak > 0 else 0.0)
    return out
