"""Scoring a finished run.

metrics.py  the performance summary written to metrics.json
equity.py   reshaping the equity curve for charts and reports
"""

from sentinels.backtest.analysis.equity import drawdown_series, limit, resample, sort_by_time
from sentinels.backtest.analysis.metrics import (
    MIN_SHARPE_POINTS,
    compute_metrics,
    max_drawdown_pct,
    sharpe_ratio,
)

__all__ = [
    "MIN_SHARPE_POINTS",
    "compute_metrics",
    "drawdown_series",
    "limit",
    "max_drawdown_pct",
    "resample",
    "sharpe_ratio",
    "sort_by_time",
]
