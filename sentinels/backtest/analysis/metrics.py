"""Performance metrics computed from the equity curve and the fill log.

Sharpe is computed on log returns of the **per-bar** equity curve — one point
per bar of the decision timeframe, whether or not the agent was consulted on
that bar — and annualised by how many such bars fit in a 365-day year, since
crypto trades continuously. Sampling per bar rather than per decision is what
makes the annualisation factor and the sampling interval agree.

It returns 0.0 rather than a number when there are too few points or no
variance. A Sharpe from a handful of bars is noise, and reporting it as though
it were a measurement would be worse than reporting nothing; `SharpeNote` says
which of the two happened.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from sentinels.backtest.core.types import EquityPoint

MIN_SHARPE_POINTS = 10
SECONDS_PER_YEAR = 365.0 * 24 * 3600
PROFIT_FACTOR_CAP = 100.0


def compute_metrics(
    equity_points: Sequence[EquityPoint],
    fills: Sequence[Dict[str, Any]],
    initial_balance: float,
    decision_tf_seconds: int,
    liquidated: bool = False,
) -> Dict[str, Any]:
    initial = initial_balance if initial_balance > 0 else 1.0

    last_equity = initial
    if equity_points and equity_points[-1].equity > 0:
        last_equity = equity_points[-1].equity

    curve = [p.equity for p in equity_points]

    closing = [
        f
        for f in fills
        if f.get("liquidation")
        or str(f.get("action", "")).startswith("close")
        or f.get("realized_pnl")
    ]

    wins = [float(f["realized_pnl"]) for f in closing if float(f.get("realized_pnl", 0)) > 0]
    losses = [-float(f["realized_pnl"]) for f in closing if float(f.get("realized_pnl", 0)) < 0]

    total_win = sum(wins)
    total_loss = sum(losses)

    if total_loss > 0:
        profit_factor = total_win / total_loss
    elif total_win > 0:
        profit_factor = PROFIT_FACTOR_CAP
    else:
        profit_factor = 0.0

    per_symbol: Dict[str, Dict[str, Any]] = {}
    for f in closing:
        symbol = str(f.get("symbol", "?"))
        stats = per_symbol.setdefault(
            symbol,
            {"TotalTrades": 0, "WinningTrades": 0, "LosingTrades": 0, "TotalPnL": 0.0},
        )
        pnl = float(f.get("realized_pnl", 0.0))
        stats["TotalTrades"] += 1
        stats["TotalPnL"] += pnl
        if pnl > 0:
            stats["WinningTrades"] += 1
        elif pnl < 0:
            stats["LosingTrades"] += 1
    for stats in per_symbol.values():
        n = stats["TotalTrades"]
        stats["AvgPnL"] = stats["TotalPnL"] / n if n else 0.0
        stats["WinRate"] = stats["WinningTrades"] / n * 100.0 if n else 0.0

    best = max(per_symbol, key=lambda s: per_symbol[s]["TotalPnL"], default="")
    worst = min(per_symbol, key=lambda s: per_symbol[s]["TotalPnL"], default="")

    sharpe = sharpe_ratio(curve, decision_tf_seconds)

    return {
        "TotalReturnPct": (last_equity - initial) / initial * 100.0,
        "MaxDrawdownPct": max_drawdown_pct(curve),
        "SharpeRatio": sharpe,
        "WinRate": (len(wins) / len(closing) * 100.0) if closing else 0.0,
        "ProfitFactor": profit_factor,
        "TotalTrades": len(closing),
        "Liquidated": bool(liquidated),
        "FinalEquity": last_equity,
        "InitialBalance": initial,
        "TotalPnL": last_equity - initial,
        "AvgWin": (total_win / len(wins)) if wins else 0.0,
        "AvgLoss": -(total_loss / len(losses)) if losses else 0.0,
        "WinningTrades": len(wins),
        "LosingTrades": len(losses),
        "BestSymbol": best,
        "WorstSymbol": worst,
        "SymbolStats": per_symbol,
        "SharpeNote": _sharpe_note(sharpe, len(equity_points)),
    }


def _sharpe_note(sharpe: float, n_points: int) -> str:
    """Why a Sharpe of 0.0 is 0.0, so the number is never read as a measurement."""
    if n_points < MIN_SHARPE_POINTS:
        return f"not reported: {n_points} equity points, need {MIN_SHARPE_POINTS}"
    if sharpe == 0.0:
        return "not reported: per-bar returns have no variance"
    return ""


def max_drawdown_pct(equity: Sequence[float]) -> float:
    """Largest peak-to-trough decline, in percent of the running peak."""
    if not equity:
        return 0.0
    peak = equity[0] if equity[0] > 0 else 1.0
    worst = 0.0
    for value in equity:
        if value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def sharpe_ratio(equity: Sequence[float], period_seconds: int) -> float:
    if len(equity) < MIN_SHARPE_POINTS:
        return 0.0

    returns: List[float] = []
    for prev, curr in zip(equity, equity[1:]):
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))
    if len(returns) < MIN_SHARPE_POINTS - 1:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std < 1e-10:
        return 0.0

    periods_per_year = SECONDS_PER_YEAR / max(period_seconds, 1)
    return (mean / std) * math.sqrt(periods_per_year)
