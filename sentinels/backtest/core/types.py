"""Typed records that move between the loop, the store and the metrics.

Everything the simulator writes to disk passes through one of these, so the
on-disk schema is defined here rather than implied by whatever dictionary the
runner happened to build. `from_dict` on each type is what makes a checkpoint or
an equity stream readable back after a crash or across a resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RunState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    LIQUIDATED = "liquidated"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


@dataclass
class EquityPoint:
    """One row of the equity curve, written once per decision bar."""

    ts: datetime
    bar: int
    cycle: int
    equity: float
    cash: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float
    positions: int
    initial_balance: float = 0.0

    @property
    def pnl(self) -> float:
        return self.equity - self.initial_balance

    @property
    def pnl_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return self.pnl / self.initial_balance * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "bar": self.bar,
            "cycle": self.cycle,
            "equity": round(self.equity, 8),
            "cash": round(self.cash, 8),
            "margin_used": round(self.margin_used, 8),
            "unrealized_pnl": round(self.unrealized_pnl, 8),
            "realized_pnl": round(self.realized_pnl, 8),
            "pnl": round(self.pnl, 8),
            "pnl_pct": round(self.pnl_pct, 8),
            "drawdown_pct": round(self.drawdown_pct, 8),
            "positions": self.positions,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EquityPoint":
        equity = float(raw.get("equity", 0.0))
        pnl = float(raw.get("pnl", 0.0))
        return cls(
            ts=_parse_ts(raw.get("ts")),
            bar=int(raw.get("bar", 0)),
            cycle=int(raw.get("cycle", 0)),
            equity=equity,
            cash=float(raw.get("cash", 0.0)),
            margin_used=float(raw.get("margin_used", 0.0)),
            unrealized_pnl=float(raw.get("unrealized_pnl", 0.0)),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            drawdown_pct=float(raw.get("drawdown_pct", 0.0)),
            positions=int(raw.get("positions", 0)),
            initial_balance=equity - pnl,
        )


@dataclass
class Checkpoint:
    """Everything needed to pick a run back up where it stopped."""

    bar_index: int
    cycle: int
    ai_calls: int
    cash: float
    realized_pnl: float
    max_equity: float
    liquidated: bool = False
    liquidation_note: str = ""
    bar_ts: Optional[datetime] = None
    positions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_index": self.bar_index,
            "bar_ts": self.bar_ts.isoformat() if self.bar_ts else None,
            "cycle": self.cycle,
            "ai_calls": self.ai_calls,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "max_equity": self.max_equity,
            "liquidated": self.liquidated,
            "liquidation_note": self.liquidation_note,
            "positions": self.positions,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Checkpoint":
        return cls(
            bar_index=int(raw.get("bar_index", 0)),
            cycle=int(raw.get("cycle", 0)),
            ai_calls=int(raw.get("ai_calls", 0)),
            cash=float(raw.get("cash", 0.0)),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            max_equity=float(raw.get("max_equity", 0.0)),
            liquidated=bool(raw.get("liquidated", False)),
            liquidation_note=str(raw.get("liquidation_note", "")),
            bar_ts=_parse_ts(raw.get("bar_ts")) if raw.get("bar_ts") else None,
            positions=list(raw.get("positions", [])),
        )


@dataclass
class RunSummary:
    symbols: List[str]
    decision_timeframe: str
    total_bars: int
    processed_bars: int
    decision_cycles: int
    ai_calls: int
    liquidated: bool = False
    liquidation_note: str = ""

    @property
    def progress_pct(self) -> float:
        if self.total_bars <= 0:
            return 0.0
        return min(self.processed_bars / self.total_bars * 100.0, 100.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "decision_timeframe": self.decision_timeframe,
            "total_bars": self.total_bars,
            "processed_bars": self.processed_bars,
            "progress_pct": round(self.progress_pct, 4),
            "decision_cycles": self.decision_cycles,
            "ai_calls": self.ai_calls,
            "liquidated": self.liquidated,
            "liquidation_note": self.liquidation_note,
        }


HEADLINE_METRICS = (
    "TotalReturnPct",
    "MaxDrawdownPct",
    "SharpeRatio",
    "WinRate",
    "ProfitFactor",
    "TotalTrades",
    "Liquidated",
)


@dataclass
class RunMetadata:
    """What run.json holds — enough to understand a run without reading the streams."""

    run_id: str
    state: RunState
    config: Dict[str, Any]
    summary: RunSummary
    created_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        created = self.created_at or datetime.now(timezone.utc)
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "state": str(self.state),
            "created_at": created.isoformat(),
            "config": self.config,
            "summary": self.summary.to_dict(),
        }
        if self.metrics:
            payload["metrics"] = {k: self.metrics[k] for k in HEADLINE_METRICS if k in self.metrics}
        if self.last_error:
            payload["last_error"] = self.last_error
        return payload


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromisoformat(str(value))
