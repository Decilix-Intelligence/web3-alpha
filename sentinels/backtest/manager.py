"""Running several agents over the same bars and putting the results side by side.

A single return number is not a result. This runs each agent through the same
feed, the same account rules and the same metrics, so the only thing that differs
between rows is the agent itself.

    manager = BacktestManager(cfg)
    report = manager.compare({
        "llm": LLMAgent(client, cfg),
        "buy_and_hold": BuyAndHoldAgent(cfg.risk),
        "ema_cross": EmaCrossAgent(cfg.risk),
    })
    print(report.to_table())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from sentinels.backtest.agents import LLMAgent
from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.data.feed import DataFeed
from sentinels.backtest.storage.store import InMemoryStore, RunStore
from sentinels.backtest.core.protocols import Agent, Storage
from sentinels.backtest.runner import BacktestRunner
from sentinels.backtest.core.types import HEADLINE_METRICS

logger = logging.getLogger(__name__)


@dataclass
class ComparisonRow:
    label: str
    run_id: str
    agent: str
    metrics: Dict[str, Any]
    error: str = ""

    def headline(self) -> Dict[str, Any]:
        return {k: self.metrics.get(k) for k in HEADLINE_METRICS}


@dataclass
class ComparisonReport:
    rows: List[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {row.label: row.headline() for row in self.rows}

    def best(self, metric: str = "TotalReturnPct") -> Optional[ComparisonRow]:
        scored = [r for r in self.rows if isinstance(r.metrics.get(metric), (int, float))]
        return max(scored, key=lambda r: r.metrics[metric]) if scored else None

    def to_table(self) -> str:
        header = f"{'agent':<16}{'return%':>10}{'maxDD%':>10}{'sharpe':>10}{'win%':>8}{'PF':>8}{'trades':>8}{'liq':>6}"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            if row.error:
                lines.append(f"{row.label:<16}{'failed: ' + row.error[:50]:>54}")
                continue
            m = row.metrics
            lines.append(
                f"{row.label:<16}"
                f"{m.get('TotalReturnPct', 0):>+10.2f}"
                f"{m.get('MaxDrawdownPct', 0):>10.2f}"
                f"{m.get('SharpeRatio', 0):>10.2f}"
                f"{m.get('WinRate', 0):>8.1f}"
                f"{m.get('ProfitFactor', 0):>8.2f}"
                f"{m.get('TotalTrades', 0):>8d}"
                f"{('yes' if m.get('Liquidated') else 'no'):>6}"
            )
        return "\n".join(lines)


class BacktestManager:
    """Runs agents over one shared feed and collects their metrics.

    The feed is loaded once and reused, so every agent is scored on bit-identical
    data — reloading it per run would leave room for the inputs to differ.
    """

    def __init__(self, cfg: BacktestConfig, feed: Optional[DataFeed] = None):
        self.cfg = cfg.validate()
        self.feed = feed or DataFeed(self.cfg)

    def run(
        self,
        agent: Agent,
        label: Optional[str] = None,
        store: Optional[Storage] = None,
        persist: bool = True,
        **overrides: Any,
    ) -> ComparisonRow:
        label = label or getattr(agent, "name", "agent")
        run_id = overrides.pop("run_id", f"{self.cfg.run_id}__{label}")
        # each run gets its own risk object, so an override in one sweep entry
        # cannot leak into the others through a shared reference
        cfg = replace(self.cfg, run_id=run_id, risk=replace(self.cfg.risk), **overrides).validate()

        if store is None:
            store = RunStore(cfg.output_dir, run_id) if persist else InMemoryStore(run_id)

        logger.info("running %s as %s", label, run_id)
        if type(agent) is LLMAgent:
            # A comparison run has a child run id (for example ``sweep__llm``),
            # so it must also own the DecisionClient and cache for that child.
            # Reusing the caller's pre-built agent would send model calls and
            # cache hits through its parent-run client while the runner reports
            # statistics from a different, unused client/cache pair.
            runner = BacktestRunner(cfg, llm=agent.client.llm, feed=self.feed, store=store)
        else:
            runner = BacktestRunner(cfg, feed=self.feed, store=store, agent=agent)
        try:
            metrics = runner.run()
        except Exception as exc:  # one agent failing must not sink the sweep
            logger.warning("%s failed: %s", label, exc)
            return ComparisonRow(label, run_id, getattr(agent, "name", "?"), {}, str(exc))

        return ComparisonRow(label, run_id, getattr(agent, "name", "?"), metrics)

    def compare(
        self, agents: Dict[str, Agent], persist: bool = True, **overrides: Any
    ) -> ComparisonReport:
        return ComparisonReport(
            rows=[
                self.run(agent, label=label, persist=persist, **overrides)
                for label, agent in agents.items()
            ]
        )
