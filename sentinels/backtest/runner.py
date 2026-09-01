"""The simulation loop.

The loop itself is deliberately small. It decides *when* things happen; the
components it drives decide *what* happens:

    DataFeed        what the market looked like at ts, with no look-ahead
    Broker          protective exits, order sizing, fills against the account
    DecisionClient  cache, retry and parsing behind one call to the model
    validate        which decisions are allowed to reach the broker
    RunStore        where the typed records land

One iteration covers one bar of the decision timeframe:

  1. Resolve protective exits against the bar's own high/low - liquidation
     first, then stop-loss, then take-profit. This runs on every bar, not only
     on bars where the model is consulted, and before the model sees the account
     so the state in the prompt is the state after those exits.
  2. Mark the account to the bar's close.
  3. On a decision bar, build the context, ask the model, validate every
     decision, and submit the survivors closes-first.
  4. Orders fill at the next bar's open under the default policy, so no order is
     ever filled at a price its own signal was derived from.

A position opened at step 4 is first exposed to protective exits on the
following bar, which is exactly the bar its fill price came from.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sentinels.backtest.execution.account import Account
from sentinels.backtest.agents import LLMAgent
from sentinels.backtest.agents.client import DecisionClient
from sentinels.backtest.storage.cache import AICache, CacheMiss
from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.decisions.context import DecisionContext, build_context
from sentinels.backtest.data.feed import DataFeed, SymbolSnapshot
from sentinels.backtest.core.schema import Decision
from sentinels.backtest.execution import Broker
from sentinels.backtest.analysis.metrics import compute_metrics
from sentinels.backtest.storage.store import RunStore
from sentinels.backtest.core.protocols import Agent, AgentDecision, Storage
from sentinels.backtest.core.types import Checkpoint, EquityPoint, RunMetadata, RunState, RunSummary
from sentinels.backtest.decisions.validate import PortfolioState, validate_all

logger = logging.getLogger(__name__)

# Closes go first so margin they release is available to the same cycle's opens.
ACTION_PRIORITY = {
    "close_long": 0,
    "close_short": 0,
    "open_long": 1,
    "open_short": 1,
    "hold": 2,
    "wait": 2,
}


class BacktestRunner:
    def __init__(
        self,
        cfg: BacktestConfig,
        llm: Any = None,
        feed: Optional[DataFeed] = None,
        store: Optional[Storage] = None,
        agent: Optional[Agent] = None,
    ):
        """`agent` runs whatever strategy you hand it. Passing an `llm` instead
        is shorthand for the default LLM agent, which is what the CLI does."""
        self.cfg = cfg.validate()
        self.feed = feed or DataFeed(self.cfg)
        self.store = store or RunStore(self.cfg.output_dir, self.cfg.run_id)
        self.account = Account(cfg.initial_balance, cfg.fee_bps, cfg.slippage_bps)
        self.broker = Broker(self.cfg, self.account, self.feed)
        self.cache = AICache(getattr(self.store, "ai_cache_path", None) if cfg.cache_ai else None)
        self.client: Optional[DecisionClient] = None

        if agent is None and llm is not None and hasattr(llm, "decide"):
            agent = llm  # an Agent was passed positionally
            llm = None

        if agent is not None:
            self.agent = agent
        else:
            self.client = DecisionClient(
                llm,
                self.cache,
                variant=self.cfg.prompt_variant,
                max_retries=self.cfg.ai_max_retries,
                retry_base_delay=self.cfg.ai_retry_base_delay,
                cache_ai=self.cfg.cache_ai,
                replay_only=self.cfg.replay_only,
            )
            self.agent = LLMAgent(self.client, self.cfg)

        self.bar_index = 0
        self.cycle = 0
        self.liquidated = False
        self.liquidation_note = ""
        self.max_equity = cfg.initial_balance
        self._equity_points: List[EquityPoint] = []
        self._fills: List[Dict[str, Any]] = []

    @property
    def ai_calls(self) -> int:
        return self.client.calls if self.client else 0

    # ------------------------------------------------------------------ run

    def run(self, resume: bool = False) -> Dict[str, Any]:
        if resume:
            if not self._restore():
                logger.info(
                    "no usable checkpoint for %s, starting from the beginning", self.cfg.run_id
                )
        else:
            self._prepare_fresh_run()
        if self.bar_index == 0:
            self.store.truncate_streams()

        total = self.feed.decision_bar_count()
        logger.info(
            "run %s: %d bars of %s over %s, symbols %s, fill policy %s",
            self.cfg.run_id,
            total,
            self.cfg.decision_timeframe,
            ", ".join(self.cfg.timeframes),
            "/".join(self.cfg.symbols),
            self.cfg.fill_policy,
        )
        self._write_run(RunState.RUNNING)

        while self.bar_index < total:
            try:
                self.step(self.bar_index, final=self.bar_index == total - 1)
            except CacheMiss as exc:
                self._write_run(RunState.FAILED, last_error=str(exc))
                raise
            self.bar_index += 1
            if self.bar_index % self.cfg.checkpoint_interval_bars == 0:
                self._checkpoint()
            if self.liquidated:
                logger.warning("account liquidated: %s", self.liquidation_note)
                break

        # Only reachable when a resume starts at or past the final bar, so the
        # loop body never ran and nothing was flattened inside it.
        if self.cfg.flatten_at_end and self.account.positions():
            self._record(self.broker.flatten(self._current_ts()))

        self._checkpoint()
        return self._finalise()

    def _prepare_fresh_run(self) -> None:
        """Start independently of every artifact left by the same run id.

        The cache is loaded while the runner is constructed, before ``run``
        knows whether this is a resume.  Clearing only the file here would
        therefore leave the old entries live in memory; reset that same object
        in place so the already-wired DecisionClient and LLMAgent keep pointing
        at the clean cache.
        """
        self.store.reset_run()
        self.cache.entries.clear()
        self.cache.hits = 0
        self.cache.misses = 0
        if self.client is not None:
            self.client.calls = 0

        # A runner is normally single-use, but ``run(resume=False)`` should be
        # fresh even when the caller deliberately reuses the object.
        self.account = Account(self.cfg.initial_balance, self.cfg.fee_bps, self.cfg.slippage_bps)
        self.broker = Broker(self.cfg, self.account, self.feed)
        self.bar_index = 0
        self.cycle = 0
        self.liquidated = False
        self.liquidation_note = ""
        self.max_equity = self.cfg.initial_balance
        self._equity_points.clear()
        self._fills.clear()

    # ----------------------------------------------------------------- step

    def step(self, index: int, final: bool = False) -> None:
        """One bar. Exits resolve first, then the agent, then the equity point.

        The equity point is written last and exactly once per bar, so the curve
        has one row per bar with no repeated timestamps and its final value is
        the account after everything that happened on that bar — including the
        closing costs of a flatten.
        """
        ts = self.feed.decision_timestamp(index)
        if ts is None:
            return

        market = self.feed.build_market_snapshot(ts)
        price_map = {symbol: snap.current_price for symbol, snap in market.items()}

        # 1. involuntary exits, against this bar's own range
        exits, liquidation_note = self.broker.protective_exits(ts)
        self._record(exits)
        if liquidation_note:
            self.liquidated = True
            self.liquidation_note = liquidation_note

        # 2. the agent, unless the account has just been wiped out
        record = None
        if not self.liquidated and self._should_decide(index):
            self.cycle += 1
            record = self._run_decision_cycle(ts, index, market, price_map)

        # 3. on the last bar of the run, realise whatever is still open
        if (final or self.liquidated) and self.cfg.flatten_at_end:
            self._record(self.broker.flatten(ts))

        self._append_equity(ts, index, price_map)

        if record is not None:
            record["equity_after"] = round(self._equity_points[-1].equity, 8)
            self.store.write_decision(self.cycle, record)

    def _should_decide(self, index: int) -> bool:
        if self.cfg.max_decisions is not None and self.cycle >= self.cfg.max_decisions:
            return False
        if self.cfg.decision_cadence_nbars <= 1:
            return True
        return index % self.cfg.decision_cadence_nbars == 0

    # ------------------------------------------------------------- decisions

    def _run_decision_cycle(
        self,
        ts: datetime,
        index: int,
        market: Dict[str, SymbolSnapshot],
        price_map: Dict[str, float],
    ) -> Dict[str, Any]:
        ctx = build_context(self.cfg, self.account, market, ts, self.cycle)
        record = self._new_record(ts, index, ctx)

        try:
            outcome = self.agent.decide(ctx)
        except CacheMiss as exc:
            record["from_cache"] = False
            record["error"] = str(exc)
            self.store.write_decision(self.cycle, record)
            raise

        record.update(outcome.provenance)
        if outcome.error:
            record["error"] = outcome.error
            logger.warning("cycle %d: %s", self.cycle, outcome.error)

        record["actions"] = self._apply(outcome, ts, ctx, price_map)
        record["cache_stats"] = self.cache.stats
        return record

    def _apply(
        self,
        outcome: AgentDecision,
        ts: datetime,
        ctx: DecisionContext,
        price_map: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Validate, then submit what survives — closes before opens."""
        state = PortfolioState(
            equity=ctx.account["equity"],
            cash=self.account.cash,
            margin_used=ctx.account["margin_used"],
            open_keys=[p.key for p in self.account.positions()],
        )
        verdicts = validate_all(
            outcome.decisions, self.cfg.risk, state, price_map, self.cfg.symbols
        )

        actions: List[Dict[str, Any]] = []
        approved: List[Tuple[Decision, Dict[str, Any]]] = []
        for verdict in verdicts:
            entry = {
                "requested": verdict.decision.to_dict(),
                "validation": verdict.to_dict(),
                "executed": False,
            }
            actions.append(entry)
            if verdict.ok:
                approved.append((verdict.decision, entry))
            else:
                logger.info(
                    "cycle %d rejected %s %s: %s",
                    self.cycle,
                    verdict.decision.symbol,
                    verdict.decision.action,
                    verdict.reason,
                )

        for decision, entry in sorted(
            approved, key=lambda pair: ACTION_PRIORITY.get(pair[0].action, 9)
        ):
            fill, note = self.broker.submit(decision, ts, price_map)
            entry["executed"] = fill is not None
            entry["note"] = note
            if fill is not None:
                entry["fill"] = fill.to_dict()
                self._record([fill])

        return actions

    def _new_record(self, ts: datetime, index: int, ctx: DecisionContext) -> Dict[str, Any]:
        """The part of a cycle record the loop knows. The agent fills in the rest."""
        return {
            "cycle": self.cycle,
            "bar": index,
            "ts": ts.isoformat(),
            "agent": getattr(self.agent, "name", "unknown"),
            "account": ctx.account,
            "positions": ctx.positions,
            "candidates": ctx.candidates,
            "risk_limits": self.cfg.risk.to_dict(),
        }

    # ------------------------------------------------------------ bookkeeping

    def _record(self, fills) -> None:
        for fill in fills:
            payload = fill.to_dict()
            self._fills.append(payload)
            self.store.append_trade(payload)

    def _append_equity(self, ts: datetime, index: int, price_map: Dict[str, float]) -> None:
        equity, unrealized, margin_used = self.account.mark_to_market(price_map)
        self.max_equity = max(self.max_equity, equity)
        drawdown = (
            (self.max_equity - equity) / self.max_equity * 100.0 if self.max_equity > 0 else 0.0
        )
        point = EquityPoint(
            ts=ts,
            bar=index,
            cycle=self.cycle,
            equity=equity,
            cash=self.account.cash,
            margin_used=margin_used,
            unrealized_pnl=unrealized,
            realized_pnl=self.account.realized_pnl,
            drawdown_pct=drawdown,
            positions=self.account.position_count(),
            initial_balance=self.account.initial_balance,
        )
        self._equity_points.append(point)
        self.store.append_equity(point)

    def _current_ts(self) -> datetime:
        index = min(self.bar_index, self.feed.decision_bar_count() - 1)
        return self.feed.decision_timestamp(index)

    def _checkpoint(self) -> None:
        self.store.save_checkpoint(
            Checkpoint(
                bar_index=self.bar_index,
                bar_ts=self._current_ts(),
                cycle=self.cycle,
                ai_calls=self.ai_calls,
                cash=self.account.cash,
                realized_pnl=self.account.realized_pnl,
                max_equity=self.max_equity,
                liquidated=self.liquidated,
                liquidation_note=self.liquidation_note,
                positions=self.account.snapshot(),
            )
        )

    def _restore(self) -> bool:
        ckpt = self.store.load_checkpoint()
        if ckpt is None:
            return False
        self.bar_index = ckpt.bar_index
        self.cycle = ckpt.cycle
        if self.client is not None:
            self.client.calls = ckpt.ai_calls
        self.max_equity = ckpt.max_equity or self.cfg.initial_balance
        self.liquidated = ckpt.liquidated
        self.liquidation_note = ckpt.liquidation_note
        self.account.restore(ckpt.cash, ckpt.realized_pnl, ckpt.positions)
        self._equity_points = self.store.read_equity()
        self._fills = self.store.read_trades()
        logger.info("resumed %s at bar %d, cycle %d", self.cfg.run_id, self.bar_index, self.cycle)
        return True

    def _finalise(self) -> Dict[str, Any]:
        metrics = compute_metrics(
            self._equity_points,
            self._fills,
            self.cfg.initial_balance,
            self.cfg.decision_tf_seconds,
            liquidated=self.liquidated,
        )
        metrics["AICalls"] = self.ai_calls
        metrics["DecisionCycles"] = self.cycle
        metrics["CacheStats"] = self.cache.stats
        self.store.write_metrics(metrics)
        self._write_run(
            RunState.LIQUIDATED if self.liquidated else RunState.COMPLETED, metrics=metrics
        )
        return metrics

    def _summary(self) -> RunSummary:
        return RunSummary(
            symbols=list(self.cfg.symbols),
            decision_timeframe=self.cfg.decision_timeframe,
            total_bars=self.feed.decision_bar_count(),
            processed_bars=self.bar_index,
            decision_cycles=self.cycle,
            ai_calls=self.ai_calls,
            liquidated=self.liquidated,
            liquidation_note=self.liquidation_note,
        )

    def _write_run(
        self,
        state: RunState,
        metrics: Optional[Dict[str, Any]] = None,
        last_error: str = "",
    ) -> None:
        existing = self.store.read_run() or {}
        created = existing.get("created_at")
        self.store.write_run(
            RunMetadata(
                run_id=self.cfg.run_id,
                state=state,
                config=self.cfg.to_dict(),
                summary=self._summary(),
                created_at=(
                    datetime.fromisoformat(created) if created else datetime.now(timezone.utc)
                ),
                metrics=metrics,
                last_error=last_error,
            )
        )
