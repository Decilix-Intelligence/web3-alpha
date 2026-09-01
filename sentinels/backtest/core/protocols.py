"""The seams of the simulator.

Three things are pluggable, and these are their contracts. Everything else in
the package is concrete on purpose — indirection you cannot name a second
implementation for is a cost, not a design.

    Agent       anything that turns market state into decisions
    Storage     where a run's records land
    LLMClient   any chat model that takes a system and a user turn

`Agent` is the one that matters for evaluation: the LLM agent and a rule-based
baseline are interchangeable behind it, so the same loop, the same account and
the same metrics can score both. A number is only meaningful next to a baseline
produced by the identical machinery.

Execution price is deliberately *not* a seam. The four policies in
`execution.fills` are branches of one function selected by config, because they
differ by a line of arithmetic rather than by behaviour; wrapping each in a class
would add a layer without adding a capability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from sentinels.backtest.core.types import Checkpoint, EquityPoint, RunMetadata

if TYPE_CHECKING:  # lives in a package that depends on this one
    from sentinels.backtest.core.schema import Decision
    from sentinels.backtest.decisions.context import DecisionContext


@runtime_checkable
class LLMClient(Protocol):
    """A chat completion endpoint. Matches sentinels.llm.base.BaseLLM."""

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str: ...


class AgentDecision:
    """What an agent returns for one cycle: the decisions, plus how it got there.

    `provenance` is free-form and lands verbatim in the cycle's decision record.
    An LLM agent puts its prompts and raw reply there; a rule-based agent puts
    the indicator values it fired on. Both stay auditable.
    """

    __slots__ = ("decisions", "provenance", "error")

    def __init__(
        self,
        decisions: List["Decision"],
        provenance: Optional[Dict[str, Any]] = None,
        error: str = "",
    ):
        self.decisions = decisions
        self.provenance = provenance or {}
        self.error = error


@runtime_checkable
class Agent(Protocol):
    """Turns the state at one decision point into orders to attempt.

    Implementations must not touch the account or the feed directly: whatever
    they need is on the context they are handed, which the feed has already
    sliced to exclude anything that had not closed at that moment.
    """

    name: str

    def decide(self, ctx: "DecisionContext") -> AgentDecision: ...


@runtime_checkable
class Storage(Protocol):
    """Where a run's records land. The filesystem implementation is the default."""

    def append_equity(self, point: EquityPoint) -> None: ...
    def append_trade(self, fill: Dict[str, Any]) -> None: ...
    def read_equity(self) -> List[EquityPoint]: ...
    def read_trades(self) -> List[Dict[str, Any]]: ...
    def truncate_streams(self) -> None: ...
    def reset_run(self) -> None: ...
    def write_decision(self, cycle: int, record: Dict[str, Any]) -> Any: ...
    def write_metrics(self, metrics: Dict[str, Any]) -> None: ...
    def write_run(self, meta: RunMetadata) -> None: ...
    def read_run(self) -> Optional[Dict[str, Any]]: ...
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def load_checkpoint(self) -> Optional[Checkpoint]: ...
