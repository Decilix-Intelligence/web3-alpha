"""Definitions every other subpackage depends on, and that depend on nothing.

config.py     run parameters and the risk limits, which are the single
              source of truth for what the prompt says and what is enforced
schema.py     the action vocabulary every layer speaks, and its parser
types.py      the records that reach disk
protocols.py  the seams: Agent, Storage, LLMClient
"""

from sentinels.backtest.core.config import (
    FILL_BAR_VWAP,
    FILL_MARK,
    FILL_MID,
    FILL_NEXT_OPEN,
    FILL_POLICIES,
    PROMPT_VARIANTS,
    BacktestConfig,
    RiskControlConfig,
    is_btc_eth,
    tf_seconds,
)
from sentinels.backtest.core.schema import (
    ACTIONS,
    CLOSING_ACTIONS,
    OPENING_ACTIONS,
    Decision,
    ParsedResponse,
    parse_response,
)
from sentinels.backtest.core.protocols import (
    Agent,
    AgentDecision,
    LLMClient,
    Storage,
)
from sentinels.backtest.core.types import (
    HEADLINE_METRICS,
    Checkpoint,
    EquityPoint,
    RunMetadata,
    RunState,
    RunSummary,
)

__all__ = [
    "ACTIONS",
    "Agent",
    "AgentDecision",
    "BacktestConfig",
    "CLOSING_ACTIONS",
    "Checkpoint",
    "Decision",
    "EquityPoint",
    "OPENING_ACTIONS",
    "ParsedResponse",
    "FILL_BAR_VWAP",
    "FILL_MARK",
    "FILL_MID",
    "FILL_NEXT_OPEN",
    "FILL_POLICIES",
    "HEADLINE_METRICS",
    "LLMClient",
    "PROMPT_VARIANTS",
    "RiskControlConfig",
    "RunMetadata",
    "RunState",
    "RunSummary",
    "Storage",
    "is_btc_eth",
    "parse_response",
    "tf_seconds",
]
