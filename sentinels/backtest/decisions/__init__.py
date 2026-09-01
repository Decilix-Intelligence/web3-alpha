"""One decision cycle: what the agent is shown, what it may say, what is allowed.

    context.py   the state an agent sees at a decision point
    validate.py  the hard risk checks, enforced in code rather than in the prompt

The action vocabulary itself lives in `core.schema`, because every layer speaks
it and putting it here would make execution and decisions depend on each other.
"""

from sentinels.backtest.decisions.context import DecisionContext, build_context
from sentinels.backtest.core.schema import Decision, ParsedResponse, parse_response
from sentinels.backtest.decisions.validate import (
    PortfolioState,
    ValidationResult,
    risk_reward,
    validate,
    validate_all,
)

__all__ = [
    "Decision",
    "DecisionContext",
    "ParsedResponse",
    "PortfolioState",
    "ValidationResult",
    "build_context",
    "parse_response",
    "risk_reward",
    "validate",
    "validate_all",
]
