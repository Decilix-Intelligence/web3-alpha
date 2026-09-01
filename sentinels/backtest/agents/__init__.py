"""The strategies. Every one of them satisfies `core.protocols.Agent`.

    llm.py        prompts a language model and parses its decisions
    baselines.py  deterministic rules: buy and hold, EMA cross
    client.py     cache, retry and parsing behind one call, used by the LLM agent

Because they share the protocol, the loop, the account and the metrics are the
same for all of them - which is what makes their numbers comparable.
"""

from sentinels.backtest.agents.baselines import (
    BASELINE_AGENTS,
    RULE_AGENT_CONFIDENCE,
    BuyAndHoldAgent,
    EmaCrossAgent,
    build_baseline,
)
from sentinels.backtest.agents.client import DecisionClient, DecisionResult
from sentinels.backtest.agents.llm import LLMAgent

__all__ = [
    "BASELINE_AGENTS",
    "BuyAndHoldAgent",
    "DecisionClient",
    "DecisionResult",
    "EmaCrossAgent",
    "LLMAgent",
    "RULE_AGENT_CONFIDENCE",
    "build_baseline",
]
