"""The language-model agent."""

from __future__ import annotations

import logging

from sentinels.backtest.agents.client import DecisionClient
from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.core.protocols import AgentDecision
from sentinels.backtest.decisions.context import DecisionContext
from sentinels.engine.system_prompt import build_system_prompt

logger = logging.getLogger(__name__)


class LLMAgent:
    """Prompts a language model and parses what comes back.

    The strategy memo and the risk limits go in the system prompt; the account
    and the market go in the user turn. Everything needed to audit the cycle -
    both prompts, the raw reply, whether it came from cache - is returned as
    provenance and written to the decision record.
    """

    name = "llm"

    def __init__(self, client: DecisionClient, cfg: BacktestConfig):
        self.client = client
        self.cfg = cfg

    def decide(self, ctx: DecisionContext) -> AgentDecision:
        memo = (self.cfg.custom_prompt or "").strip()
        system_prompt = build_system_prompt(
            account_equity=ctx.account["equity"],
            variant=self.cfg.prompt_variant,
            custom_prompt=self.cfg.custom_prompt,
            risk=self.cfg.risk,
        )
        user_prompt = ctx.to_user_prompt()

        result = self.client.decide(ctx.cache_payload(), system_prompt, user_prompt, ctx.timestamp)

        return AgentDecision(
            decisions=result.parsed.decisions,
            error=result.error,
            provenance={
                "agent": self.name,
                "prompt_variant": self.cfg.prompt_variant,
                "system_prompt_chars": len(system_prompt),
                "custom_prompt_chars": len(self.cfg.custom_prompt or ""),
                "custom_prompt_in_system": bool(memo and memo[:80] in system_prompt),
                "user_prompt": user_prompt,
                "raw_response": result.raw,
                "reasoning": result.parsed.reasoning,
                "cache_key": result.cache_key,
                "from_cache": result.from_cache,
                "attempts": result.attempts,
                "parsed": result.parsed.to_dict(),
            },
        )
