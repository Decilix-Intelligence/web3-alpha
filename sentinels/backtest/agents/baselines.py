"""Rule-based agents: the references the model has to beat.

They emit the same decision schema as the model and pass through the same
validator, so they are held to the same risk limits rather than waved past them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sentinels.backtest.core.config import RiskControlConfig
from sentinels.backtest.core.protocols import AgentDecision
from sentinels.backtest.decisions.context import DecisionContext
from sentinels.backtest.core.schema import Decision

logger = logging.getLogger(__name__)

# Rule agents have no calibrated confidence; they state the threshold so the
# validator's confidence gate neither blocks them nor is silently bypassed.
RULE_AGENT_CONFIDENCE = 80


class BuyAndHoldAgent:
    """Unlevered long on every symbol from the first cycle, then nothing.

    Leverage is 1 so it cannot be liquidated, which is what makes it a floor
    rather than another strategy.
    """

    name = "buy_and_hold"

    def __init__(self, risk: RiskControlConfig, allocation: float = 0.4):
        self.risk = risk
        self.allocation = allocation

    def decide(self, ctx: DecisionContext) -> AgentDecision:
        held = {p["symbol"] for p in ctx.positions if p["side"] == "long"}
        decisions: List[Decision] = []

        for symbol in ctx.candidates:
            snap = ctx.market.get(symbol)
            if snap is None:
                continue
            if symbol in held:
                decisions.append(Decision(symbol=symbol, action="hold", reasoning="buy and hold"))
                continue

            price = snap.current_price
            size = ctx.account["equity"] * self.allocation
            if size < self.risk.min_position_size_for(symbol):
                decisions.append(
                    Decision(symbol=symbol, action="wait", reasoning="allocation below minimum")
                )
                continue

            decisions.append(
                Decision(
                    symbol=symbol,
                    action="open_long",
                    leverage=1,
                    position_size_usd=size,
                    # Wide bracket kept only to satisfy the risk-reward floor; at
                    # 1x there is no liquidation and neither level is expected to
                    # be reached inside a run this short.
                    stop_loss=price * 0.5,
                    take_profit=price * 2.5,
                    confidence=RULE_AGENT_CONFIDENCE,
                    reasoning="buy and hold entry",
                )
            )

        return AgentDecision(decisions, provenance={"agent": self.name, "held": sorted(held)})


class EmaCrossAgent:
    """Long while EMA20 is above EMA50, flat otherwise. No model involved.

    Stops and targets are sized from ATR so the bracket adapts to volatility and
    clears the configured risk-reward floor by construction.
    """

    name = "ema_cross"

    def __init__(
        self,
        risk: RiskControlConfig,
        timeframe: str = "15m",
        atr_stop_multiple: float = 1.5,
        allocation: float = 0.5,
    ):
        self.risk = risk
        self.timeframe = timeframe
        self.atr_stop_multiple = atr_stop_multiple
        self.allocation = allocation

    def decide(self, ctx: DecisionContext) -> AgentDecision:
        held = {p["symbol"] for p in ctx.positions if p["side"] == "long"}
        decisions: List[Decision] = []
        signals: Dict[str, Any] = {}

        for symbol in ctx.candidates:
            snap = ctx.market.get(symbol)
            tf = snap.by_tf.get(self.timeframe) if snap else None
            if tf is None or tf.ema20 is None or tf.ema50 is None or tf.atr14 is None:
                decisions.append(
                    Decision(symbol=symbol, action="wait", reasoning="indicators not warm yet")
                )
                continue

            bullish = tf.ema20 > tf.ema50
            signals[symbol] = {
                "ema20": tf.ema20,
                "ema50": tf.ema50,
                "atr14": tf.atr14,
                "bullish": bullish,
            }

            if bullish and symbol not in held:
                decisions.append(self._entry(symbol, snap.current_price, tf.atr14, ctx))
            elif not bullish and symbol in held:
                decisions.append(
                    Decision(
                        symbol=symbol,
                        action="close_long",
                        reasoning=f"EMA20 {tf.ema20:.2f} fell below EMA50 {tf.ema50:.2f}",
                    )
                )
            else:
                decisions.append(
                    Decision(
                        symbol=symbol,
                        action="hold" if symbol in held else "wait",
                        reasoning="no cross",
                    )
                )

        return AgentDecision(decisions, provenance={"agent": self.name, "signals": signals})

    def _entry(self, symbol: str, price: float, atr: float, ctx: DecisionContext) -> Decision:
        risk_distance = max(atr * self.atr_stop_multiple, price * 0.001)
        # a hair above the floor, so rounding cannot push it under
        reward_distance = risk_distance * (self.risk.min_risk_reward_ratio + 0.2)

        cap = self.risk.max_position_value_for(symbol, ctx.account["equity"])
        size = min(ctx.account["equity"] * self.allocation, cap)

        if size < self.risk.min_position_size_for(symbol):
            return Decision(symbol=symbol, action="wait", reasoning="allocation below minimum")

        return Decision(
            symbol=symbol,
            action="open_long",
            leverage=self.risk.max_leverage_for(symbol),
            position_size_usd=size,
            stop_loss=price - risk_distance,
            take_profit=price + reward_distance,
            confidence=RULE_AGENT_CONFIDENCE,
            reasoning=f"EMA20 above EMA50, stop {self.atr_stop_multiple}x ATR",
        )


BASELINE_AGENTS = {
    BuyAndHoldAgent.name: BuyAndHoldAgent,
    EmaCrossAgent.name: EmaCrossAgent,
}


def build_baseline(name: str, risk: RiskControlConfig, **kwargs):
    """Construct a baseline agent by name."""
    if name not in BASELINE_AGENTS:
        raise ValueError(f"unknown baseline '{name}' (have {', '.join(sorted(BASELINE_AGENTS))})")
    return BASELINE_AGENTS[name](risk, **kwargs)
