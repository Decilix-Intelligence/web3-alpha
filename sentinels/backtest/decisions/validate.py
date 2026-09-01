"""Hard risk checks applied to model output before anything reaches the account.

These limits are enforced in code, not merely described in the prompt. Every
number comes from the run's RiskControlConfig, the same object system_prompt.py
renders for the model, so what the model is told and what it is held to cannot
drift apart.

A rejected decision is recorded with its reason in the cycle's decision log; it
is never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sentinels.backtest.core.config import RiskControlConfig
from sentinels.backtest.core.schema import Decision

# Position value may exceed the cap by this fraction before it is rejected,
# which absorbs rounding in the model's own arithmetic.
VALUE_TOLERANCE = 0.01


@dataclass
class ValidationResult:
    ok: bool
    decision: Decision
    reason: str = ""
    adjustments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "adjustments": list(self.adjustments),
        }


@dataclass
class PortfolioState:
    """What the validator needs to know about the account right now."""

    equity: float
    cash: float
    margin_used: float
    open_keys: List[str] = field(default_factory=list)

    def has(self, symbol: str, side: str) -> bool:
        return f"{symbol}:{side}" in self.open_keys


def validate(
    decision: Decision,
    risk: RiskControlConfig,
    state: PortfolioState,
    reference_price: Optional[float] = None,
    known_symbols: Optional[List[str]] = None,
) -> ValidationResult:
    """Check one decision. Returns a possibly-adjusted copy on success."""
    d = Decision(**decision.to_dict())
    adjustments: List[str] = []

    def reject(reason: str) -> ValidationResult:
        return ValidationResult(ok=False, decision=d, reason=reason, adjustments=adjustments)

    if d.action in ("hold", "wait"):
        return ValidationResult(ok=True, decision=d, adjustments=adjustments)

    if not d.symbol:
        return reject("decision carries no symbol")
    if known_symbols and d.symbol not in known_symbols:
        return reject(f"{d.symbol} is not part of this run (universe: {', '.join(known_symbols)})")

    if d.is_closing:
        if not state.has(d.symbol, d.side):
            return reject(f"no open {d.side} position on {d.symbol} to close")
        return ValidationResult(ok=True, decision=d, adjustments=adjustments)

    # ---- opening a position ----
    max_leverage = risk.max_leverage_for(d.symbol)
    if d.leverage <= 0:
        d.leverage = max_leverage
        adjustments.append(f"leverage unset, defaulted to {max_leverage}x")
    elif d.leverage > max_leverage:
        adjustments.append(f"leverage {d.leverage}x capped to {max_leverage}x")
        d.leverage = max_leverage

    if d.confidence < risk.min_confidence:
        return reject(f"confidence {d.confidence} below the {risk.min_confidence} required to open")

    if d.position_size_usd <= 0:
        return reject("position_size_usd must be greater than 0")

    min_size = risk.min_position_size_for(d.symbol)
    if d.position_size_usd < min_size:
        return reject(
            f"position size {d.position_size_usd:.2f} USDT below the {min_size:.0f} USDT "
            f"minimum for {d.symbol}"
        )

    max_value = risk.max_position_value_for(d.symbol, state.equity)
    if d.position_size_usd > max_value * (1 + VALUE_TOLERANCE):
        ratio = risk.position_value_ratio_for(d.symbol)
        return reject(
            f"position value {d.position_size_usd:.0f} USDT exceeds the {max_value:.0f} USDT cap "
            f"({ratio:.1f}x equity {state.equity:.0f})"
        )

    if not state.has(d.symbol, d.side) and len(state.open_keys) >= risk.max_positions:
        return reject(
            f"already holding {len(state.open_keys)} positions, limit is {risk.max_positions}"
        )

    projected_margin = state.margin_used + d.position_size_usd / d.leverage
    if state.equity > 0:
        usage = projected_margin / state.equity
        if usage > risk.max_margin_usage:
            return reject(
                f"margin usage would reach {usage * 100:.1f}%, limit is "
                f"{risk.max_margin_usage * 100:.0f}%"
            )

    if not d.stop_loss or not d.take_profit:
        return reject("opening a position requires both stop_loss and take_profit")

    if d.action == "open_long" and d.stop_loss >= d.take_profit:
        return reject(
            f"long needs stop_loss < take_profit (got {d.stop_loss:.2f} >= {d.take_profit:.2f})"
        )
    if d.action == "open_short" and d.stop_loss <= d.take_profit:
        return reject(
            f"short needs stop_loss > take_profit (got {d.stop_loss:.2f} <= {d.take_profit:.2f})"
        )

    rr = risk_reward(d, reference_price)
    if rr is None:
        return reject("cannot evaluate risk-reward without a reference price")
    if rr < risk.min_risk_reward_ratio:
        return reject(
            f"risk-reward {rr:.2f}:1 below the required {risk.min_risk_reward_ratio:.1f}:1 "
            f"(entry {reference_price:.2f}, SL {d.stop_loss:.2f}, TP {d.take_profit:.2f})"
        )

    return ValidationResult(ok=True, decision=d, adjustments=adjustments)


def risk_reward(decision: Decision, entry: Optional[float]) -> Optional[float]:
    """Reward-to-risk measured against the price the order would actually fill at.

    Deriving the entry from the stop/target levels themselves — for instance
    placing it a fixed fraction into the stop-to-target range — makes this ratio
    a constant determined by that fraction alone, so the check would pass for
    every stop and target the model could name. Anchoring on the real entry is
    what makes it a check.
    """
    if not entry or entry <= 0 or not decision.stop_loss or not decision.take_profit:
        return None

    if decision.action == "open_long":
        risk = entry - decision.stop_loss
        reward = decision.take_profit - entry
    else:
        risk = decision.stop_loss - entry
        reward = entry - decision.take_profit

    if risk <= 0:
        return 0.0  # stop is already on the wrong side of the entry
    if reward <= 0:
        return 0.0
    return reward / risk


def validate_all(
    decisions: List[Decision],
    risk: RiskControlConfig,
    state: PortfolioState,
    price_map: Optional[Dict[str, float]] = None,
    known_symbols: Optional[List[str]] = None,
) -> List[ValidationResult]:
    """Validate a batch, rejecting per decision rather than failing the whole cycle."""
    price_map = price_map or {}
    return [validate(d, risk, state, price_map.get(d.symbol), known_symbols) for d in decisions]
