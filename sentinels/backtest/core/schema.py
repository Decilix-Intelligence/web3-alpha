"""The action schema the model speaks, and a tolerant parser for its replies.

There is exactly one action vocabulary in this project:
    open_long | open_short | close_long | close_short | hold | wait

The parser never raises on a malformed reply. Anything it cannot understand
becomes a single `wait` decision carrying the reason, so one bad generation
costs a cycle instead of killing the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ACTIONS = ("open_long", "open_short", "close_long", "close_short", "hold", "wait")
OPENING_ACTIONS = ("open_long", "open_short")
CLOSING_ACTIONS = ("close_long", "close_short")

_REASONING_TAG = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_DECISION_TAG = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)
_JSON_ARRAY = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)
_INVISIBLE = re.compile(r"[​-‏  ﻿]")

# Models sometimes emit thousands separators ("position_size_usd": 1,200). Only
# object values are repaired: anchoring on the preceding colon keeps a genuine
# two-element array such as [1,234] from being silently fused into [1234].
_THOUSANDS = re.compile(r"(:\s*)(\d{1,3}(?:,\d{3})+)(?!\d)")

# Models drifting into full-width punctuation is common enough to be worth fixing
# rather than failing on.
_FULLWIDTH = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "：": ":",
    "，": ",",
    "　": " ",
    "【": "[",
    "】": "]",
    "、": ",",
}


@dataclass
class Decision:
    symbol: str
    action: str
    leverage: int = 0
    position_size_usd: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: int = 0
    risk_usd: float = 0.0
    reasoning: str = ""

    @property
    def is_opening(self) -> bool:
        return self.action in OPENING_ACTIONS

    @property
    def is_closing(self) -> bool:
        return self.action in CLOSING_ACTIONS

    @property
    def side(self) -> str:
        return "long" if self.action.endswith("_long") else "short"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "leverage": self.leverage,
            "position_size_usd": self.position_size_usd,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "risk_usd": self.risk_usd,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Optional["Decision"]:
        if not isinstance(raw, dict):
            return None
        action = str(raw.get("action", "")).strip().lower()
        if action not in ACTIONS:
            return None
        return cls(
            symbol=str(raw.get("symbol", "")).strip().upper(),
            action=action,
            leverage=_as_int(raw.get("leverage")),
            position_size_usd=_as_float(raw.get("position_size_usd")),
            stop_loss=_as_opt_float(raw.get("stop_loss")),
            take_profit=_as_opt_float(raw.get("take_profit")),
            confidence=_as_int(raw.get("confidence")),
            risk_usd=_as_float(raw.get("risk_usd")),
            reasoning=str(raw.get("reasoning", ""))[:2000],
        )


@dataclass
class ParsedResponse:
    reasoning: str = ""
    decisions: List[Decision] = field(default_factory=list)
    raw: str = ""
    fallback: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "decisions": [d.to_dict() for d in self.decisions],
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


def parse_response(text: str) -> ParsedResponse:
    """Pull reasoning and the decision array out of a model reply."""
    raw = text or ""
    if not raw.strip():
        return _safe_wait(raw, "model returned an empty response")

    cleaned = _INVISIBLE.sub("", raw)
    for bad, good in _FULLWIDTH.items():
        cleaned = cleaned.replace(bad, good)

    reasoning = _extract_reasoning(cleaned)

    tagged = _DECISION_TAG.search(cleaned)
    payload = tagged.group(1).strip() if tagged else cleaned

    blob = None
    fenced = _JSON_FENCE.search(payload)
    if fenced:
        blob = fenced.group(1)
    else:
        array = _JSON_ARRAY.search(payload)
        if array:
            blob = array.group(0)
        else:
            start, end = payload.find("["), payload.rfind("]")
            if start != -1 and end > start:
                blob = payload[start : end + 1]

    if blob is None:
        return _safe_wait(raw, "no JSON decision array found in response", reasoning)

    blob = _THOUSANDS.sub(lambda m: m.group(1) + m.group(2).replace(",", ""), blob)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return _safe_wait(raw, f"decision JSON did not parse: {exc}", reasoning)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return _safe_wait(raw, "decision JSON was not a list", reasoning)

    decisions = [d for d in (Decision.from_dict(item) for item in data) if d is not None]
    if not decisions:
        return _safe_wait(raw, "no decisions with a recognised action", reasoning)

    return ParsedResponse(reasoning=reasoning, decisions=decisions, raw=raw)


def _extract_reasoning(text: str) -> str:
    match = _REASONING_TAG.search(text)
    if match:
        return match.group(1).strip()
    idx = text.find("<decision>")
    if idx > 0:
        return text[:idx].strip()
    idx = text.find("[")
    if idx > 0:
        return text[:idx].strip()
    return text.strip()


def _safe_wait(raw: str, reason: str, reasoning: str = "") -> ParsedResponse:
    return ParsedResponse(
        reasoning=reasoning,
        decisions=[Decision(symbol="ALL", action="wait", reasoning=reason)],
        raw=raw,
        fallback=True,
        fallback_reason=reason,
    )


def _as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_opt_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None
