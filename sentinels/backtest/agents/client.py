"""Everything between "here is the state" and "here are the decisions".

Cache lookup, retry with backoff, response parsing and cache write live here so
the loop never has to know whether a decision came from the provider or from
disk. The loop asks once and gets a typed answer.

Failure is a value, not an exception: a provider that will not answer produces a
result whose decisions are a single `wait`, and the run continues. The one
exception is `replay_only` on a cache miss, which is a configuration error the
caller must see.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from sentinels.backtest.storage.cache import AICache, CacheMiss
from sentinels.backtest.core.schema import ParsedResponse, parse_response

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    raw: str
    parsed: ParsedResponse
    cache_key: str
    from_cache: bool = False
    error: str = ""
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "from_cache": self.from_cache,
            "attempts": self.attempts,
            "error": self.error,
            "parsed": self.parsed.to_dict(),
        }


class DecisionClient:
    """Asks the model for decisions, or recovers them from the cache."""

    def __init__(
        self,
        llm: Any,
        cache: AICache,
        *,
        variant: str,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        cache_ai: bool = True,
        replay_only: bool = False,
    ):
        self.llm = llm
        self.cache = cache
        self.variant = variant
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay
        self.cache_ai = cache_ai
        self.replay_only = replay_only
        self.calls = 0

    def decide(
        self,
        cache_payload: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        ts: datetime,
    ) -> DecisionResult:
        # Prompts carry the news memo and rendered risk limits, while the model
        # identity changes provider behavior. All are part of replay identity.
        keyed_payload = {
            "context": cache_payload,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "provider_class": (f"{type(self.llm).__module__}.{type(self.llm).__qualname__}"),
            "model": getattr(self.llm, "model", ""),
            "temperature": getattr(self.llm, "temperature", None),
            "max_tokens": getattr(self.llm, "max_tokens", None),
        }
        key = AICache.key_for(self.variant, keyed_payload)

        cached = self.cache.get(key) if self.cache_ai else None
        if cached is not None:
            raw = cached.get("raw", "")
            return DecisionResult(
                raw=raw, parsed=parse_response(raw), cache_key=key, from_cache=True
            )

        if self.replay_only:
            raise CacheMiss(
                f"replay_only is set but no cached decision exists for {ts.isoformat()}"
            )

        raw, error, attempts = self._call_with_retry(system_prompt, user_prompt)
        self.calls += 1

        if raw and self.cache_ai:
            self.cache.put(key, self.variant, ts.isoformat(), {"raw": raw})

        return DecisionResult(
            raw=raw,
            parsed=parse_response(raw),
            cache_key=key,
            from_cache=False,
            error=error,
            attempts=attempts,
        )

    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> tuple[str, str, int]:
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                raw = self.llm.complete(user_prompt, system_prompt=system_prompt)
                if raw and raw.strip():
                    return raw, "", attempt + 1
                last_error = "model returned an empty completion"
            except Exception as exc:  # provider errors vary too much to enumerate
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt + 1 < self.max_retries:
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "model call failed (attempt %d/%d): %s - retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        return (
            "",
            f"model call failed after {self.max_retries} attempts: {last_error}",
            self.max_retries,
        )
