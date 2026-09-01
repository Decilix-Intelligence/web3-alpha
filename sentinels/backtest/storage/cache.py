"""Content-addressed cache of model decisions, so a run can be replayed exactly.

The key is a SHA-256 over the prompt variant, complete prompts, model identity,
and canonical decision context (account, positions, market snapshot at ts).
Two cycles share an entry only when every model-visible input is identical.

With `replay_only` set, a miss is an error rather than a live call: that is the
mode to use when results must be reproducible and no network access is wanted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheMiss(RuntimeError):
    """replay_only was requested but this context has no cached decision."""


class AICache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", {}) or {}
            logger.info("loaded %d cached decisions from %s", len(self.entries), self.path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read AI cache at %s (%s), starting empty", self.path, exc)
            self.entries = {}

    @staticmethod
    def key_for(variant: str, payload: Dict[str, Any]) -> str:
        blob = json.dumps(
            {"variant": variant, "context": payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry.get("response")

    def put(self, key: str, variant: str, ts: str, response: Dict[str, Any]) -> None:
        self.entries[key] = {"variant": variant, "ts": ts, "response": response}
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"entries": self.entries}, indent=2, default=str), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    @property
    def stats(self) -> Dict[str, int]:
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses}
