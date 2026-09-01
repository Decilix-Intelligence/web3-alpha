"""Offline sentiment annotator for NewsRecLib smoke and batch runs."""

from __future__ import annotations


class DummySentimentAnnotator:
    """Return a stable neutral sentiment without external model downloads."""

    def __call__(self, text: str) -> tuple[str, float]:
        return "neutral", 0.0
