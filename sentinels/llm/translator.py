"""
Batch translation utilities built on top of sentinels.llm BaseLLM providers.

Goal: translate Chinese financial news titles to English before feeding FinBERT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .base import BaseLLM

logger = logging.getLogger(__name__)


def _chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese_chars / max(1, len(text))


def _needs_zh_to_en_translation(text: str, ratio_threshold: float = 0.001) -> bool:
    # Translate if there's any Chinese character (threshold ~0.1%)
    return bool(text and text.strip()) and _chinese_char_ratio(text) >= ratio_threshold


def _extract_json_array(text: str) -> str:
    """
    Extract the first JSON array substring from a response.
    We intentionally tolerate extra prose around the JSON to be resilient.
    """
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in response")
    return match.group(0)


@dataclass
class BatchTranslator:
    """
    Batch Chinese -> English translator using an OpenAI-compatible LLM via BaseLLM.complete().

    Notes:
    - Returns a list of strings with the same length as input.
    - Only translates items that look Chinese (by heuristic); others are passed through.
    - On parse/format failure, falls back to per-item translate_to_english().
    - Retries on empty translations.
    - Supports persistent caching to avoid repeated API calls for the same text.
    """

    llm: BaseLLM
    cache_path: Optional[str] = None  # Path to JSON cache file
    _cache: Dict[str, str] = field(default_factory=dict, init=False)  # Memory cache

    def __post_init__(self):
        """Load translation cache from file if it exists."""
        if self.cache_path:
            cache_file = Path(self.cache_path)
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                    logger.info(
                        f"Loaded {len(self._cache)} cached translations from {self.cache_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load translation cache: {e}")
                    self._cache = {}
            else:
                # Ensure parent directory exists
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                self._cache = {}

    def _save_cache(self):
        """Persist translation cache to file."""
        if self.cache_path:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {len(self._cache)} translations to cache")
            except Exception as e:
                logger.warning(f"Failed to save translation cache: {e}")

    def _translate_single_with_retry(self, text: str, max_retries: int = 3) -> str:
        """Translate single text with retry on empty result."""
        for attempt in range(max_retries):
            try:
                result = self.llm.translate_to_english(text)
                if result and result.strip():
                    return result
                logger.warning(f"Attempt {attempt+1}: Empty translation for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: Translation error: {e}")

        # After all retries failed, return original
        logger.error(f"All retries failed for: {text[:50]}..., keeping original")
        return text

    batch_size: int = 30

    def _translate_chunk(self, chunk: List[str], max_retries: int = 5) -> Optional[List[str]]:
        """Translate a single chunk via the batch API. Returns None on failure."""
        system_prompt = (
            "You are a financial news translator.\n"
            "Translate each non-English item to natural, concise English.\n"
            "Rules:\n"
            "1) Keep financial terms accurate (translate monetary-policy jargon precisely)\n"
            "2) Preserve emotional tone (crash/surge/panic/rally)\n"
            "3) Keep names/tickers (BTC/ETH) unchanged\n"
            "4) Output ONLY a valid JSON array of strings, same length/order as input\n"
            "5) No markdown, no explanations"
        )
        prompt = (
            "Translate this JSON array into English.\n"
            "Return ONLY a JSON array of strings.\n\n"
            f"INPUT_JSON={json.dumps(chunk, ensure_ascii=False)}"
        )

        for attempt in range(max_retries):
            try:
                response = self.llm.complete(prompt=prompt, system_prompt=system_prompt)
                arr_text = _extract_json_array(response or "")
                translations = json.loads(arr_text)
                if not isinstance(translations, list) or len(translations) != len(chunk):
                    raise ValueError(
                        f"Length mismatch: expected {len(chunk)}, got {len(translations)}"
                    )
                translations = [str(x).strip() for x in translations]
                empty_count = sum(1 for t in translations if not t)
                if empty_count > 0 and attempt < max_retries - 1:
                    logger.warning(
                        f"Chunk attempt {attempt+1}: {empty_count} empty translations, retrying..."
                    )
                    continue
                return translations
            except Exception as e:
                logger.warning(f"Chunk attempt {attempt+1}/{max_retries}: {e}")

        return None

    def translate_batch(self, texts: Sequence[str]) -> List[str]:
        texts_list = list(texts)
        if not texts_list:
            return []

        to_translate: List[str] = []
        to_translate_indices: List[int] = []
        out: List[str] = list(texts_list)
        cache_hits = 0

        for i, t in enumerate(texts_list):
            ratio = _chinese_char_ratio(t)
            needs_trans = _needs_zh_to_en_translation(t)
            if needs_trans:
                if t in self._cache:
                    out[i] = self._cache[t]
                    cache_hits += 1
                    logger.debug(f"[{i}] CACHE HIT: {t[:50]}...")
                else:
                    to_translate.append(t)
                    to_translate_indices.append(i)
                    logger.debug(f"[{i}] WILL translate (zh_ratio={ratio:.2%}): {t[:50]}...")
            else:
                logger.debug(f"[{i}] SKIP translate (zh_ratio={ratio:.2%}): {t[:50]}...")

        if cache_hits > 0:
            logger.info(f"Translation cache: {cache_hits} hits, {len(to_translate)} misses")

        if not to_translate:
            return out

        translations: List[str] = []
        n_chunks = (len(to_translate) + self.batch_size - 1) // self.batch_size
        logger.info(
            f"Translating {len(to_translate)} texts in {n_chunks} chunks "
            f"(batch_size={self.batch_size})"
        )

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.batch_size
            end = min(start + self.batch_size, len(to_translate))
            chunk = to_translate[start:end]

            chunk_result = self._translate_chunk(chunk)
            if chunk_result is not None:
                translations.extend(chunk_result)
            else:
                logger.warning(f"Chunk {chunk_idx+1}/{n_chunks} failed, falling back to per-item")
                for t in chunk:
                    translations.append(self._translate_single_with_retry(t, max_retries=3))

        new_cache_entries = 0
        for idx, translated in zip(to_translate_indices, translations):
            if translated:
                out[idx] = translated
                original_text = texts_list[idx]
                if original_text not in self._cache:
                    self._cache[original_text] = translated
                    new_cache_entries += 1
            else:
                logger.error(
                    f"Translation still empty for: "
                    f"{to_translate[to_translate_indices.index(idx)][:50]}..."
                )

        if new_cache_entries > 0:
            logger.info(f"Added {new_cache_entries} new translations to cache")
            self._save_cache()

        return out
