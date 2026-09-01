"""
FinBERT sentiment analysis wrapper.
Uses ProsusAI/finbert for financial sentiment classification.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Protocol
import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

logger = logging.getLogger(__name__)


class Translator(Protocol):
    def translate_batch(self, texts: Sequence[str]) -> List[str]: ...


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""

    text: str
    label: str  # positive / negative / neutral
    positive: float
    negative: float
    neutral: float
    score: float  # positive - negative (range: -1 to 1)
    confidence: float  # max probability


class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer."""

    LABEL_MAP = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "Positive": "positive",
        "Negative": "negative",
        "Neutral": "neutral",
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: str = "auto",
        translator: Optional[Translator] = None,
        enable_translation: Optional[bool] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize FinBERT analyzer.

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            device: 'auto', 'cuda', or 'cpu'
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.translator = translator
        self.config = config

        if enable_translation is None:
            # Translation can disclose article text to an external service, so it
            # is disabled unless the operator explicitly opts in.
            env_flag = os.getenv("FINBERT_TRANSLATE")
            if env_flag is None or not env_flag.strip():
                enable_translation = False
            else:
                enable_translation = env_flag.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }
        self.enable_translation = bool(enable_translation)

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading FinBERT model: {model_name} on {self.device}")

        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

        # Create pipeline
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            batch_size=batch_size,
            return_all_scores=True,  # Get all class probabilities
        )

        logger.info("FinBERT model loaded successfully")

    def _ensure_translator(self) -> None:
        if self.translator is not None or not self.enable_translation:
            return

        api_key = (os.getenv("LLM_API_KEY") or "").strip()
        provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        base_url = (os.getenv("LLM_BASE_URL") or "").strip()
        model = (os.getenv("LLM_MODEL") or "").strip()

        # If not provided by env, try config (ConfigLoader-style).
        if self.config is not None:
            try:
                if not provider and hasattr(self.config, "get_llm_provider"):
                    provider = str(self.config.get_llm_provider() or "").strip().lower()
                if not base_url and hasattr(self.config, "get_llm_base_url"):
                    base_url = str(self.config.get_llm_base_url() or "").strip()
                if not model and hasattr(self.config, "get_llm_model"):
                    model = str(self.config.get_llm_model() or "").strip()
                if not api_key and hasattr(self.config, "get_llm_api_key"):
                    api_key = str(self.config.get_llm_api_key() or "").strip()
            except Exception:
                # Keep best-effort; translator init below will handle missing values.
                pass

        if not api_key:
            logger.warning(
                "FINBERT_TRANSLATE is enabled but LLM_API_KEY is missing; skipping translation."
            )
            return

        provider = provider or "openai"

        try:
            from sentinels.llm import create_llm
            from sentinels.llm.translator import BatchTranslator

            llm = create_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

            self.translator = BatchTranslator(llm=llm)
        except Exception as e:
            logger.warning(f"Failed to initialize translator; skipping translation. Error: {e}")

    def _maybe_translate_batch(self, texts: Sequence[str]) -> List[str]:
        self._ensure_translator()
        if self.translator is None:
            return list(texts)

        try:
            translated = self.translator.translate_batch(texts)

            return translated
        except Exception as e:
            logger.warning(f"Translation failed; using original text. Error: {e}")
            return list(texts)

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text (title or summary)

        Returns:
            SentimentResult object
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label="neutral",
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                score=0.0,
                confidence=1.0,
            )

        # Optional translate (Chinese -> English) before truncation / FinBERT
        text = self._maybe_translate_batch([text])[0]

        # Truncate long text
        text = text[:512]

        results = self.pipeline(text)

        # Parse results (list of dicts with label and score)
        probs = {self.LABEL_MAP.get(r["label"], r["label"]): r["score"] for r in results[0]}

        positive = probs.get("positive", 0.0)
        negative = probs.get("negative", 0.0)
        neutral = probs.get("neutral", 0.0)

        # Determine label
        max_prob = max(positive, negative, neutral)
        if max_prob == positive:
            label = "positive"
        elif max_prob == negative:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            text=text,
            label=label,
            positive=positive,
            negative=negative,
            neutral=neutral,
            score=positive - negative,
            confidence=max_prob,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        # Filter empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text[:512])  # Truncate
                valid_indices.append(i)

        # Initialize results with neutral for empty texts
        results = [
            SentimentResult(
                text=texts[i],
                label="neutral",
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                score=0.0,
                confidence=1.0,
            )
            for i in range(len(texts))
        ]

        if not valid_texts:
            return results

        # Optional translate (Chinese -> English) before truncation / FinBERT
        valid_texts = self._maybe_translate_batch(valid_texts)

        # Truncate after translation (FinBERT input limit)
        valid_texts = [t[:512] for t in valid_texts]

        # Batch inference
        try:
            batch_results = self.pipeline(valid_texts)

            for idx, (i, text_result) in enumerate(zip(valid_indices, batch_results)):
                probs = {
                    self.LABEL_MAP.get(r["label"], r["label"]): r["score"] for r in text_result
                }

                positive = probs.get("positive", 0.0)
                negative = probs.get("negative", 0.0)
                neutral = probs.get("neutral", 0.0)

                max_prob = max(positive, negative, neutral)
                if max_prob == positive:
                    label = "positive"
                elif max_prob == negative:
                    label = "negative"
                else:
                    label = "neutral"

                results[i] = SentimentResult(
                    text=valid_texts[idx],
                    label=label,
                    positive=positive,
                    negative=negative,
                    neutral=neutral,
                    score=positive - negative,
                    confidence=max_prob,
                )

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            # Fallback to individual analysis
            for idx, i in enumerate(valid_indices):
                try:
                    results[i] = self.analyze(valid_texts[idx])
                except Exception as inner_e:
                    logger.error(f"Individual analysis failed: {inner_e}")

        return results
