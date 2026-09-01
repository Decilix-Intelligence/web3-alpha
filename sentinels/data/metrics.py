"""
Sentiment aggregation and metrics computation.
"""
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable
import numpy as np
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class SentimentResultProtocol(Protocol):
    """Protocol for sentiment result objects."""
    text: str
    label: str
    positive: float
    negative: float
    neutral: float
    score: float
    confidence: float


@dataclass
class AggregatedMetrics:
    """Aggregated sentiment metrics."""
    # Basic statistics
    sentiment_mean: float  # St = mean(scores)
    sentiment_std: float  # σt = std(scores)

    # Advanced metrics
    dispersion: float  # disagreement = sigma_t
    impact: float  # impact = St * log(1 + n)
    bull_bear_ratio: float  # bull/bear = positive_count / negative_count
    extreme_ratio: float  # extreme ratio = high_confidence_count / total

    # Counts
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int

    # Aggregated probabilities
    avg_positive: float
    avg_negative: float
    avg_neutral: float
    avg_confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sentiment_mean": round(self.sentiment_mean, 4),
            "sentiment_std": round(self.sentiment_std, 4),
            "dispersion": round(self.dispersion, 4),
            "impact": round(self.impact, 4),
            "bull_bear_ratio": round(self.bull_bear_ratio, 4),
            "extreme_ratio": round(self.extreme_ratio, 4),
            "news_count": self.news_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "avg_positive": round(self.avg_positive, 4),
            "avg_negative": round(self.avg_negative, 4),
            "avg_neutral": round(self.avg_neutral, 4),
            "avg_confidence": round(self.avg_confidence, 4)
        }


def compute_metrics(results: List[SentimentResultProtocol]) -> AggregatedMetrics:
    """
    Compute aggregated sentiment metrics from individual results.

    Args:
        results: List of SentimentResult objects

    Returns:
        AggregatedMetrics object
    """
    if not results:
        return AggregatedMetrics(
            sentiment_mean=0.0,
            sentiment_std=0.0,
            dispersion=0.0,
            impact=0.0,
            bull_bear_ratio=1.0,
            extreme_ratio=0.0,
            news_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            avg_positive=0.0,
            avg_negative=0.0,
            avg_neutral=0.0,
            avg_confidence=0.0
        )

    # Extract scores
    scores = [r.score for r in results]
    n = len(scores)

    # Basic statistics
    sentiment_mean = float(np.mean(scores))
    sentiment_std = float(np.std(scores)) if n > 1 else 0.0

    # Counts
    positive_count = sum(1 for r in results if r.label == "positive")
    negative_count = sum(1 for r in results if r.label == "negative")
    neutral_count = sum(1 for r in results if r.label == "neutral")

    # Advanced metrics
    dispersion = sentiment_std
    impact = sentiment_mean * np.log(1 + n)  # impact = St * log(1 + Nt)
    # Bull/Bear ratio with Laplace smoothing to avoid division by zero and extreme spikes
    bull_bear_ratio = (positive_count + 1) / (negative_count + 1)
    extreme_count = sum(1 for r in results if r.confidence > 0.7 and r.label != "neutral")
    extreme_ratio = extreme_count / n  # extreme ratio (high-confidence positive/negative only)

    # Average probabilities
    avg_positive = float(np.mean([r.positive for r in results]))
    avg_negative = float(np.mean([r.negative for r in results]))
    avg_neutral = float(np.mean([r.neutral for r in results]))
    avg_confidence = float(np.mean([r.confidence for r in results]))

    return AggregatedMetrics(
        sentiment_mean=sentiment_mean,
        sentiment_std=sentiment_std,
        dispersion=dispersion,
        impact=impact,
        bull_bear_ratio=bull_bear_ratio,
        extreme_ratio=extreme_ratio,
        news_count=n,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        avg_positive=avg_positive,
        avg_negative=avg_negative,
        avg_neutral=avg_neutral,
        avg_confidence=avg_confidence
    )
