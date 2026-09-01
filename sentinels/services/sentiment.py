"""Sentiment analysis service.

Encapsulates news loading, FinBERT analysis, and metrics computation.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from sentinels.data import NewsDataLoader, compute_metrics
from sentinels.data.metrics import AggregatedMetrics
from sentinels.analysis import FinBERTAnalyzer
from sentinels.config import ConfigLoader

logger = logging.getLogger(__name__)

SENTIMENT_CACHE_DIR = Path(__file__).parent.parent.parent / "output" / "sentiment_cache"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""

    metrics: any
    target_date: str
    articles_count: int
    sentiment_mean: float
    dispersion: float
    bull_bear_ratio: float
    impact: float


class SentimentService:
    """Service for sentiment analysis on news articles."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize the sentiment analysis service.

        Args:
            config: Configuration loader. Uses global config if None.
        """
        if config is None:
            from sentinels.config import get_config

            config = get_config()

        self.config = config
        self.loader: Optional[NewsDataLoader] = None
        self.analyzer: Optional[FinBERTAnalyzer] = None

    def _ensure_loader(self):
        """Lazy load data loader."""
        if self.loader is None:
            self.loader = NewsDataLoader(
                articles_path=self.config.get_articles_path(),
                recommendations_path=self.config.get_recommendations_path(),
                max_news=self.config.get_max_news(),
            )

    def _ensure_analyzer(self):
        """Lazy load FinBERT analyzer."""
        if self.analyzer is None:
            self.analyzer = FinBERTAnalyzer(
                model_name=self.config.get_finbert_model(),
                batch_size=self.config.get_finbert_batch_size(),
                device=self.config.get_finbert_device(),
                config=self.config,
            )

    def _cache_key(self, target_date: str, user_id: int) -> str:
        """Build a key from inputs and inference settings, excluding credentials."""
        article_path = Path(self.config.get_articles_path())
        rec_path = Path(self.config.get_recommendations_path())
        raw = "|".join(
            [
                target_date,
                str(user_id),
                self._file_digest(article_path),
                self._file_digest(rec_path),
                self.config.get_finbert_model(),
                str(self.config.get_finbert_batch_size()),
                os.getenv("FINBERT_TRANSLATE", "0").strip().lower(),
                os.getenv("LLM_PROVIDER", "").strip(),
                os.getenv("LLM_MODEL", "").strip(),
            ]
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _file_digest(path: Path) -> str:
        if not path.is_file():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_cached(self, target_date: str, user_id: int) -> Optional[SentimentResult]:
        cache_dir = SENTIMENT_CACHE_DIR
        if not cache_dir.exists():
            return None
        path = cache_dir / f"{target_date}_u{user_id}_{self._cache_key(target_date, user_id)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            m = data["metrics"]
            metrics = AggregatedMetrics(**m)
            result = SentimentResult(
                metrics=metrics,
                target_date=data["target_date"],
                articles_count=data["articles_count"],
                sentiment_mean=data["sentiment_mean"],
                dispersion=data["dispersion"],
                bull_bear_ratio=data["bull_bear_ratio"],
                impact=data["impact"],
            )
            logger.info(
                f"Sentiment cache HIT: {target_date} user={user_id} "
                f"(mean={result.sentiment_mean:+.4f}, articles={result.articles_count})"
            )
            return result
        except Exception as e:
            logger.debug(f"Sentiment cache read failed for {target_date}/u{user_id}: {e}")
            return None

    def _save_to_cache(self, target_date: str, user_id: int, result: SentimentResult):
        cache_dir = SENTIMENT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{target_date}_u{user_id}_{self._cache_key(target_date, user_id)}.json"
        try:
            data = {
                "target_date": result.target_date,
                "articles_count": result.articles_count,
                "sentiment_mean": result.sentiment_mean,
                "dispersion": result.dispersion,
                "bull_bear_ratio": result.bull_bear_ratio,
                "impact": result.impact,
                "metrics": result.metrics.to_dict(),
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Sentiment cache write failed: {e}")

    def analyze_for_date(
        self, target_date: Optional[str] = None, user_id: Optional[int] = None
    ) -> SentimentResult:
        """Execute complete sentiment analysis for a specific date.

        Args:
            target_date: Target date (YYYY-MM-DD). Uses latest if None.
            user_id: User ID. Uses config value if None.

        Returns:
            SentimentResult object.
        """
        self._ensure_loader()
        self._ensure_analyzer()

        if target_date is None:
            target_date = self.loader.get_latest_date()
            logger.info(f"Using latest date: {target_date}")

        if user_id is None:
            user_id = self.config.get_target_user_id()

        cached = self._load_cached(target_date, user_id)
        if cached is not None:
            return cached

        logger.info(f"Loading articles for date={target_date}, user={user_id}")
        articles = self.loader.get_news_for_date(target_date, user_id=user_id)

        if not articles:
            raise ValueError(f"No articles found for date {target_date} and user {user_id}")

        logger.info(f"Loaded {len(articles)} articles")

        logger.info("Running FinBERT sentiment analysis...")
        titles = [a.title or a.synopsis or "" for a in articles]
        valid_titles = [t for t in titles if t.strip()]

        if not valid_titles:
            raise ValueError("No valid article titles to analyze")

        results = self.analyzer.analyze_batch(valid_titles)
        logger.info(f"Analyzed {len(results)} articles")

        logger.info("Computing sentiment metrics...")
        metrics = compute_metrics(results)

        logger.info(f"Sentiment Mean: {metrics.sentiment_mean:+.4f}")
        logger.info(f"Dispersion: {metrics.dispersion:.4f}")
        logger.info(f"Bull/Bear Ratio: {metrics.bull_bear_ratio:.2f}")
        logger.info(f"Impact: {metrics.impact:.4f}")

        result = SentimentResult(
            metrics=metrics,
            target_date=target_date,
            articles_count=len(articles),
            sentiment_mean=metrics.sentiment_mean,
            dispersion=metrics.dispersion,
            bull_bear_ratio=metrics.bull_bear_ratio,
            impact=metrics.impact,
        )

        self._save_to_cache(target_date, user_id, result)

        return result

    def get_latest_date(self) -> str:
        """Get the latest news date."""
        self._ensure_loader()
        return self.loader.get_latest_date()

    def get_news_for_date(self, target_date: str, user_id: Optional[int] = None) -> List:
        """Get news for a specific date.

        Args:
            target_date: Target date (YYYY-MM-DD).
            user_id: User ID. Uses config value if None.

        Returns:
            List of news articles.
        """
        self._ensure_loader()

        if user_id is None:
            user_id = self.config.get_target_user_id()

        return self.loader.get_news_for_date(target_date, user_id=user_id)
