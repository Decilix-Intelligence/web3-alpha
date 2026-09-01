"""Data loader for news articles and daily recommendations."""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Single news article."""
    id: int
    title: str
    content: Optional[str] = None
    ai_synopsis: Optional[str] = None
    synopsis: Optional[str] = None
    url: Optional[str] = None
    language: str = "cn"

    @property
    def title_for_sentiment(self) -> str:
        """Get title for FinBERT sentiment analysis."""
        return self.title or ""


class NewsDataLoader:
    """Load and manage news data from parquet and CSV files."""

    def __init__(
        self,
        articles_path: str,
        recommendations_path: str,
        max_news: int = 30
    ):
        self.articles_path = articles_path
        self.recommendations_path = recommendations_path
        self.max_news = max_news

        # Lazy load
        self._articles_df: Optional[pd.DataFrame] = None
        self._recommendations_df: Optional[pd.DataFrame] = None

    _BASE_COLUMNS = ["id", "title"]
    _ARTICLE_COLUMNS = ["id", "title", "ai_synopsis", "synopsis", "url", "language"]

    @property
    def articles_df(self) -> pd.DataFrame:
        """Lazy load articles dataframe."""
        if self._articles_df is None:
            logger.info(f"Loading articles from {self.articles_path}")
            if str(self.articles_path).lower().endswith(".csv"):
                self._articles_df = pd.read_csv(self.articles_path)
            else:
                # W3 article Parquet files do not necessarily contain `url`.
                # Inspect the schema first and read only columns that exist.
                import pyarrow.parquet as pq

                available = set(pq.ParquetFile(self.articles_path).schema_arrow.names)
                columns = [name for name in self._ARTICLE_COLUMNS if name in available]
                self._articles_df = pd.read_parquet(self.articles_path, columns=columns)

            missing_base = [name for name in self._BASE_COLUMNS if name not in self._articles_df]
            if missing_base:
                raise ValueError(f"Article data missing required columns: {missing_base}")
            for name in self._ARTICLE_COLUMNS:
                if name not in self._articles_df:
                    self._articles_df[name] = ""
            logger.info(f"Loaded {len(self._articles_df)} articles")
        return self._articles_df

    @property
    def recommendations_df(self) -> pd.DataFrame:
        """Lazy load recommendations dataframe."""
        if self._recommendations_df is None:
            logger.info(f"Loading recommendations from {self.recommendations_path}")
            self._recommendations_df = pd.read_csv(self.recommendations_path)
            logger.info(f"Loaded {len(self._recommendations_df)} recommendation records")
        return self._recommendations_df

    def get_article_by_id(self, article_id: int) -> Optional[NewsArticle]:
        """Get single article by ID."""
        row = self.articles_df[self.articles_df['id'] == article_id]
        if row.empty:
            return None

        row = row.iloc[0]
        return NewsArticle(
            id=int(row['id']),
            title=str(row['title']) if pd.notna(row['title']) else "",
            content=str(row['content']) if pd.notna(row.get('content')) else None,
            ai_synopsis=str(row['ai_synopsis']) if pd.notna(row.get('ai_synopsis')) else None,
            synopsis=str(row['synopsis']) if pd.notna(row.get('synopsis')) else None,
            url=str(row['url']) if pd.notna(row.get('url')) else None,
            language=str(row['language']) if pd.notna(row.get('language')) else "cn"
        )

    def get_articles_by_ids(self, article_ids: List[int]) -> List[NewsArticle]:
        """Get multiple articles by IDs."""
        articles = []
        rows = self.articles_df[self.articles_df['id'].isin(article_ids)]

        for _, row in rows.iterrows():
            articles.append(NewsArticle(
                id=int(row['id']),
                title=str(row['title']) if pd.notna(row['title']) else "",
                content=str(row['content']) if pd.notna(row.get('content')) else None,
                ai_synopsis=str(row['ai_synopsis']) if pd.notna(row.get('ai_synopsis')) else None,
                synopsis=str(row['synopsis']) if pd.notna(row.get('synopsis')) else None,
                url=str(row['url']) if pd.notna(row.get('url')) else None,
                language=str(row['language']) if pd.notna(row.get('language')) else "cn"
            ))

        return articles

    def get_recommendations_for_date(
        self,
        date: str,
        user_id: Optional[int] = None
    ) -> Dict[int, List[int]]:
        """
        Get recommended article IDs for a specific date, grouped by user_id.

        Args:
            date: Date string in 'YYYY-MM-DD' format
            user_id: Optional user ID to filter by

        Returns:
            Dict of user_id -> list of article IDs (each limited by max_news)
        """
        df = self.recommendations_df
        df['updated_at'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d')

        # Filter by date
        filtered = df[df['updated_at'] == date]

        if filtered.empty:
            logger.warning(f"No recommendations found for date {date}")
            return {}

        if user_id is not None:
            filtered = filtered[filtered['user_id'] == user_id]
            if filtered.empty:
                logger.warning(f"No recommendations found for date {date} and user_id {user_id}")
                return {}

        recommendations_map: Dict[int, List[int]] = {}
        for _, row in filtered.iterrows():
            rec_user_id = int(row['user_id'])
            ids = [int(x.strip()) for x in str(row['article_ids']).split(',') if x.strip()]
            recommendations_map[rec_user_id] = ids[:self.max_news]

        logger.info(
            "Found %s user recommendation sets for date %s (each limited to %s)",
            len(recommendations_map),
            date,
            self.max_news
        )

        return recommendations_map

    def get_news_for_date(self, date: str, user_id: Optional[int] = None) -> List[NewsArticle]:
        """
        Get news articles for a specific date.

        Args:
            date: Date string in 'YYYY-MM-DD' format
            user_id: Optional user ID to filter by

        Returns:
            List of NewsArticle objects
        """
        recommendations_map = self.get_recommendations_for_date(date, user_id)
        if not recommendations_map:
            return []

        if user_id is None:
            logger.warning("user_id is required to load news articles for a specific user")
            return []

        article_ids = recommendations_map.get(user_id, [])
        if not article_ids:
            return []

        articles = self.get_articles_by_ids(article_ids)
        logger.info(f"Loaded {len(articles)} articles for date {date}")

        return articles

    def get_available_dates(self) -> List[str]:
        """Get list of available dates in recommendations."""
        df = self.recommendations_df.copy()
        df['updated_at'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d')
        return sorted(df['updated_at'].unique().tolist())

    def get_latest_date(self) -> str:
        """Get the most recent date in recommendations."""
        dates = self.get_available_dates()
        return dates[-1] if dates else datetime.now().strftime('%Y-%m-%d')
