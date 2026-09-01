"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsSummary:
    """Summary result from LLM."""

    original_title: str
    original_content: Optional[str]
    summary: str
    category: str  # MACRO / COIN_SPECIFIC / EDUCATION / MARKET_ANALYSIS / OTHER
    coins_mentioned: List[str]
    title_en: str  # English translation of title (for FinBERT)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Basic completion API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response text
        """
        pass

    def translate_to_english(self, text: str) -> str:
        """
        Translate Chinese text to English.
        Preserves financial terminology.
        """
        if not text:
            return ""

        # Check if already English (simple heuristic)
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if chinese_chars < len(text) * 0.001:
            return text  # Already mostly English (< 0.1% Chinese)

        system_prompt = """You are a financial news translator.
Translate the following text to English.
Rules:
1. Keep financial terms accurate (translate monetary-policy jargon precisely)
2. Preserve the emotional tone (bullish/bearish/neutral)
3. Keep it concise
4. Output ONLY the translation, no explanations."""

        prompt = f"Translate to English:\n{text}"

        try:
            result = self.complete(prompt, system_prompt)
            # If translation is empty or whitespace, return original
            if result and result.strip():
                return result
            else:
                logger.warning(f"Empty translation result, keeping original: {text[:50]}...")
                return text
        except Exception as e:
            logger.warning(f"Translation error, keeping original: {e}")
            return text

    def summarize_news(self, title: str, content: Optional[str] = None) -> NewsSummary:
        """
        Generate structured summary of a news article.

        Args:
            title: News title
            content: Optional full content

        Returns:
            NewsSummary object
        """
        system_prompt = """You are a financial news analyst. Analyze and summarize the news.

Output format (JSON):
{
    "summary": "1-2 sentence summary in Chinese",
    "category": "MACRO|COIN_SPECIFIC|EDUCATION|MARKET_ANALYSIS|OTHER",
    "coins_mentioned": ["BTC", "ETH", ...],
    "title_en": "English translation of the title"
}

Category definitions:
- MACRO: Macroeconomic news (interest rates, inflation, Fed policy)
- COIN_SPECIFIC: News about specific cryptocurrencies
- EDUCATION: Educational content, tutorials
- MARKET_ANALYSIS: Technical analysis, market commentary
- OTHER: Other news

Rules:
1. Keep summary factual and concise
2. Identify ALL cryptocurrency symbols mentioned
3. For title_en, preserve emotional words (crash, surge, panic, rally)
4. Output valid JSON only"""

        news_text = title
        if content:
            news_text = f"Title: {title}\n\nContent: {content[:500]}..."  # Limit content length

        prompt = f"Analyze this news:\n{news_text}"

        response = self.complete(prompt, system_prompt)

        # Parse JSON response
        return self._parse_summary_response(response, title, content)

    def _parse_summary_response(
        self, response: str, original_title: str, original_content: Optional[str]
    ) -> NewsSummary:
        """Parse LLM response into NewsSummary object."""
        import json
        import re

        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:100]}")
            data = {}

        return NewsSummary(
            original_title=original_title,
            original_content=original_content,
            summary=data.get("summary", original_title),
            category=data.get("category", "OTHER"),
            coins_mentioned=data.get("coins_mentioned", []),
            title_en=data.get("title_en", self.translate_to_english(original_title)),
        )

    def batch_summarize(self, articles: List[tuple]) -> List[NewsSummary]:
        """
        Batch summarize multiple articles.

        Args:
            articles: List of (title, content) tuples

        Returns:
            List of NewsSummary objects
        """
        summaries = []
        for title, content in articles:
            try:
                summary = self.summarize_news(title, content)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize article: {title[:50]}... Error: {e}")
                # Fallback to basic summary
                summaries.append(
                    NewsSummary(
                        original_title=title,
                        original_content=content,
                        summary=title,
                        category="OTHER",
                        coins_mentioned=[],
                        title_en=title,
                    )
                )

        return summaries
