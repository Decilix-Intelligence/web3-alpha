"""Daily sentiment pipeline: news articles in, strategy memo out.

    FinBERT sentiment -> aggregated factors -> the memo the agent is given.

Both steps run offline against the sample data. The memo is consumed by
scripts/run_mini_backtest.py, which injects it into the agent's system prompt.
"""

import logging
from typing import Optional

from sentinels.pipeline.base import Pipeline, PipelineStep, PipelineContext
from sentinels.config import ConfigLoader, get_config
from sentinels.services.sentiment import SentimentService
from sentinels.engine.formatter import EnginePromptFormatter

logger = logging.getLogger(__name__)


class SentimentAnalysisStep(PipelineStep):
    """Sentiment analysis step."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        super().__init__("SentimentAnalysis")
        self.config = config or get_config()
        self.service = SentimentService(self.config)

    def execute(self, context: PipelineContext) -> bool:
        """Execute sentiment analysis."""
        self.logger.info("Running sentiment analysis...")

        target_date = context.get("target_date")
        user_id = context.get("user_id")

        result = self.service.analyze_for_date(target_date=target_date, user_id=user_id)

        context.set("sentiment_result", result)
        context.set("target_date", result.target_date)
        context.set("sentiment_metrics", result.metrics)

        self.logger.info(f"Sentiment analysis completed for {result.target_date}")
        self.logger.info(f"  Articles: {result.articles_count}")
        self.logger.info(f"  Sentiment: {result.sentiment_mean:+.4f}")
        self.logger.info(f"  Dispersion: {result.dispersion:.4f}")

        return True


class FormatPromptStep(PipelineStep):
    """Format prompt step."""

    def __init__(self):
        super().__init__("FormatPrompt")
        self.formatter = EnginePromptFormatter()

    def execute(self, context: PipelineContext) -> bool:
        """Format the engine prompt."""
        self.logger.info("Formatting prompt for Engine...")

        # Get sentiment metrics from context
        sentiment_metrics = context.get("sentiment_metrics")
        target_date = context.get("target_date")

        if not sentiment_metrics:
            self.logger.error("No sentiment metrics found in context")
            return False

        # Format
        prompt_text = self.formatter.format_for_engine(sentiment_metrics, target_date)

        # Store in context
        context.set("prompt_text", prompt_text)

        self.logger.info(f"Prompt formatted: {len(prompt_text)} characters")

        return True


class DailySentimentPipeline:
    """Scores one day of news and formats the result as a strategy memo."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        self.config = config or get_config()
        self.pipeline = Pipeline(
            name="DailySentimentPipeline",
            steps=[SentimentAnalysisStep(self.config), FormatPromptStep()],
            stop_on_error=True,
        )

    def run(
        self,
        target_date: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> PipelineContext:
        """Run the pipeline.

        Args:
            target_date: target date (YYYY-MM-DD); uses the latest available if None
            user_id: user id; uses the config value if None

        Returns:
            The pipeline context, carrying `sentiment_metrics` and `prompt_text`.
        """
        return self.pipeline.run(PipelineContext({"target_date": target_date, "user_id": user_id}))
