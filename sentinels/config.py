"""Configuration loader for news2alpha.

Loads settings from config.yaml and backtest_config.yaml with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


class ConfigLoader:
    """Unified configuration loader."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_dir: Config directory path. Defaults to project_root/config.
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._config = {}
        self._backtest_config = {}

        self._load_configs()

    def _load_configs(self):
        """Load all configuration files."""
        config_file = self.config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

        backtest_file = self.config_dir / "backtest_config.yaml"
        if backtest_file.exists():
            with open(backtest_file, "r", encoding="utf-8") as f:
                self._backtest_config = yaml.safe_load(f) or {}

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path.

        Args:
            path: Config path like "llm.provider" or "backtest.symbols".
            default: Default value.

        Returns:
            Configuration value.
        """
        parts = path.split(".")

        value = self._get_nested(self._backtest_config, parts)
        if value is not None:
            return value

        value = self._get_nested(self._config, parts)
        if value is not None:
            return value

        return default

    def _get_nested(self, data: Dict, keys: list) -> Any:
        """Get value from nested dictionary."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def get_backtest_symbols(self) -> list:
        """Get backtest trading pairs."""
        return self.get("backtest.symbols", ["BTCUSDT", "ETHUSDT"])

    def get_backtest_timeframes(self) -> list:
        """Get backtest timeframes."""
        return self.get("backtest.timeframes", ["15m", "4h"])

    def get_decision_timeframe(self) -> str:
        """Get decision timeframe."""
        return self.get("backtest.decision_timeframe", "15m")

    def get_decision_cadence(self) -> int:
        """Get decision cadence (bars per decision)."""
        return self.get("backtest.decision_cadence_nbars", 20)

    def get_initial_balance(self) -> float:
        """Get initial balance."""
        return self.get("backtest.initial_balance", 1000.0)

    def get_leverage_config(self) -> Dict[str, int]:
        """Get leverage configuration."""
        return self.get("backtest.leverage", {"btc_eth": 5, "altcoin": 5})

    def get_backtest_date_range(self) -> tuple[int, int]:
        """Get backtest date range as Unix timestamps (seconds).

        Returns:
            Tuple of (start_ts, end_ts).
        """
        start_date_str = self.get("backtest.start_date")
        end_date_str = self.get("backtest.end_date")

        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        else:
            days = self.get("backtest.backtest_days", 1)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        return int(start_date.timestamp()), int(end_date.timestamp())

    def get_target_user_id(self) -> int:
        """Get target user ID."""
        return self.get("news.target_user_id", 1)

    def get_max_news(self) -> int:
        """Get maximum news count."""
        return self.get("news.max_news", 50)

    def get_articles_path(self) -> str:
        """Get news articles file path."""
        path = os.getenv("NEWS_ARTICLES_PATH") or self.get(
            "news.articles_path", "examples/sample_data/articles.csv"
        )
        if not os.path.isabs(path):
            project_root = Path(__file__).parent.parent
            path = str(project_root / path)
        return path

    def get_recommendations_path(self) -> str:
        """Get recommendations file path."""
        path = os.getenv("NEWS_RECOMMENDATIONS_PATH") or self.get(
            "news.recommendations_path", "examples/sample_data/daily_recommendations.csv"
        )
        if not os.path.isabs(path):
            project_root = Path(__file__).parent.parent
            path = str(project_root / path)
        return path

    def get_finbert_model(self) -> str:
        """Get FinBERT model name."""
        return self.get("finbert.model_name", "ProsusAI/finbert")

    def get_finbert_batch_size(self) -> int:
        """Get FinBERT batch size."""
        return self.get("finbert.batch_size", 32)

    def get_finbert_device(self) -> str:
        """Get FinBERT device."""
        return self.get("finbert.device", "auto")

    def get_llm_provider(self) -> str:
        """Get LLM provider ('deepseek' or 'openai'). Env var takes precedence."""
        return (os.getenv("LLM_PROVIDER") or self.get("llm.provider", "deepseek")).strip()

    def get_llm_base_url(self) -> str:
        """Get LLM base URL. Env var takes precedence."""
        return (os.getenv("LLM_BASE_URL") or self.get("llm.base_url", "")).strip()

    def get_llm_model(self) -> str:
        """Get LLM model name. Env var takes precedence."""
        return (os.getenv("LLM_MODEL") or self.get("llm.model", "")).strip()

    def get_llm_api_key(self) -> str:
        """
        Get LLM API key.
        Priority:
          1) LLM_API_KEY
          2) provider-specific env vars (DEEPSEEK_API_KEY / OPENAI_API_KEY)
          3) config llm.api_key
        """
        key = (os.getenv("LLM_API_KEY") or "").strip()
        if key:
            return key

        provider = (os.getenv("LLM_PROVIDER") or self.get("llm.provider", "") or "").strip().lower()
        if provider == "deepseek":
            key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
            if key:
                return key
        if provider == "openai":
            key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if key:
                return key

        return (self.get("llm.api_key", "") or "").strip()

    def get_llm_temperature(self) -> float:
        """Get LLM temperature."""
        return float(self.get("llm.temperature", 0.3))

    def get_llm_max_tokens(self) -> int:
        """Get LLM max_tokens."""
        return int(self.get("llm.max_tokens", 500))

    def get_results_dir(self) -> str:
        """Get results directory path."""
        path = self.get("output.results_dir", "output/backtest_results")
        if not os.path.isabs(path):
            project_root = Path(__file__).parent.parent
            path = str(project_root / path)
        return path

    def get_log_level(self) -> str:
        """Get log level."""
        return self.get("output.log_level", "INFO")

    def get_backtest_config_dict(self) -> Dict[str, Any]:
        """Get complete backtest configuration dictionary."""
        return self._backtest_config.get("backtest", {})


_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
