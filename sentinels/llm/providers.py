"""
DeepSeek LLM provider implementation.
"""

from typing import Optional
import logging
from openai import OpenAI

from .base import BaseLLM

logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """DeepSeek API implementation using OpenAI-compatible SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        logger.info(f"Initialized DeepSeek LLM with model: {model}")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call DeepSeek API for completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


class OpenAILLM(BaseLLM):
    """OpenAI API implementation."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        logger.info(f"Initialized OpenAI LLM with model: {model}")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API for completion."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""

            # Debug: log if content is empty
            if not content:
                logger.error(f"Empty response from API. Model: {self.model}, Response: {response}")
                logger.error(
                    f"Choices: {response.choices if hasattr(response, 'choices') else 'N/A'}"
                )

            return content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


def create_llm(
    provider: str, api_key: str, base_url: str = "", model: str = "", **kwargs
) -> BaseLLM:
    """
    Factory function to create LLM instance.

    Args:
        provider: 'deepseek' or 'openai'
        api_key: API key
        base_url: API base URL (for DeepSeek)
        model: Model name
        **kwargs: Additional arguments

    Returns:
        BaseLLM instance
    """
    if provider == "deepseek":
        return DeepSeekLLM(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
            model=model or "deepseek-chat",
            **kwargs,
        )
    elif provider == "openai":
        return OpenAILLM(
            api_key=api_key, base_url=base_url or None, model=model or "gpt-4o-mini", **kwargs
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
