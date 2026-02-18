"""
File: llm/groq_provider.py
Purpose: Groq LLM provider — real API calls via Groq's OpenAI-compatible API.
Dependencies: groq SDK (pip install groq).
Performance: ~500ms-2s per call (network-bound).

Implements the LLMProvider interface with Groq's ultra-fast inference.
Handles rate limits (429), server errors (503), timeouts, and
malformed responses with proper error classification.

API key is NEVER stored in code — read from environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from agents.log_agent.llm.classifier import LLMProvider, LLMProviderError
from agents.log_agent.telemetry import get_logger

logger = get_logger("log_agent.llm.groq_provider")


class GroqProvider(LLMProvider):
    """Groq LLM provider — fastest inference available.

    Uses Groq's OpenAI-compatible chat completions API.
    Default model: llama-3.3-70b-versatile (fast + capable).

    API key is read from:
        1. Constructor argument
        2. ``GROQ_API_KEY`` environment variable
        3. ``LOG_AGENT_LLM_API_KEY`` environment variable

    Args:
        api_key: Groq API key (prefer env var instead).
        model: Model ID (default: llama-3.3-70b-versatile).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Max output tokens.

    Example::

        # Set env var: GROQ_API_KEY=gsk_xxx
        provider = GroqProvider()
        result = provider.call(
            system_prompt="You are a classifier.",
            user_prompt="Classify these signals...",
        )

    Raises:
        LLMProviderError: On API failure with retryable flag set
            appropriately for circuit breaker compatibility.
    """

    # Groq pricing per 1M tokens (as of 2026)
    _PRICING: Dict[str, Dict[str, float]] = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("GROQ_API_KEY", "")
            or os.environ.get("LOG_AGENT_LLM_API_KEY", "")
        )
        if not self._api_key:
            raise LLMProviderError(
                "No Groq API key found. Set GROQ_API_KEY environment variable.",
                retryable=False,
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Lazy-import groq to avoid import error if not installed
        try:
            import groq
            self._client = groq.Groq(api_key=self._api_key)
        except ImportError:
            raise LLMProviderError(
                "groq package not installed. Run: pip install groq",
                retryable=False,
            )

        logger.info(
            f"GroqProvider initialized — model={model}, "
            f"temperature={temperature}, max_tokens={max_tokens}",
        )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a prompt to Groq and return parsed JSON response.

        Args:
            system_prompt: System role instruction.
            user_prompt: User message with signal data.
            timeout_seconds: Max wait time (Groq is usually <1s).

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            LLMProviderError: With retryable=True for 429/503,
                retryable=False for auth/validation errors.
        """
        import groq

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                timeout=timeout_seconds,
            )

            # Extract content
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError(
                    "Empty response from Groq", retryable=True
                )

            # Log usage
            usage = response.usage
            if usage:
                logger.debug(
                    f"Groq usage: {usage.prompt_tokens} in / "
                    f"{usage.completion_tokens} out / "
                    f"{usage.total_tokens} total",
                )

            # Parse JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Groq returned invalid JSON: {content[:200]}")
                raise LLMProviderError(
                    f"Invalid JSON from Groq: {e}",
                    retryable=True,
                )

            return parsed

        except groq.RateLimitError as e:
            logger.warning(f"Groq rate limit: {e}")
            raise LLMProviderError(
                f"Groq rate limit exceeded: {e}",
                retryable=True,
                status_code=429,
            )
        except groq.APIStatusError as e:
            status = getattr(e, "status_code", 500)
            retryable = status in (429, 500, 502, 503, 504)
            logger.error(f"Groq API error ({status}): {e}")
            raise LLMProviderError(
                f"Groq API error: {e}",
                retryable=retryable,
                status_code=status,
            )
        except groq.APITimeoutError as e:
            logger.warning(f"Groq timeout: {e}")
            raise LLMProviderError(
                f"Groq timeout after {timeout_seconds}s: {e}",
                retryable=True,
                status_code=408,
            )
        except groq.APIConnectionError as e:
            logger.error(f"Groq connection error: {e}")
            raise LLMProviderError(
                f"Groq connection error: {e}",
                retryable=True,
                status_code=503,
            )
        except LLMProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected Groq error: {e}", exc_info=True)
            raise LLMProviderError(
                f"Unexpected error: {e}",
                retryable=False,
            )

    def count_tokens(self, text: str) -> int:
        """Estimate token count (Groq uses LLaMA tokenizer ~4 chars/token).

        Args:
            text: Input text.

        Returns:
            Approximate token count.
        """
        return max(1, len(text) // 4)

    def get_cost_per_1k(self) -> Dict[str, float]:
        """Return cost per 1K tokens for the current model.

        Returns:
            Dict with 'input' and 'output' cost per 1K tokens.
        """
        pricing = self._PRICING.get(
            self._model, {"input": 0.59, "output": 0.79}
        )
        # Convert from per-1M to per-1K
        return {
            "input": pricing["input"] / 1000,
            "output": pricing["output"] / 1000,
        }
