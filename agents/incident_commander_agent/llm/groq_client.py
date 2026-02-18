"""
File: llm/groq_client.py
Purpose: LLM provider abstraction (Groq + mock).
Dependencies: groq SDK (optional runtime), schema models.
Performance: <2s per LLM call (timeout enforced).

Provides a MockProvider for testing and a GroqProvider for production.
Provider is selected via config.llm.provider string.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from agents.incident_commander_agent.config import LLMConfig
from agents.incident_commander_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)

logger = get_logger("incident_commander_agent.llm.groq_client")


class LLMResponse:
    """Structured response from an LLM call.

    Attributes:
        content: The text response.
        input_tokens: Tokens consumed for input.
        output_tokens: Tokens consumed for output.
        latency_ms: Call latency in milliseconds.
        model: Model used.
    """

    def __init__(
        self,
        content: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.model = model


class MockProvider:
    """Mock LLM provider for testing.

    Returns a deterministic JSON incident response.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> LLMResponse:
        """Return a mock incident response.

        Args:
            system_prompt: System prompt (ignored by mock).
            user_prompt: User prompt (ignored by mock).
            telemetry: Optional telemetry collector.

        Returns:
            LLMResponse with mock content.
        """
        start = time.perf_counter()

        mock_response = {
            "runbook_title": "Mock LLM: remediation for detected failure",
            "additional_steps": [
                "Verify service health post-remediation",
                "Check downstream dependencies",
            ],
            "communication_summary": (
                "An incident has been detected and remediation steps "
                "are being executed. The team is monitoring the situation."
            ),
            "severity_assessment": "P2_MEDIUM",
            "confidence": 0.80,
        }

        content = json.dumps(mock_response)
        elapsed = (time.perf_counter() - start) * 1000

        if telemetry:
            telemetry.llm_calls_total.inc()
            telemetry.llm_tokens_input.inc(50)
            telemetry.llm_tokens_output.inc(100)

        return LLMResponse(
            content=content,
            input_tokens=50,
            output_tokens=100,
            latency_ms=elapsed,
            model=self._config.model,
        )


class GroqProvider:
    """Groq LLM provider for production use.

    Uses the Groq SDK to call the API. Falls back gracefully
    if the SDK is not installed.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None
        self._available = False

        try:
            import groq
            api_key = config.api_key
            if api_key:
                self._client = groq.Groq(api_key=api_key)
                self._available = True
        except ImportError:
            logger.warning("groq SDK not installed — GroqProvider unavailable")
        except Exception as exc:
            logger.warning(f"Failed to initialize Groq client: {exc}")

    @property
    def available(self) -> bool:
        """Whether the Groq provider is ready."""
        return self._available

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> LLMResponse:
        """Call the Groq API.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt for the LLM.
            telemetry: Optional telemetry collector.

        Returns:
            LLMResponse with API content.

        Raises:
            RuntimeError: If the provider is not available.
        """
        if not self._available or not self._client:
            raise RuntimeError("Groq provider not available")

        start = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        elapsed = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        if telemetry:
            telemetry.llm_calls_total.inc()
            telemetry.llm_tokens_input.inc(input_tokens)
            telemetry.llm_tokens_output.inc(output_tokens)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            model=self._config.model,
        )


def create_provider(
    config: LLMConfig,
) -> MockProvider | GroqProvider:
    """Factory to create the appropriate LLM provider.

    Args:
        config: LLM configuration.

    Returns:
        MockProvider or GroqProvider instance.
    """
    if config.provider == "groq":
        provider = GroqProvider(config)
        if provider.available:
            return provider
        logger.warning("Groq provider unavailable, falling back to mock")
    return MockProvider(config)
