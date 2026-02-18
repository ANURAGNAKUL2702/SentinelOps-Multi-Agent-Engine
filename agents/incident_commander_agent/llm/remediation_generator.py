"""
File: llm/remediation_generator.py
Purpose: LLM-enhanced runbook refinement.
Dependencies: groq_client, circuit_breaker, response_cache.
Performance: Cached path <1ms; LLM path <2s with fallback.

Wraps the LLM provider with circuit breaker and cache.
If the LLM is unavailable, returns None so the caller
can fall back to the deterministic runbook generator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.incident_commander_agent.llm.circuit_breaker import CircuitBreaker
from agents.incident_commander_agent.llm.groq_client import (
    LLMResponse,
    MockProvider,
    GroqProvider,
    create_provider,
)
from agents.incident_commander_agent.llm.response_cache import ResponseCache
from agents.incident_commander_agent.config import LLMConfig
from agents.incident_commander_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)

logger = get_logger("incident_commander_agent.llm.remediation_generator")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = _PROMPT_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


class RemediationGenerator:
    """LLM-enhanced remediation generator.

    Args:
        config: LLM configuration.
        telemetry: Telemetry collector.
        circuit_breaker: Optional circuit breaker (shared instance).
        cache: Optional response cache (shared instance).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        telemetry: TelemetryCollector | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self._config = config or LLMConfig()
        self._telemetry = telemetry
        self._provider = create_provider(self._config)
        self._cb = circuit_breaker or CircuitBreaker(
            telemetry=telemetry,
        )
        self._cache = cache or ResponseCache()

    def generate(
        self,
        root_cause: str,
        affected_services: List[str],
        confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """Generate LLM-enhanced remediation suggestions.

        Args:
            root_cause: Root cause description.
            affected_services: List of affected services.
            confidence: Confidence score 0.0–1.0.

        Returns:
            Dict with LLM suggestions, or None on failure.
        """
        # Check circuit breaker
        if not self._cb.can_execute():
            logger.warning("Circuit breaker OPEN — skipping LLM call")
            if self._telemetry:
                self._telemetry.fallback_triggers.inc()
            return None

        # Check cache
        cache_key = ResponseCache.compute_key({
            "root_cause": root_cause,
            "services": sorted(affected_services),
            "confidence": round(confidence, 2),
        })
        cached = self._cache.get(cache_key)
        if cached is not None:
            if self._telemetry:
                self._telemetry.cache_hits.inc()
            return cached

        if self._telemetry:
            self._telemetry.cache_misses.inc()

        # Build prompt
        system_prompt = _load_prompt("runbook_generator.txt") or (
            "You are an expert SRE generating remediation runbook steps. "
            "Respond only with valid JSON."
        )
        user_prompt = (
            f"Root cause: {root_cause}\n"
            f"Affected services: {', '.join(affected_services)}\n"
            f"Confidence: {confidence:.0%}\n\n"
            "Generate additional remediation steps as a JSON array of objects "
            "with keys: description, command, expected_outcome."
        )

        try:
            response: LLMResponse = self._provider.call(
                system_prompt, user_prompt, self._telemetry,
            )
            self._cb.record_success()

            parsed = json.loads(response.content)
            self._cache.put(cache_key, parsed)
            return parsed

        except Exception as exc:
            logger.warning(f"LLM remediation call failed: {exc}")
            self._cb.record_failure()
            if self._telemetry:
                self._telemetry.llm_calls_failed.inc()
            return None
