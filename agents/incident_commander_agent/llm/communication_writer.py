"""
File: llm/communication_writer.py
Purpose: LLM-enhanced communication template generation.
Dependencies: groq_client, circuit_breaker, response_cache.
Performance: Cached path <1ms; LLM path <2s with fallback.

Generates polished incident communication messages via LLM.
Falls back to the deterministic communication_builder if
the LLM is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.incident_commander_agent.llm.circuit_breaker import CircuitBreaker
from agents.incident_commander_agent.llm.groq_client import (
    LLMResponse,
    create_provider,
)
from agents.incident_commander_agent.llm.response_cache import ResponseCache
from agents.incident_commander_agent.config import LLMConfig
from agents.incident_commander_agent.telemetry import (
    TelemetryCollector,
    get_logger,
)

logger = get_logger("incident_commander_agent.llm.communication_writer")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = _PROMPT_DIR / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


class CommunicationWriter:
    """LLM-enhanced communication writer.

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
        severity: str,
        affected_services: List[str],
        estimated_users: int,
    ) -> Optional[Dict[str, Any]]:
        """Generate LLM-enhanced communication messages.

        Args:
            root_cause: Root cause description.
            severity: Incident severity string.
            affected_services: List of affected services.
            estimated_users: Estimated impacted users.

        Returns:
            Dict with status_update, stakeholder_message, external_comms;
            or None on failure.
        """
        if not self._cb.can_execute():
            logger.warning("Circuit breaker OPEN — skipping LLM comm call")
            if self._telemetry:
                self._telemetry.fallback_triggers.inc()
            return None

        cache_key = ResponseCache.compute_key({
            "type": "communication",
            "root_cause": root_cause,
            "severity": severity,
            "services": sorted(affected_services),
            "users": estimated_users,
        })
        cached = self._cache.get(cache_key)
        if cached is not None:
            if self._telemetry:
                self._telemetry.cache_hits.inc()
            return cached

        if self._telemetry:
            self._telemetry.cache_misses.inc()

        system_prompt = _load_prompt("communication_writer.txt") or (
            "You are an expert incident communicator. Generate three "
            "messages: status_update (internal), stakeholder_message "
            "(executive), and external_comms (customer-facing). "
            "Respond only with valid JSON."
        )
        user_prompt = (
            f"Root cause: {root_cause}\n"
            f"Severity: {severity}\n"
            f"Affected services: {', '.join(affected_services)}\n"
            f"Estimated users impacted: {estimated_users:,}\n\n"
            "Generate communication templates as JSON with keys: "
            "status_update, stakeholder_message, external_comms."
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
            logger.warning(f"LLM communication call failed: {exc}")
            self._cb.record_failure()
            if self._telemetry:
                self._telemetry.llm_calls_failed.inc()
            return None
