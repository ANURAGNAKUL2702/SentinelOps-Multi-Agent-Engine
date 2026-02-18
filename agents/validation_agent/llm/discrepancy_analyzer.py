"""
File: llm/discrepancy_analyzer.py
Purpose: LLM-based analysis of verdict vs ground truth mismatches.
Dependencies: groq_client, circuit_breaker, response_cache.
Performance: <2s with LLM, <1ms on cache hit.

When the verdict is incorrect, uses LLM to analyze WHY the
agents got it wrong and generate improvement recommendations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from agents.validation_agent.config import ValidationAgentConfig
from agents.validation_agent.llm.circuit_breaker import CircuitBreaker
from agents.validation_agent.llm.groq_client import create_provider, LLMResponse
from agents.validation_agent.llm.response_cache import ResponseCache
from agents.validation_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("validation_agent.llm.discrepancy_analyzer")


class DiscrepancyAnalyzer:
    """LLM-based discrepancy analysis with circuit breaker and cache.

    Analyzes why the AI pipeline got the root cause wrong and
    produces improvement recommendations.

    Args:
        config: Validation agent configuration.
        telemetry: Telemetry collector.
    """

    def __init__(
        self,
        config: ValidationAgentConfig,
        telemetry: TelemetryCollector,
    ) -> None:
        self._config = config
        self._telemetry = telemetry
        self._provider = create_provider(config.llm)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=60.0,
            telemetry=telemetry,
        )
        self._cache = ResponseCache(ttl_seconds=300)
        self._system_prompt = self._load_prompt("discrepancy_analysis.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from the prompts directory.

        Args:
            filename: Prompt file name.

        Returns:
            Prompt text content.
        """
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "prompts"
        )
        filepath = os.path.join(prompts_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return (
                "You are a system reliability expert. Analyze why "
                "the AI root cause verdict differs from ground truth. "
                "Return JSON with 'recommendations' (list of strings) "
                "and 'analysis' (string explanation)."
            )

    def analyze(
        self,
        verdict_root_cause: str,
        actual_root_cause: str,
        accuracy_score: float,
        discrepancy_summary: str,
        correlation_id: str = "",
    ) -> Tuple[List[str], bool, bool]:
        """Analyze discrepancies between verdict and ground truth.

        Args:
            verdict_root_cause: What the AI predicted.
            actual_root_cause: Ground truth root cause.
            accuracy_score: Current accuracy score.
            discrepancy_summary: Summary of discrepancies found.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (recommendations, cache_hit, success).
        """
        # Build cache key
        cache_signals = {
            "verdict": verdict_root_cause,
            "actual": actual_root_cause,
            "accuracy": accuracy_score,
        }
        cache_key = self._cache.compute_key(cache_signals)

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._telemetry.record_cache_hit()
            recs = cached.get("recommendations", [])
            return recs, True, True

        self._telemetry.record_cache_miss()

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker OPEN — skipping LLM analysis",
                extra={"correlation_id": correlation_id},
            )
            return [], False, False

        # Build prompt
        user_prompt = (
            f"The AI predicted root cause: '{verdict_root_cause}'\n"
            f"Ground truth root cause: '{actual_root_cause}'\n"
            f"Accuracy score: {accuracy_score:.2f}\n"
            f"Discrepancies found: {discrepancy_summary}\n\n"
            f"Analyze why the AI pipeline got this wrong and "
            f"provide specific recommendations to improve accuracy."
        )

        try:
            response: LLMResponse = self._provider.call(
                self._system_prompt,
                user_prompt,
                self._telemetry,
            )
            self._circuit_breaker.record_success()

            # Parse response
            result = self._parse_response(response.content)
            recommendations = result.get("recommendations", [])

            # Cache result
            self._cache.put(cache_key, result)

            return recommendations, False, True

        except Exception as exc:
            self._circuit_breaker.record_failure()
            self._telemetry.record_llm_call(success=False)
            logger.error(
                f"LLM discrepancy analysis failed: {exc}",
                extra={"correlation_id": correlation_id},
            )
            return [], False, False

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response content as JSON.

        Args:
            content: Raw LLM response text.

        Returns:
            Parsed dict with 'recommendations' key.
        """
        try:
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: wrap raw text as recommendation
        return {
            "recommendations": [content[:500]] if content else [],
            "analysis": content,
        }
