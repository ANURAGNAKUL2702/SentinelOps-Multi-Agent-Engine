"""
File: llm/verdict_generator.py
Purpose: LLM-powered verdict synthesis with circuit breaker + caching.
Dependencies: circuit_breaker, response_cache, groq_client, schema.
Performance: <2s per LLM call, <1ms cache hit.

Orchestrates the LLM call pipeline:
1. Check cache → return if hit.
2. Check circuit breaker → abort if OPEN.
3. Build prompt from synthesis result.
4. Call LLM provider.
5. Parse JSON response.
6. Cache and return.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.llm.circuit_breaker import CircuitBreaker
from agents.root_cause_agent.llm.groq_client import (
    LLMResponse,
    create_provider,
)
from agents.root_cause_agent.llm.response_cache import ResponseCache
from agents.root_cause_agent.schema import SynthesisResult
from agents.root_cause_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("root_cause_agent.llm.verdict_generator")

_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts"
)


def _load_prompt(filename: str) -> str:
    """Load a prompt template from prompts/ directory."""
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {path}")
        return ""


class VerdictGenerator:
    """LLM-powered root cause verdict synthesis.

    Pipeline::

        SynthesisResult → cache check → circuit check → prompt build
        → LLM call → parse JSON → cache store → return verdict dict

    Args:
        config: Agent configuration.
        telemetry: Telemetry collector.
    """

    def __init__(
        self,
        config: Optional[RootCauseAgentConfig] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or RootCauseAgentConfig()
        self._telemetry = telemetry

        self._provider = create_provider(self._config.llm)
        self._circuit = CircuitBreaker(
            failure_threshold=self._config.llm.circuit_failure_threshold,
            cooldown_seconds=self._config.llm.circuit_cooldown_seconds,
            success_threshold=self._config.llm.circuit_success_threshold,
            telemetry=telemetry,
        )
        self._cache = ResponseCache(
            ttl_seconds=self._config.llm.cache_ttl_seconds,
        )

        self._system_prompt = _load_prompt("verdict_synthesizer.txt")
        self._few_shot = _load_prompt("few_shot_examples.txt")

    def generate(
        self,
        synthesis: SynthesisResult,
        correlation_id: str = "",
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """Generate verdict via LLM.

        Args:
            synthesis: Synthesis result with evidence.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (verdict_dict, used_cache, success).
        """
        start = time.perf_counter()

        # ── Cache check ─────────────────────────────────────────
        cache_key = self._build_cache_key(synthesis)
        if self._config.features.enable_caching:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if self._telemetry:
                    self._telemetry.cache_hits.inc()
                logger.info(
                    "Cache hit for verdict generation",
                    extra={"correlation_id": correlation_id},
                )
                return (cached, True, True)
            if self._telemetry:
                self._telemetry.cache_misses.inc()

        # ── Circuit breaker check ───────────────────────────────
        if not self._circuit.can_execute():
            logger.warning(
                "Circuit breaker OPEN — skipping LLM call",
                extra={"correlation_id": correlation_id},
            )
            return ({}, False, False)

        # ── Build prompt ────────────────────────────────────────
        user_prompt = self._build_user_prompt(synthesis)

        # ── LLM call with retry ─────────────────────────────────
        max_retries = self._config.llm.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self._provider.call(
                    system_prompt=self._system_prompt,
                    user_prompt=user_prompt,
                    telemetry=self._telemetry,
                )
                verdict = self._parse_response(response)
                self._circuit.record_success()

                # Cache the result
                if self._config.features.enable_caching:
                    self._cache.put(cache_key, verdict)

                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    f"LLM verdict generated in {elapsed:.1f}ms "
                    f"(attempt {attempt + 1})",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "verdict_generation",
                    },
                )

                return (verdict, False, True)

            except Exception as exc:
                last_error = exc
                self._circuit.record_failure()
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}): {exc}",
                    extra={"correlation_id": correlation_id},
                )

                if not self._circuit.can_execute():
                    break

        logger.error(
            f"LLM verdict generation failed after {max_retries} attempts: "
            f"{last_error}",
            extra={"correlation_id": correlation_id},
        )
        return ({}, False, False)

    def _build_user_prompt(self, synthesis: SynthesisResult) -> str:
        """Build user prompt from synthesis result.

        Args:
            synthesis: Synthesis result.

        Returns:
            Formatted user prompt string.
        """
        evidence_summary = []
        for ev in synthesis.evidence_trail[:20]:  # limit context
            evidence_summary.append({
                "source": ev.source.value,
                "type": ev.evidence_type.value,
                "description": ev.description,
                "confidence": ev.confidence,
            })

        data = {
            "evidence_count": len(synthesis.evidence_trail),
            "sources": [s.value for s in synthesis.sources_present],
            "agreement_score": synthesis.agreement_score,
            "primary_service": synthesis.primary_service,
            "evidence_sample": evidence_summary,
        }

        few_shot = ""
        if self._few_shot:
            few_shot = f"\n\n## Examples\n{self._few_shot}\n\n"

        prompt = (
            f"## Evidence Synthesis\n"
            f"```json\n{json.dumps(data, indent=2)}\n```\n"
            f"{few_shot}"
            f"Based on the above evidence, provide a root cause verdict "
            f"as a JSON object with these fields:\n"
            f"- root_cause: string (the identified root cause)\n"
            f"- confidence: float 0.0-1.0\n"
            f"- reasoning: string (detailed reasoning)\n"
            f"- category: string (infrastructure|application|database|"
            f"network|configuration|deployment|security|unknown)\n"
            f"- severity: string (critical|high|medium|low)\n"
            f"- estimated_mttr_minutes: int\n"
        )

        return prompt

    def _parse_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse LLM response into verdict dict.

        Args:
            response: LLM response.

        Returns:
            Parsed verdict dict.

        Raises:
            ValueError: If response cannot be parsed.
        """
        content = response.content.strip()

        # Try to extract JSON from markdown code block
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            content = content[start:end].strip()

        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response: {exc}") from exc

    def _build_cache_key(self, synthesis: SynthesisResult) -> str:
        """Build cache key from synthesis result.

        Args:
            synthesis: Synthesis result.

        Returns:
            SHA-256 cache key.
        """
        signals = {
            "evidence_count": len(synthesis.evidence_trail),
            "sources": [s.value for s in synthesis.sources_present],
            "agreement": synthesis.agreement_score,
            "primary_service": synthesis.primary_service,
        }
        return ResponseCache.compute_key(signals)

    @property
    def circuit_state(self) -> str:
        """Current circuit breaker state."""
        return self._circuit.state.value

    @property
    def cache_size(self) -> int:
        """Current cache size."""
        return self._cache.size
