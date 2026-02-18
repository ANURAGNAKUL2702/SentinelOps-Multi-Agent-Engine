"""
File: llm/theory_generator.py
Purpose: Algorithm 3 — LLM Hypothesis Generation with circuit breaker + caching.
Dependencies: Standard library + schema models.
Performance: <2s per LLM call (timeout enforced), <1ms cache hit.

Uses PRE-CALCULATED evidence and pattern signals — LLM generates theories,
never re-scans raw data. Circuit breaker prevents cascade on provider outages.

Adapted from dependency_agent/llm/classifier.py — same infrastructure,
hypothesis-domain prompts and parsing.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.hypothesis_agent.config import HypothesisAgentConfig, LLMConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CausalChain,
    CausalChainLink,
    CausalRelationship,
    Hypothesis,
    HypothesisStatus,
    IncidentCategory,
    PatternMatch,
    Severity,
)
from agents.hypothesis_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("hypothesis_agent.llm.theory_generator")


# ═══════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for LLM calls.

    State transitions::

        CLOSED --[N failures]--> OPEN --[cooldown]--> HALF_OPEN
        HALF_OPEN --[M successes]--> CLOSED
        HALF_OPEN --[failure]--> OPEN

    Args:
        failure_threshold: Consecutive failures to open circuit.
        cooldown_seconds: Seconds before half-open attempt.
        success_threshold: Consecutive successes to close circuit.
        telemetry: Optional telemetry collector.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        success_threshold: int = 2,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._success_threshold = success_threshold
        self._telemetry = telemetry
        self._lock = threading.Lock()

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (evaluates half-open on read)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
            return self._state

    def can_execute(self) -> bool:
        """Check if a call is allowed."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful LLM call."""
        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._success_threshold:
                    self._state = CircuitState.CLOSED
            else:
                self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed LLM call."""
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                if self._telemetry:
                    self._telemetry.circuit_breaker_trips.inc()
            elif self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                if self._telemetry:
                    self._telemetry.circuit_breaker_trips.inc()

    def reset(self) -> None:
        """Reset to CLOSED state (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_failure_time = 0.0


# ═══════════════════════════════════════════════════════════════
#  RESPONSE CACHE
# ═══════════════════════════════════════════════════════════════


class ResponseCache:
    """In-memory TTL cache for LLM hypothesis responses.

    Args:
        ttl_seconds: Time-to-live for cache entries.
        max_entries: Maximum cache size.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: Dict[str, Tuple[float, List[Hypothesis]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[Hypothesis]]:
        """Retrieve cached hypotheses if exists and not expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._store[key]
                return None
            return result

    def put(self, key: str, result: List[Hypothesis]) -> None:
        """Store hypotheses in the cache."""
        with self._lock:
            if (
                len(self._store) >= self._max_entries
                and key not in self._store
            ):
                oldest_key = min(
                    self._store, key=lambda k: self._store[k][0]
                )
                del self._store[oldest_key]
            self._store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def compute_key(signals: Dict[str, Any]) -> str:
        """Compute a deterministic cache key from signals."""
        payload = json.dumps(signals, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
#  LLM PROVIDER INTERFACE
# ═══════════════════════════════════════════════════════════════


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class LLMProvider:
    """Abstract base for LLM provider adapters."""

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        """Send a prompt and return parsed JSON."""
        raise NotImplementedError("Subclass must implement call()")

    def count_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars/token)."""
        return max(1, len(text) // 4)


class MockLLMProvider(LLMProvider):
    """Mock provider for testing — returns deterministic hypotheses.

    Args:
        should_fail: If True, always raises LLMProviderError.
        failure_count: Fail first N calls then succeed.
    """

    def __init__(
        self,
        should_fail: bool = False,
        failure_count: int = 0,
    ) -> None:
        self._should_fail = should_fail
        self._failure_count = failure_count
        self._call_count = 0

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        self._call_count += 1
        if self._should_fail:
            raise LLMProviderError(
                "Mock failure", retryable=True, status_code=503
            )
        if self._call_count <= self._failure_count:
            raise LLMProviderError(
                f"Mock transient failure "
                f"({self._call_count}/{self._failure_count})",
                retryable=True,
                status_code=429,
            )
        return {
            "hypotheses": [
                {
                    "theory": (
                        "Database connection pool exhaustion caused "
                        "cascading timeouts across dependent services"
                    ),
                    "category": "database",
                    "severity": "critical",
                    "evidence_supporting": [
                        "Database errors in logs",
                        "Connection timeout keywords",
                    ],
                    "evidence_contradicting": [],
                    "causal_chain": [
                        {
                            "step": 1,
                            "service": "database",
                            "event": "Connection pool saturated",
                            "relationship": "causes",
                        },
                        {
                            "step": 2,
                            "service": "api-gateway",
                            "event": "Requests timing out",
                            "relationship": "causes",
                        },
                    ],
                    "reasoning": (
                        "Evidence strongly suggests DB pool exhaustion"
                    ),
                },
                {
                    "theory": (
                        "Memory leak in application service causing "
                        "gradual performance degradation"
                    ),
                    "category": "application",
                    "severity": "high",
                    "evidence_supporting": [
                        "Increasing memory metrics",
                    ],
                    "evidence_contradicting": [
                        "No OOM events observed",
                    ],
                    "causal_chain": [
                        {
                            "step": 1,
                            "service": "app-service",
                            "event": "Memory growing over time",
                            "relationship": "causes",
                        },
                    ],
                    "reasoning": (
                        "Memory patterns suggest a slow leak"
                    ),
                },
                {
                    "theory": (
                        "Network configuration change caused "
                        "connectivity issues between services"
                    ),
                    "category": "network",
                    "severity": "medium",
                    "evidence_supporting": [
                        "Connection errors detected",
                    ],
                    "evidence_contradicting": [],
                    "causal_chain": [],
                    "reasoning": (
                        "Network errors present but limited evidence"
                    ),
                },
            ],
            "confidence": 0.75,
            "source": "llm",
        }


# ═══════════════════════════════════════════════════════════════
#  THEORY GENERATOR
# ═══════════════════════════════════════════════════════════════


class TheoryGenerator:
    """LLM-powered hypothesis generation with circuit breaker and caching.

    Pipeline::

        signals → cache check → circuit check → LLM call → parse → cache store

    Args:
        config: Agent configuration.
        provider: LLM provider adapter.
        telemetry: Telemetry collector instance.
    """

    def __init__(
        self,
        config: Optional[HypothesisAgentConfig] = None,
        provider: Optional[LLMProvider] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or HypothesisAgentConfig()
        self._llm_config = self._config.llm
        self._provider = provider or MockLLMProvider()
        self._telemetry = telemetry or TelemetryCollector()

        self._circuit = CircuitBreaker(
            failure_threshold=self._llm_config.circuit_failure_threshold,
            cooldown_seconds=self._llm_config.circuit_cooldown_seconds,
            success_threshold=self._llm_config.circuit_success_threshold,
            telemetry=self._telemetry,
        )

        self._cache = ResponseCache(
            ttl_seconds=self._llm_config.cache_ttl_seconds,
        )

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    def generate(
        self,
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
        correlation_id: str = "",
    ) -> List[Hypothesis]:
        """Generate hypotheses using LLM with safeguards.

        Args:
            evidence: Aggregated evidence from all agents.
            pattern_matches: Matched known patterns.
            correlation_id: Request correlation ID.

        Returns:
            List of generated Hypothesis objects.

        Raises:
            LLMProviderError: If LLM fails after retries.
        """
        start = time.perf_counter()

        # Build signals for cache key and prompt
        signals = self._build_signals(evidence, pattern_matches)

        # ── 1. cache check ──────────────────────────────────────
        cache_key = ""
        if self._config.features.enable_caching:
            cache_key = ResponseCache.compute_key(signals)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._telemetry.record_cache_hit()
                return cached
            self._telemetry.record_cache_miss()

        # ── 2. circuit breaker ──────────────────────────────────
        if not self._circuit.can_execute():
            raise LLMProviderError(
                "Circuit breaker is OPEN", retryable=False
            )

        # ── 3. build prompt ─────────────────────────────────────
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(signals)
        input_tokens = self._provider.count_tokens(
            system_prompt + user_prompt
        )

        # ── 4. call with retry ──────────────────────────────────
        response = self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            correlation_id=correlation_id,
        )

        output_tokens = self._provider.count_tokens(
            json.dumps(response, default=str)
        )

        # ── 5. telemetry ────────────────────────────────────────
        self._telemetry.record_llm_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_per_1k_input=self._llm_config.cost_per_1k_input_tokens,
            cost_per_1k_output=self._llm_config.cost_per_1k_output_tokens,
            success=True,
            correlation_id=correlation_id,
        )
        self._circuit.record_success()

        # ── 6. parse response ───────────────────────────────────
        hypotheses = self._parse_response(response)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # ── 7. cache ────────────────────────────────────────────
        if self._config.features.enable_caching and cache_key:
            self._cache.put(cache_key, hypotheses)

        logger.info(
            f"LLM hypothesis generation completed — "
            f"{len(hypotheses)} hypotheses in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "llm",
                "context": {
                    "hypothesis_count": len(hypotheses),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return hypotheses

    # ── retry logic ─────────────────────────────────────────────

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        correlation_id: str = "",
    ) -> Dict[str, Any]:
        """Call LLM with exponential backoff retry."""
        max_retries = self._llm_config.max_retries
        base_delay = self._llm_config.retry_base_delay
        last_error: Optional[LLMProviderError] = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout_seconds=self._llm_config.timeout_seconds,
                )
            except LLMProviderError as e:
                last_error = e
                self._telemetry.record_llm_call(
                    success=False, correlation_id=correlation_id
                )

                if not e.retryable or attempt == max_retries:
                    self._circuit.record_failure()
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)

        self._circuit.record_failure()
        raise last_error or LLMProviderError("Unknown LLM failure")

    # ── signal building ─────────────────────────────────────────

    @staticmethod
    def _build_signals(
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
    ) -> Dict[str, Any]:
        """Build signal dict for prompt and cache key."""
        return {
            "evidence_count": evidence.total_evidence_count,
            "strong_evidence": evidence.strong_evidence_count,
            "sources": [
                s.value for s in evidence.sources_represented
            ],
            "dominant_severity": evidence.dominant_severity.value,
            "evidence_summaries": [
                {
                    "source": e.source.value,
                    "description": e.description,
                    "severity": e.severity.value,
                    "strength": e.strength.value,
                }
                for e in evidence.evidence_items[:20]  # limit
            ],
            "correlations": [
                {
                    "sources": [s.value for s in c.sources],
                    "description": c.description,
                    "score": c.correlation_score,
                }
                for c in evidence.correlations
            ],
            "pattern_matches": [
                {
                    "pattern": pm.pattern_name.value,
                    "score": pm.match_score,
                    "category": pm.category.value,
                }
                for pm in pattern_matches
            ],
        }

    # ── prompt building ─────────────────────────────────────────

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt for hypothesis generation."""
        return (
            "You are an expert incident root cause analyst. "
            "You receive pre-computed evidence signals from log, metric, "
            "and dependency agents. Your job is to generate 3-5 root cause "
            "hypotheses explaining the observed incident.\n\n"
            "For each hypothesis:\n"
            "- Provide a clear theory statement\n"
            "- Classify the category (database, application, network, "
            "deployment, configuration, infrastructure, security)\n"
            "- Assess severity (critical, high, medium, low)\n"
            "- List supporting and contradicting evidence\n"
            "- Build a causal chain showing propagation\n"
            "- Explain your reasoning\n\n"
            "Respond with ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "hypotheses": [\n'
            "    {\n"
            '      "theory": "<clear theory statement>",\n'
            '      "category": "<category>",\n'
            '      "severity": "<severity>",\n'
            '      "evidence_supporting": ["<evidence 1>", ...],\n'
            '      "evidence_contradicting": ["<evidence>", ...],\n'
            '      "causal_chain": [\n'
            "        {\n"
            '          "step": 1,\n'
            '          "service": "<service>",\n'
            '          "event": "<what happened>",\n'
            '          "relationship": "causes|contributes_to|correlates_with"\n'
            "        }\n"
            "      ],\n"
            '      "reasoning": "<why this hypothesis>"\n'
            "    }\n"
            "  ],\n"
            '  "confidence": <float 0.0-1.0>\n'
            "}\n\n"
            "Generate EXACTLY 3-5 hypotheses ranked by likelihood. "
            "Do NOT re-analyze raw data. Only reason from the signals."
        )

    @staticmethod
    def _build_user_prompt(signals: Dict[str, Any]) -> str:
        """Build the user prompt with pre-calculated signals."""
        return (
            "Generate root cause hypotheses from these signals:\n\n"
            f"SIGNALS:\n{json.dumps(signals, indent=2, default=str)}\n\n"
            "Respond with JSON: {hypotheses, confidence}"
        )

    # ── response parsing ────────────────────────────────────────

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> List[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses: List[Hypothesis] = []

        try:
            raw_list = response.get("hypotheses", [])
            if not isinstance(raw_list, list):
                raw_list = []

            for raw in raw_list:
                if not isinstance(raw, dict):
                    continue

                # Parse causal chain
                causal_chain = None
                chain_raw = raw.get("causal_chain", [])
                if isinstance(chain_raw, list) and chain_raw:
                    links: List[CausalChainLink] = []
                    for link in chain_raw:
                        if isinstance(link, dict):
                            rel_str = link.get(
                                "relationship", "causes"
                            )
                            try:
                                rel = CausalRelationship(rel_str)
                            except ValueError:
                                rel = CausalRelationship.CAUSES

                            links.append(CausalChainLink(
                                step=int(link.get("step", len(links) + 1)),
                                service=str(
                                    link.get("service", "unknown")
                                ),
                                event=str(
                                    link.get("event", "unknown")
                                ),
                                relationship=rel,
                            ))

                    if links:
                        causal_chain = CausalChain(
                            chain=links,
                            root_cause_service=(
                                links[0].service if links else ""
                            ),
                            terminal_effect=(
                                links[-1].event if links else ""
                            ),
                            chain_confidence=0.6,
                        )

                # Parse category
                cat_str = raw.get("category", "unknown")
                try:
                    category = IncidentCategory(cat_str)
                except ValueError:
                    category = IncidentCategory.UNKNOWN

                # Parse severity
                sev_str = raw.get("severity", "medium")
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.MEDIUM

                hypotheses.append(Hypothesis(
                    theory=str(raw.get("theory", "Unknown theory")),
                    category=category,
                    severity=severity,
                    evidence_supporting=raw.get(
                        "evidence_supporting", []
                    ),
                    evidence_contradicting=raw.get(
                        "evidence_contradicting", []
                    ),
                    causal_chain=causal_chain,
                    reasoning=str(raw.get("reasoning", "")),
                    status=HypothesisStatus.ACTIVE,
                ))

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM response: {e}",
                extra={"layer": "llm"},
            )

        # Ensure minimum hypotheses
        if not hypotheses:
            hypotheses.append(Hypothesis(
                theory="Undetermined root cause — insufficient LLM data",
                category=IncidentCategory.UNKNOWN,
                severity=Severity.MEDIUM,
                reasoning="LLM response did not contain valid hypotheses",
                status=HypothesisStatus.ACTIVE,
            ))

        return hypotheses
