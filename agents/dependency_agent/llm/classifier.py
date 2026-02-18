"""
File: llm/classifier.py
Purpose: LLM wrapper for dependency classification with circuit breaker + caching.
Dependencies: Standard library only (provider adapters pluggable).
Performance: <10s per call (timeout enforced), <1ms cache hit.

Uses PRE-CALCULATED signals from graph/impact/trace analysis — LLM classifies,
never re-scans raw data.  Circuit breaker prevents cascade on provider outages.

Adapted from metrics_agent/llm/classifier.py — same infrastructure,
dependency-domain prompts and parsing.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.dependency_agent.config import DependencyAgentConfig, LLMConfig
from agents.dependency_agent.schema import (
    CascadePattern,
    CascadingFailureRisk,
    ClassificationResult,
    DependencyAnalysisInput,
    DependencyAnalysisSummary,
    FailedServiceInfo,
    ImpactAnalysisResult,
    Severity,
    SinglePointOfFailure,
)
from agents.dependency_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("dependency_agent.llm.classifier")


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
                    logger.info(
                        f"Circuit breaker → HALF_OPEN "
                        f"(cooldown {elapsed:.1f}s elapsed)"
                    )
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
                    logger.info("Circuit breaker → CLOSED (recovered)")
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
                logger.warning(
                    "Circuit breaker → OPEN (half-open probe failed)"
                )
                if self._telemetry:
                    self._telemetry.circuit_breaker_trips.inc()
            elif self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                logger.warning(
                    f"Circuit breaker → OPEN "
                    f"({self._consecutive_failures} consecutive failures)"
                )
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
#  RESPONSE CACHE (TTL-based, thread-safe)
# ═══════════════════════════════════════════════════════════════


class ResponseCache:
    """In-memory TTL cache for LLM classification responses.

    Keys are SHA-256 hashes of the input signals.

    Args:
        ttl_seconds: Time-to-live for cache entries.
        max_entries: Maximum cache size (LRU eviction).
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: Dict[str, Tuple[float, ClassificationResult]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ClassificationResult]:
        """Retrieve a cached result if exists and not expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._store[key]
                return None
            return result

    def put(self, key: str, result: ClassificationResult) -> None:
        """Store a result in the cache."""
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
        """Compute a deterministic cache key from analysis signals.

        Args:
            signals: Dict of analysis signals to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
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
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a prompt to the LLM and return parsed JSON.

        Args:
            system_prompt: System role prompt.
            user_prompt: User message.
            timeout_seconds: Max wait time.

        Returns:
            Parsed JSON dict.

        Raises:
            LLMProviderError: On failure.
        """
        raise NotImplementedError("Subclass must implement call()")

    def count_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars/token)."""
        return max(1, len(text) // 4)


class MockLLMProvider(LLMProvider):
    """Mock provider for testing — returns deterministic results.

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
        timeout_seconds: float = 10.0,
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
        return {"classification_source": "llm", "mock": True}


# ═══════════════════════════════════════════════════════════════
#  LLM CLASSIFIER
# ═══════════════════════════════════════════════════════════════


class LLMClassifier:
    """LLM-powered classification with circuit breaker, retry, and caching.

    Pipeline::

        signals → cache check → circuit check → LLM call → validate → cache store

    Args:
        config: Agent configuration.
        provider: LLM provider adapter.
        telemetry: Telemetry collector instance.
    """

    def __init__(
        self,
        config: Optional[DependencyAgentConfig] = None,
        provider: Optional[LLMProvider] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or DependencyAgentConfig()
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

    def classify(
        self,
        signals: Dict[str, Any],
        input_data: DependencyAnalysisInput,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Classify dependency analysis using LLM with safeguards.

        Args:
            signals: Pre-calculated analysis signals.
            input_data: Original input.
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult with source="llm" or "cached".

        Raises:
            LLMProviderError: If LLM fails after all retries.
        """
        start = time.perf_counter()

        # ── 1. cache check ──────────────────────────────────────
        cache_key = ""
        if self._config.features.enable_caching:
            cache_key = ResponseCache.compute_key(signals)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._telemetry.record_cache_hit()
                elapsed = (time.perf_counter() - start) * 1000
                return ClassificationResult(
                    failed_service=cached.failed_service,
                    dependency_analysis=cached.dependency_analysis,
                    impact_analysis=cached.impact_analysis,
                    critical_path=cached.critical_path,
                    bottlenecks=cached.bottlenecks,
                    cascading_failure_risk=cached.cascading_failure_risk,
                    single_points_of_failure=cached.single_points_of_failure,
                    confidence_score=cached.confidence_score,
                    confidence_reasoning=cached.confidence_reasoning,
                    classification_source="cached",
                    classification_latency_ms=round(elapsed, 2),
                )
            self._telemetry.record_cache_miss()

        # ── 2. circuit breaker ──────────────────────────────────
        if not self._circuit.can_execute():
            raise LLMProviderError(
                "Circuit breaker is OPEN", retryable=False
            )

        # ── 3. build prompt ─────────────────────────────────────
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(signals, input_data)
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
        result = self._parse_response(response, input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result_with_latency = ClassificationResult(
            failed_service=result.failed_service,
            dependency_analysis=result.dependency_analysis,
            impact_analysis=result.impact_analysis,
            critical_path=result.critical_path,
            bottlenecks=result.bottlenecks,
            cascading_failure_risk=result.cascading_failure_risk,
            single_points_of_failure=result.single_points_of_failure,
            confidence_score=result.confidence_score,
            confidence_reasoning=result.confidence_reasoning,
            classification_source="llm",
            classification_latency_ms=round(elapsed_ms, 2),
        )

        # ── 7. cache ────────────────────────────────────────────
        if self._config.features.enable_caching and cache_key:
            self._cache.put(cache_key, result_with_latency)

        logger.info(
            f"LLM classification completed in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "llm",
                "context": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return result_with_latency

    # ── retry logic ─────────────────────────────────────────────

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        correlation_id: str = "",
    ) -> Dict[str, Any]:
        """Call LLM with exponential backoff retry.

        Args:
            system_prompt: System role prompt.
            user_prompt: User message.
            correlation_id: For logging.

        Returns:
            Parsed JSON response dict.

        Raises:
            LLMProviderError: After all retries exhausted.
        """
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
                    logger.error(
                        f"LLM call failed "
                        f"(attempt {attempt}/{max_retries}): {e}",
                        extra={
                            "correlation_id": correlation_id,
                            "layer": "llm",
                            "context": {
                                "attempt": attempt,
                                "retryable": e.retryable,
                                "status_code": e.status_code,
                            },
                        },
                    )
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"LLM call failed "
                    f"(attempt {attempt}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "layer": "llm",
                    },
                )
                time.sleep(delay)

        self._circuit.record_failure()
        raise last_error or LLMProviderError("Unknown LLM failure")

    # ── prompt building ─────────────────────────────────────────

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt for dependency classification."""
        return (
            "You are a production dependency graph incident classifier. "
            "You receive pre-computed dependency signals (blast radius, "
            "criticality scores, bottlenecks, cycle detection) from "
            "deterministic analysis. Your job is to identify cascading "
            "failure risks and classify severity.\n\n"
            "Respond with ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "summary": {\n'
            '    "total_services": <int>,\n'
            '    "total_dependencies": <int>,\n'
            '    "has_cycles": <bool>,\n'
            '    "cascade_risk_count": <int>,\n'
            '    "spof_count": <int>,\n'
            '    "bottleneck_count": <int>,\n'
            '    "critical_path": "<string or null>"\n'
            "  },\n"
            '  "cascading_failure_risks": [...],\n'
            '  "single_points_of_failure": [...],\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}\n\n"
            "Do NOT re-analyze raw graphs. Only classify based on signals."
        )

    @staticmethod
    def _build_user_prompt(
        signals: Dict[str, Any],
        input_data: DependencyAnalysisInput,
    ) -> str:
        """Build the user prompt with pre-calculated signals.

        Args:
            signals: Pre-calculated analysis signals.
            input_data: Original input for context.

        Returns:
            Formatted prompt string.
        """
        context = {
            "total_services": len(input_data.service_graph.nodes),
            "total_edges": len(input_data.service_graph.edges),
            "failures": [
                f.service_name
                for f in (input_data.current_failures or [])
            ],
        }
        return (
            "Classify the following pre-computed dependency signals:\n\n"
            f"SIGNALS:\n{json.dumps(signals, indent=2, default=str)}\n\n"
            f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            "Respond with JSON: {summary, cascading_failure_risks, "
            "single_points_of_failure, confidence}"
        )

    # ── response parsing ────────────────────────────────────────

    def _parse_response(
        self,
        response: Dict[str, Any],
        input_data: DependencyAnalysisInput,
    ) -> ClassificationResult:
        """Parse LLM response into ClassificationResult.

        Falls back to minimal result if parsing fails.

        Args:
            response: Parsed JSON from LLM.
            input_data: Original input.

        Returns:
            Validated ClassificationResult.
        """
        try:
            # Parse summary
            summary_raw = response.get("summary", {})
            if not isinstance(summary_raw, dict):
                summary_raw = {}

            summary = DependencyAnalysisSummary(
                total_services=int(summary_raw.get(
                    "total_services",
                    len(input_data.service_graph.nodes),
                )),
                total_dependencies=int(summary_raw.get(
                    "total_dependencies",
                    len(input_data.service_graph.edges),
                )),
                graph_has_cycles=bool(
                    summary_raw.get("graph_has_cycles", False)
                ),
                max_dependency_depth=int(
                    summary_raw.get("max_dependency_depth", 0)
                ),
            )

            # Parse SPOFs
            spof_raw = response.get(
                "single_points_of_failure", []
            )
            spofs: List[SinglePointOfFailure] = []
            if isinstance(spof_raw, list):
                for item in spof_raw:
                    if isinstance(item, dict):
                        spofs.append(SinglePointOfFailure(
                            service_name=item.get(
                                "service_name", "unknown"
                            ),
                            reason=str(
                                item.get("reason", "")
                            ),
                            mitigation=str(
                                item.get("mitigation", "")
                            ),
                        ))

            confidence = float(response.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return ClassificationResult(
                dependency_analysis=summary,
                single_points_of_failure=spofs,
                confidence_score=confidence,
                classification_source="llm",
                classification_latency_ms=0.0,
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM response, "
                f"using fallback: {e}",
                extra={"layer": "llm"},
            )
            return ClassificationResult(
                dependency_analysis=DependencyAnalysisSummary(
                    total_services=len(
                        input_data.service_graph.nodes
                    ),
                    total_dependencies=len(
                        input_data.service_graph.edges
                    ),
                ),
                confidence_score=0.3,
                classification_source="fallback",
                classification_latency_ms=0.0,
            )
