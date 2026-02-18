"""
File: llm/classifier.py
Purpose: LLM wrapper for metrics classification with circuit breaker + caching.
Dependencies: Standard library only (provider adapters are pluggable).
Performance: <10s per call (timeout enforced), <1ms cache hit.

Uses PRE-CALCULATED signals from MetricAggregator — LLM classifies,
never re-scans raw metrics.  Circuit breaker prevents cascade on provider
outages.  Response caching eliminates redundant calls.

Adapted from log_agent/llm/classifier.py — same infrastructure,
metrics-domain prompts and parsing.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.metrics_agent.config import MetricsAgentConfig, LLMConfig
from agents.metrics_agent.schema import (
    AggregationResult,
    AnomalousMetric,
    AnomalyType,
    ClassificationResult,
    CorrelationDetectionResult,
    CorrelationRelationship,
    CorrelationResult,
    MetricsAnalysisInput,
    Severity,
    SystemSummary,
    TrendType,
)
from agents.metrics_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("metrics_agent.llm.classifier")


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
                logger.warning("Circuit breaker → OPEN (half-open probe failed)")
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
            if len(self._store) >= self._max_entries and key not in self._store:
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
    def compute_key(aggregation: AggregationResult) -> str:
        """Compute a deterministic cache key from aggregation signals.

        Args:
            aggregation: Aggregation result to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = aggregation.model_dump_json(
            exclude={"aggregation_latency_ms"}
        )
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
            raise LLMProviderError("Mock failure", retryable=True, status_code=503)
        if self._call_count <= self._failure_count:
            raise LLMProviderError(
                f"Mock transient failure ({self._call_count}/{self._failure_count})",
                retryable=True, status_code=429,
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
        config: Optional[MetricsAgentConfig] = None,
        provider: Optional[LLMProvider] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or MetricsAgentConfig()
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
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Classify metrics using LLM with full production safeguards.

        Flow:
            1. Check cache → return if hit
            2. Check circuit breaker → raise if OPEN
            3. Build prompt from pre-calculated signals
            4. Call LLM with retry + exponential backoff
            5. Parse and validate response
            6. Cache result
            7. Return ClassificationResult

        Args:
            aggregation: Pre-calculated metric signals.
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
            cache_key = ResponseCache.compute_key(aggregation)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._telemetry.record_cache_hit()
                elapsed = (time.perf_counter() - start) * 1000
                return ClassificationResult(
                    anomalous_metrics=cached.anomalous_metrics,
                    correlations=cached.correlations,
                    system_summary=cached.system_summary,
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
        user_prompt = self._build_user_prompt(aggregation, input_data)
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
        result = self._parse_response(response, aggregation, input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result_with_latency = ClassificationResult(
            anomalous_metrics=result.anomalous_metrics,
            correlations=result.correlations,
            system_summary=result.system_summary,
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

        Delays: 1s, 2s, 4s (configurable).

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
                        f"LLM call failed (attempt {attempt}/{max_retries}): {e}",
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
                    f"LLM call failed (attempt {attempt}/{max_retries}), "
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
        """Build the system prompt for metrics classification."""
        return (
            "You are a production metrics incident classifier. "
            "You receive pre-computed metric signals (Z-scores, trends, "
            "threshold breaches) from a metrics aggregator. "
            "Your job is to classify anomaly patterns and severity.\n\n"
            "Respond with ONLY valid JSON matching this EXACT schema:\n"
            "{\n"
            '  "anomalous_metrics": [\n'
            "    {\n"
            '      "metric_name": "<name>",\n'
            '      "current_value": <float>,\n'
            '      "previous_value": <float>,\n'
            '      "baseline_mean": <float>,\n'
            '      "baseline_stddev": <float>,\n'
            '      "zscore": <float>,\n'
            '      "deviation_percent": <float>,\n'
            '      "growth_rate": <float>,\n'
            '      "is_anomalous": true,\n'
            '      "anomaly_type": "spike"|"sustained"|"oscillating"|"none",\n'
            '      "severity": "critical"|"high"|"medium"|"low"|"info",\n'
            '      "trend": "sudden_spike"|"increasing"|"stable"|"decreasing",\n'
            '      "threshold_breached": <bool>,\n'
            '      "reasoning": "<explanation>"\n'
            "    }\n"
            "  ],\n"
            '  "system_summary": {\n'
            '    "total_metrics_analyzed": <int>,\n'
            '    "total_anomalies_detected": <int>,\n'
            '    "resource_saturation": <bool>,\n'
            '    "cascading_degradation": <bool>\n'
            "  },\n"
            '  "confidence_score": <float 0.0-1.0>,\n'
            '  "confidence_reasoning": "<explanation>"\n'
            "}\n\n"
            "Do NOT re-analyze raw metrics. Only classify based on signals."
        )

    @staticmethod
    def _build_user_prompt(
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
    ) -> str:
        """Build the user prompt with pre-calculated signals.

        Args:
            aggregation: Pre-calculated aggregation result.
            input_data: Original input for context.

        Returns:
            Formatted prompt string.
        """
        signal_data = aggregation.model_dump(mode="json")
        context = {
            "service": input_data.service,
            "time_window": input_data.time_window,
        }
        return (
            "Classify the following pre-computed metric signals:\n\n"
            f"SIGNALS:\n{json.dumps(signal_data, indent=2)}\n\n"
            f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            "Respond with JSON: {anomalous_metrics, system_summary, "
            "confidence_score, confidence_reasoning}"
        )

    # ── response parsing ────────────────────────────────────────

    def _parse_response(
        self,
        response: Dict[str, Any],
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
    ) -> ClassificationResult:
        """Parse LLM response into ClassificationResult.

        Falls back to rule engine if parsing fails.

        Args:
            response: Parsed JSON from LLM.
            aggregation: Pre-calculated signals.
            input_data: Original input.

        Returns:
            Validated ClassificationResult.
        """
        try:
            # Parse anomalous metrics
            anomalous_raw = response.get("anomalous_metrics", [])
            anomalous_metrics: List[AnomalousMetric] = []

            if isinstance(anomalous_raw, list):
                for item in anomalous_raw:
                    if isinstance(item, dict):
                        anomalous_metrics.append(AnomalousMetric(
                            metric_name=item.get("metric_name", "unknown"),
                            current_value=float(item.get("current_value", 0)),
                            previous_value=float(item.get("previous_value", 0)),
                            baseline_mean=float(item.get("baseline_mean", 0)),
                            baseline_stddev=float(item.get("baseline_stddev", 0)),
                            zscore=float(item.get("zscore", 0)),
                            deviation_percent=float(item.get("deviation_percent", 0)),
                            growth_rate=float(item.get("growth_rate", 0)),
                            is_anomalous=bool(item.get("is_anomalous", True)),
                            anomaly_type=item.get("anomaly_type", "none"),
                            severity=item.get("severity", "info"),
                            trend=item.get("trend", "stable"),
                            threshold_breached=bool(item.get("threshold_breached", False)),
                            reasoning=str(item.get("reasoning", "")),
                        ))

            # Parse system summary
            summary_raw = response.get("system_summary", {})
            if not isinstance(summary_raw, dict):
                summary_raw = {}

            summary = SystemSummary(
                total_metrics_analyzed=int(summary_raw.get(
                    "total_metrics_analyzed",
                    aggregation.total_metrics_analyzed,
                )),
                total_anomalies_detected=int(summary_raw.get(
                    "total_anomalies_detected",
                    len(anomalous_metrics),
                )),
                resource_saturation=bool(summary_raw.get(
                    "resource_saturation", False
                )),
                cascading_degradation=bool(summary_raw.get(
                    "cascading_degradation", False
                )),
            )

            confidence = float(response.get("confidence_score", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(response.get("confidence_reasoning", ""))

            return ClassificationResult(
                anomalous_metrics=anomalous_metrics,
                correlations=[],  # LLM doesn't compute correlations
                system_summary=summary,
                confidence_score=confidence,
                confidence_reasoning=reasoning,
                classification_source="llm",
                classification_latency_ms=0.0,
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM response, using fallback: {e}",
                extra={"layer": "llm"},
            )
            from agents.metrics_agent.fallback import RuleEngine
            engine = RuleEngine(self._config)
            return engine.classify(
                aggregation, None, None, input_data
            )
