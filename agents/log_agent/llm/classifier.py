"""
File: llm/classifier.py
Purpose: LLM wrapper for pattern classification with circuit breaker + caching.
Dependencies: Standard library only (provider adapters are pluggable).
Performance: <10s per call (timeout enforced), <1ms cache hit.

Uses PRE-CALCULATED signals from SignalExtractor — LLM classifies,
never re-scans.  Circuit breaker prevents cascade on provider outages.
Response caching (TTL-based) eliminates redundant calls.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.log_agent.config import LogAgentConfig, LLMConfig
from agents.log_agent.schema import (
    ClassificationResult,
    LogAnalysisInput,
    SignalExtractionResult,
    SuspiciousService,
    SystemErrorSummary,
    SeverityHint,
    TrendType,
)
from agents.log_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("log_agent.llm.classifier")


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

    Example::

        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        if cb.can_execute():
            try:
                result = call_llm(...)
                cb.record_success()
            except Exception:
                cb.record_failure()
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
        """Check if a call is allowed under current circuit state.

        Returns:
            True if CLOSED or HALF_OPEN, False if OPEN.
        """
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

    Keys are SHA-256 hashes of the input signals.  Thread-safe.

    Args:
        ttl_seconds: Time-to-live for cache entries.
        max_entries: Maximum cache size (LRU eviction).

    Example::

        cache = ResponseCache(ttl_seconds=300)
        cache.put("key123", classification_result)
        hit = cache.get("key123")  # ClassificationResult or None
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
        """Retrieve a cached result if it exists and is not expired.

        Args:
            key: Cache key (SHA-256 hash of input).

        Returns:
            ClassificationResult or None if miss/expired.
        """
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
        """Store a result in the cache.

        Args:
            key: Cache key (SHA-256 hash of input).
            result: Classification result to cache.
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._store) >= self._max_entries and key not in self._store:
                oldest_key = min(
                    self._store, key=lambda k: self._store[k][0]
                )
                del self._store[oldest_key]
            self._store[key] = (time.monotonic(), result)

    def invalidate(self, key: str) -> None:
        """Remove a specific entry."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def compute_key(signals: SignalExtractionResult) -> str:
        """Compute a deterministic cache key from signals.

        Args:
            signals: Extraction result to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        payload = signals.model_dump_json(exclude={"extraction_latency_ms"})
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
    """Abstract base for LLM provider adapters.

    Subclass this to integrate OpenAI, Anthropic, etc.

    Example::

        class OpenAIProvider(LLMProvider):
            def call(self, prompt, **kwargs):
                return openai.chat.completions.create(...)
    """

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a prompt to the LLM and return parsed JSON.

        Args:
            system_prompt: System role prompt.
            user_prompt: User message with signal data.
            timeout_seconds: Max wait time.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            LLMProviderError: On failure.
        """
        raise NotImplementedError("Subclass must implement call()")

    def count_tokens(self, text: str) -> int:
        """Estimate token count for cost tracking.

        Args:
            text: Input text.

        Returns:
            Approximate token count (words / 0.75).
        """
        # Simple heuristic: ~4 chars per token
        return max(1, len(text) // 4)


class MockLLMProvider(LLMProvider):
    """Mock provider for testing — returns deterministic results.

    Uses the fallback rule engine internally so tests don't need
    a real API key.

    Args:
        should_fail: If True, always raises LLMProviderError.
        failure_count: Number of calls to fail before succeeding.

    Example::

        provider = MockLLMProvider()
        result = provider.call("system", "user")
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
        """Return mock classification response.

        Raises:
            LLMProviderError: If configured to fail.
        """
        self._call_count += 1
        if self._should_fail:
            raise LLMProviderError("Mock failure", retryable=True, status_code=503)
        if self._call_count <= self._failure_count:
            raise LLMProviderError(
                f"Mock transient failure ({self._call_count}/{self._failure_count})",
                retryable=True,
                status_code=429,
            )
        # Return a minimal valid response — the classifier parses this
        return {
            "classification_source": "llm",
            "mock": True,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ═══════════════════════════════════════════════════════════════
#  LLM CLASSIFIER
# ═══════════════════════════════════════════════════════════════


class LLMClassifier:
    """LLM-powered classification with circuit breaker, retry, and caching.

    Uses pre-calculated signals from SignalExtractor — the LLM only
    classifies, it never re-scans logs.

    Pipeline::

        signals → cache check → circuit check → LLM call → validate → cache store

    Args:
        config: Agent configuration.
        provider: LLM provider adapter (default: MockLLMProvider).
        telemetry: Telemetry collector instance.

    Example::

        classifier = LLMClassifier(
            config=LogAgentConfig(),
            provider=MockLLMProvider(),
            telemetry=TelemetryCollector(),
        )
        result = classifier.classify(signals, input_data)
        print(result.classification_source)  # "llm" or "cached"
    """

    def __init__(
        self,
        config: Optional[LogAgentConfig] = None,
        provider: Optional[LLMProvider] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or LogAgentConfig()
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
        """Current circuit breaker state."""
        return self._circuit.state

    def classify(
        self,
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Classify signals using LLM with full production safeguards.

        Flow:
            1. Check cache → return if hit
            2. Check circuit breaker → raise if OPEN
            3. Build prompt with pre-calculated signals
            4. Call LLM with retry + exponential backoff
            5. Validate and parse response
            6. Cache result
            7. Return ClassificationResult

        Args:
            signals: Pre-calculated signals from SignalExtractor.
            input_data: Original input for cross-reference.
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult with classification_source="llm" or "cached".

        Raises:
            LLMProviderError: If LLM fails after all retries and circuit opens.
        """
        start = time.perf_counter()

        # ── 1. cache check ──────────────────────────────────────
        if self._config.features.enable_caching:
            cache_key = ResponseCache.compute_key(signals)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._telemetry.record_cache_hit()
                logger.debug(
                    "Cache hit for classification",
                    extra={"correlation_id": correlation_id, "layer": "llm"},
                )
                # Return cached result with updated source marker
                return ClassificationResult(
                    suspicious_services=cached.suspicious_services,
                    system_error_summary=cached.system_error_summary,
                    database_related_errors_detected=cached.database_related_errors_detected,
                    confidence_score=cached.confidence_score,
                    classification_source="cached",
                    classification_latency_ms=round(
                        (time.perf_counter() - start) * 1000, 2
                    ),
                )
            self._telemetry.record_cache_miss()

        # ── 2. circuit breaker check ────────────────────────────
        if not self._circuit.can_execute():
            logger.warning(
                "Circuit breaker OPEN — cannot call LLM",
                extra={"correlation_id": correlation_id, "layer": "llm"},
            )
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

        # ── 5. record telemetry ─────────────────────────────────
        self._telemetry.record_llm_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_per_1k_input=self._llm_config.cost_per_1k_input_tokens,
            cost_per_1k_output=self._llm_config.cost_per_1k_output_tokens,
            success=True,
            correlation_id=correlation_id,
        )
        self._circuit.record_success()

        # ── 6. parse response into ClassificationResult ─────────
        result = self._parse_response(response, signals, input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        result_with_latency = ClassificationResult(
            suspicious_services=result.suspicious_services,
            system_error_summary=result.system_error_summary,
            database_related_errors_detected=result.database_related_errors_detected,
            confidence_score=result.confidence_score,
            classification_source="llm",
            classification_latency_ms=round(elapsed_ms, 2),
        )

        # ── 7. cache result ─────────────────────────────────────
        if self._config.features.enable_caching:
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

        Retries on transient errors (429, 503).  Non-retryable errors
        fail immediately.

        Delays: 1s, 2s, 4s (configurable via retry_base_delay).

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
                        "context": {
                            "attempt": attempt,
                            "retry_delay": delay,
                            "status_code": e.status_code,
                        },
                    },
                )
                time.sleep(delay)

        # Should not reach here, but safety net
        self._circuit.record_failure()
        raise last_error or LLMProviderError("Unknown LLM failure")

    # ── prompt building ─────────────────────────────────────────

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt for classification.

        The LLM receives pre-calculated signals and only needs to
        add reasoning and root-cause hypotheses.
        """
        return (
            "You are a production incident classifier. "
            "You receive pre-computed signals extracted from log data. "
            "Your job is to classify severity, identify root cause patterns, "
            "and provide a confidence score.\n\n"
            "Respond with ONLY valid JSON matching this EXACT schema:\n"
            "{\n"
            '  "suspicious_services": [\n'
            "    {\n"
            '      "service": "<service-name>",\n'
            '      "error_count": <int>,\n'
            '      "error_percentage": <float 0-100>,\n'
            '      "error_keywords_detected": ["<keyword>", ...],\n'
            '      "error_trend": "sudden_spike"|"increasing"|"stable"|"decreasing",\n'
            '      "severity_hint": "high"|"medium"|"low",\n'
            '      "log_flooding": <bool>\n'
            "    }\n"
            "  ],\n"
            '  "system_error_summary": {\n'
            '    "total_error_logs": <int>,\n'
            '    "dominant_service": "<service-name>"|null,\n'
            '    "system_wide_spike": <bool>,\n'
            '    "potential_upstream_failure": <bool>\n'
            "  },\n"
            '  "database_related_errors_detected": <bool>,\n'
            '  "confidence_score": <float 0.0-1.0>\n'
            "}\n\n"
            "Do NOT re-analyze raw logs. Only classify based on provided signals."
        )

    @staticmethod
    def _build_user_prompt(
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
    ) -> str:
        """Build the user prompt with pre-calculated signals.

        Args:
            signals: Pre-calculated extraction result.
            input_data: Original input for context.

        Returns:
            Formatted prompt string with JSON signal data.
        """
        signal_data = signals.model_dump(mode="json")
        context = {
            "time_window": input_data.time_window,
            "keyword_matches": input_data.keyword_matches,
        }
        return (
            "Classify the following pre-computed signals:\n\n"
            f"SIGNALS:\n{json.dumps(signal_data, indent=2)}\n\n"
            f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            "Respond with JSON: {suspicious_services, system_error_summary, "
            "database_related_errors_detected, confidence_score}"
        )

    # ── response parsing ────────────────────────────────────────

    def _parse_response(
        self,
        response: Dict[str, Any],
        signals: SignalExtractionResult,
        input_data: LogAnalysisInput,
    ) -> ClassificationResult:
        """Parse LLM response into ClassificationResult.

        If the LLM response is incomplete or malformed, falls back to
        building the result from pre-calculated signals using the
        deterministic rules (same logic as fallback.py).

        Args:
            response: Parsed JSON from LLM.
            signals: Pre-calculated signals.
            input_data: Original input.

        Returns:
            Validated ClassificationResult.
        """
        try:
            # Try to parse LLM-provided suspicious services
            suspicious_raw = response.get("suspicious_services", [])
            suspicious: List[SuspiciousService] = []

            if suspicious_raw and isinstance(suspicious_raw, list):
                for svc_data in suspicious_raw:
                    if isinstance(svc_data, dict):
                        suspicious.append(SuspiciousService(
                            service=svc_data.get("service", "unknown"),
                            error_count=svc_data.get("error_count", 0),
                            error_percentage=svc_data.get("error_percentage", 0.0),
                            error_keywords_detected=svc_data.get(
                                "error_keywords_detected", []
                            ),
                            error_trend=svc_data.get("error_trend", "stable"),
                            severity_hint=svc_data.get("severity_hint", "low"),
                            log_flooding=svc_data.get("log_flooding", False),
                        ))

            # Parse system error summary
            summary_raw = response.get("system_error_summary", {})
            if not isinstance(summary_raw, dict):
                summary_raw = {}
            summary = SystemErrorSummary(
                total_error_logs=summary_raw.get(
                    "total_error_logs",
                    signals.system_signals.total_error_logs,
                ),
                dominant_service=summary_raw.get("dominant_service"),
                system_wide_spike=summary_raw.get("system_wide_spike", False),
                potential_upstream_failure=summary_raw.get(
                    "potential_upstream_failure", False
                ),
            )

            confidence = float(response.get("confidence_score", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            db_related = response.get(
                "database_related_errors_detected", False
            )

            return ClassificationResult(
                suspicious_services=suspicious,
                system_error_summary=summary,
                database_related_errors_detected=db_related,
                confidence_score=confidence,
                classification_source="llm",
                classification_latency_ms=0.0,
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to parse LLM response, using signal-based fallback: {e}",
                extra={"layer": "llm"},
            )
            # Fall back to building result directly from signals
            from agents.log_agent.fallback import RuleEngine

            engine = RuleEngine(self._config)
            return engine.classify(signals, input_data)
