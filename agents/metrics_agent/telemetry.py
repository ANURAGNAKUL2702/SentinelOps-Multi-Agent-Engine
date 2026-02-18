"""
File: telemetry.py
Purpose: Multi-layer observability — latency, counts, costs, structured logging.
Dependencies: Standard library only (logging, time, json)
Performance: <0.1ms overhead per measurement

Provides correlation-ID-aware structured logging and lightweight
Prometheus-compatible counters for every pipeline stage.
Adapted from log_agent.telemetry — same patterns, metrics-domain naming.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional


# ═══════════════════════════════════════════════════════════════
#  STRUCTURED JSON LOGGER
# ═══════════════════════════════════════════════════════════════


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("correlation_id", "layer", "context"):
            val = getattr(record, key, None)
            if val is not None:
                log_obj[key] = val
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = str(record.exc_info[1])
        return json.dumps(log_obj, default=str)


def get_logger(name: str = "metrics_agent") -> logging.Logger:
    """Return a JSON-formatted logger.

    Args:
        name: Logger name (dot-separated hierarchy).

    Returns:
        Configured logging.Logger with JSON formatter.

    Example::

        log = get_logger("metrics_agent.aggregator")
        log.info("Aggregation complete", extra={"correlation_id": cid})
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


# ═══════════════════════════════════════════════════════════════
#  METRICS COUNTERS (Prometheus-compatible)
# ═══════════════════════════════════════════════════════════════


@dataclass
class _Counter:
    """Thread-safe monotonic counter."""
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


@dataclass
class _Histogram:
    """Simple histogram that tracks count, sum, min, max."""
    _count: int = 0
    _sum: float = 0.0
    _min: float = float("inf")
    _max: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            self._min = min(self._min, value)
            self._max = max(self._max, value)

    @property
    def count(self) -> int:
        return self._count

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "count": self._count,
            "sum": round(self._sum, 4),
            "min": round(self._min, 4) if self._min != float("inf") else 0.0,
            "max": round(self._max, 4),
            "avg": round(self.avg, 4),
        }

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._sum = 0.0
            self._min = float("inf")
            self._max = 0.0


# ═══════════════════════════════════════════════════════════════
#  TELEMETRY COLLECTOR
# ═══════════════════════════════════════════════════════════════


class TelemetryCollector:
    """Collects metrics for all layers of the metrics agent pipeline.

    Thread-safe.  Designed to work as a singleton per agent instance.

    Example::

        tel = TelemetryCollector()
        with tel.measure("extraction"):
            # ... do work ...
            pass
        tel.record_llm_call(input_tokens=500, output_tokens=200)
        print(tel.snapshot())
    """

    def __init__(self) -> None:
        self._log = get_logger("metrics_agent.telemetry")

        # ── latency histograms per layer ────────────────────────
        self.latency: Dict[str, _Histogram] = {
            "extraction": _Histogram(),
            "anomaly_detection": _Histogram(),
            "correlation": _Histogram(),
            "classification": _Histogram(),
            "validation": _Histogram(),
            "pipeline_total": _Histogram(),
        }

        # ── counters ────────────────────────────────────────────
        self.analyses_total = _Counter()
        self.analyses_succeeded = _Counter()
        self.analyses_failed = _Counter()
        self.llm_calls_total = _Counter()
        self.llm_calls_failed = _Counter()
        self.llm_tokens_input = _Counter()
        self.llm_tokens_output = _Counter()
        self.llm_cost_usd = _Counter()
        self.fallback_triggers = _Counter()
        self.cache_hits = _Counter()
        self.cache_misses = _Counter()
        self.validation_failures = _Counter()
        self.circuit_breaker_trips = _Counter()

    # ── latency measurement ─────────────────────────────────────

    @contextmanager
    def measure(
        self,
        layer: str,
        correlation_id: str = "",
    ) -> Generator[None, None, None]:
        """Context manager to measure and log latency for a layer.

        Args:
            layer: One of 'extraction', 'anomaly_detection', 'correlation',
                   'classification', 'validation', 'pipeline_total'.
            correlation_id: Request correlation ID for structured logging.

        Example::

            with tel.measure("extraction", cid):
                result = aggregator.aggregate(input_data)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            hist = self.latency.get(layer)
            if hist:
                hist.observe(elapsed_ms)
            self._log.debug(
                f"{layer} completed in {elapsed_ms:.2f}ms",
                extra={
                    "correlation_id": correlation_id,
                    "layer": layer,
                    "context": {"latency_ms": round(elapsed_ms, 2)},
                },
            )

    def measure_value(self, layer: str, latency_ms: float) -> None:
        """Record a pre-computed latency value.

        Args:
            layer: Pipeline layer name.
            latency_ms: Latency in milliseconds.
        """
        hist = self.latency.get(layer)
        if hist:
            hist.observe(latency_ms)

    # ── LLM tracking ───────────────────────────────────────────

    def record_llm_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_per_1k_input: float = 0.00059,
        cost_per_1k_output: float = 0.00079,
        success: bool = True,
        correlation_id: str = "",
    ) -> float:
        """Record an LLM API call with token usage and cost.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
            cost_per_1k_input: Cost per 1K input tokens (USD).
            cost_per_1k_output: Cost per 1K output tokens (USD).
            success: Whether the call succeeded.
            correlation_id: Request correlation ID.

        Returns:
            Cost of this call in USD.
        """
        self.llm_calls_total.inc()
        if not success:
            self.llm_calls_failed.inc()

        self.llm_tokens_input.inc(input_tokens)
        self.llm_tokens_output.inc(output_tokens)

        cost = (
            (input_tokens / 1000) * cost_per_1k_input
            + (output_tokens / 1000) * cost_per_1k_output
        )
        self.llm_cost_usd.inc(cost)

        self._log.info(
            f"LLM call: {input_tokens} in / {output_tokens} out, "
            f"${cost:.6f}, success={success}",
            extra={
                "correlation_id": correlation_id,
                "layer": "llm",
                "context": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": round(cost, 6),
                    "success": success,
                },
            },
        )
        return cost

    def record_fallback(self, correlation_id: str = "") -> None:
        """Record that the fallback rule engine was triggered."""
        self.fallback_triggers.inc()
        self._log.warning(
            "Fallback triggered — LLM unavailable or circuit open",
            extra={
                "correlation_id": correlation_id,
                "layer": "classification",
            },
        )

    def record_cache_hit(self) -> None:
        self.cache_hits.inc()

    def record_cache_miss(self) -> None:
        self.cache_misses.inc()

    def record_validation_failure(self, correlation_id: str = "") -> None:
        self.validation_failures.inc()
        self._log.warning(
            "Validation failed on agent output",
            extra={"correlation_id": correlation_id, "layer": "validation"},
        )

    # ── snapshot ────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a full metrics snapshot (Prometheus-export-ready).

        Returns:
            Dict with latency histograms and counter values.

        Example::

            metrics = tel.snapshot()
        """
        return {
            "latency": {k: v.snapshot() for k, v in self.latency.items()},
            "counters": {
                "analyses_total": self.analyses_total.value,
                "analyses_succeeded": self.analyses_succeeded.value,
                "analyses_failed": self.analyses_failed.value,
                "llm_calls_total": self.llm_calls_total.value,
                "llm_calls_failed": self.llm_calls_failed.value,
                "llm_tokens_input": self.llm_tokens_input.value,
                "llm_tokens_output": self.llm_tokens_output.value,
                "llm_cost_usd": round(self.llm_cost_usd.value, 6),
                "fallback_triggers": self.fallback_triggers.value,
                "cache_hits": self.cache_hits.value,
                "cache_misses": self.cache_misses.value,
                "validation_failures": self.validation_failures.value,
                "circuit_breaker_trips": self.circuit_breaker_trips.value,
            },
        }

    def reset(self) -> None:
        """Reset all counters and histograms (for testing)."""
        for h in self.latency.values():
            h.reset()
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, _Counter):
                obj.reset()
