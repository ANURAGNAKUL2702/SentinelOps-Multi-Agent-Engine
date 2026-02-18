"""
File: telemetry.py
Purpose: Structured logging + lightweight metrics for the validation agent.
Dependencies: Standard library only (logging, time, threading).
Performance: O(1) per metric operation, no I/O blocking.

Adapted from root_cause_agent/telemetry.py — validation-domain layers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


# ═══════════════════════════════════════════════════════════════
#  STRUCTURED JSON FORMATTER
# ═══════════════════════════════════════════════════════════════


class _JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        if hasattr(record, "layer"):
            payload["layer"] = record.layer
        if hasattr(record, "context"):
            payload["context"] = record.context
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = str(record.exc_info[1])
        return json.dumps(payload, default=str)


def get_logger(name: str = "validation_agent") -> logging.Logger:
    """Get or create a structured JSON logger.

    Args:
        name: Logger name (dot-separated hierarchy).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ═══════════════════════════════════════════════════════════════
#  LIGHTWEIGHT METRIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════


class _Counter:
    """Thread-safe monotonic counter."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: int = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


class _Histogram:
    """Thread-safe histogram (min / max / sum / count)."""

    __slots__ = ("_min", "_max", "_sum", "_count", "_lock")

    def __init__(self) -> None:
        self._min: float = float("inf")
        self._max: float = 0.0
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

    @property
    def count(self) -> int:
        return self._count

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "min": self._min if self._count else 0.0,
                "max": self._max,
                "avg": self.avg,
                "sum": self._sum,
                "count": float(self._count),
            }

    def reset(self) -> None:
        with self._lock:
            self._min = float("inf")
            self._max = 0.0
            self._sum = 0.0
            self._count = 0


# ═══════════════════════════════════════════════════════════════
#  TELEMETRY COLLECTOR
# ═══════════════════════════════════════════════════════════════


class TelemetryCollector:
    """Collects latency and throughput metrics for the validation pipeline.

    Layers tracked:
        accuracy_calculation, precision_recall, evidence_validation,
        timeline_validation, hallucination_detection, confusion_matrix,
        calibration, output_validation, pipeline_total

    Counters:
        validations_total / succeeded / failed,
        llm_calls_total / failed, llm_tokens_input / output,
        llm_cost_usd, fallback_triggers, cache_hits / misses,
        output_validation_failures, circuit_breaker_trips,
        hallucinations_detected
    """

    def __init__(self) -> None:
        # ── latency histograms ──────────────────────────────────
        self.accuracy_calculation = _Histogram()
        self.precision_recall = _Histogram()
        self.evidence_validation = _Histogram()
        self.timeline_validation = _Histogram()
        self.hallucination_detection = _Histogram()
        self.confusion_matrix = _Histogram()
        self.calibration = _Histogram()
        self.output_validation = _Histogram()
        self.pipeline_total = _Histogram()

        # ── throughput counters ─────────────────────────────────
        self.validations_total = _Counter()
        self.validations_succeeded = _Counter()
        self.validations_failed = _Counter()

        self.llm_calls_total = _Counter()
        self.llm_calls_failed = _Counter()
        self.llm_tokens_input = _Counter()
        self.llm_tokens_output = _Counter()
        self.llm_cost_usd_millionths = _Counter()

        self.fallback_triggers = _Counter()
        self.cache_hits = _Counter()
        self.cache_misses = _Counter()
        self.output_validation_failures = _Counter()
        self.circuit_breaker_trips = _Counter()
        self.hallucinations_detected = _Counter()

        self._logger = get_logger("validation_agent.telemetry")

    # ── histogram map ───────────────────────────────────────────

    _HISTOGRAM_MAP = {
        "accuracy_calculation": "accuracy_calculation",
        "precision_recall": "precision_recall",
        "evidence_validation": "evidence_validation",
        "timeline_validation": "timeline_validation",
        "hallucination_detection": "hallucination_detection",
        "confusion_matrix": "confusion_matrix",
        "calibration": "calibration",
        "output_validation": "output_validation",
        "pipeline_total": "pipeline_total",
    }

    @contextmanager
    def measure(self, layer: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline layer.

        Args:
            layer: Layer name (must be in _HISTOGRAM_MAP).

        Yields:
            None — records elapsed ms on exit.
        """
        hist_attr = self._HISTOGRAM_MAP.get(layer)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if hist_attr:
                getattr(self, hist_attr).observe(elapsed_ms)

    def measure_value(self, layer: str, value_ms: float) -> None:
        """Record a pre-computed latency value.

        Args:
            layer: Layer name.
            value_ms: Latency in milliseconds.
        """
        hist_attr = self._HISTOGRAM_MAP.get(layer)
        if hist_attr:
            getattr(self, hist_attr).observe(value_ms)

    # ── LLM call recording ─────────────────────────────────────

    def record_llm_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_per_1k_input: float = 0.00059,
        cost_per_1k_output: float = 0.00079,
        success: bool = True,
        correlation_id: str = "",
    ) -> None:
        """Record an LLM API call.

        Args:
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_per_1k_input: USD per 1K input tokens.
            cost_per_1k_output: USD per 1K output tokens.
            success: Whether the call succeeded.
            correlation_id: Request correlation ID.
        """
        self.llm_calls_total.inc()

        if not success:
            self.llm_calls_failed.inc()
            return

        self.llm_tokens_input.inc(input_tokens)
        self.llm_tokens_output.inc(output_tokens)

        cost_usd = (
            (input_tokens / 1000.0) * cost_per_1k_input
            + (output_tokens / 1000.0) * cost_per_1k_output
        )
        cost_millionths = int(cost_usd * 1_000_000)
        self.llm_cost_usd_millionths.inc(cost_millionths)

    def record_cache_hit(self) -> None:
        """Record an LLM cache hit."""
        self.cache_hits.inc()

    def record_cache_miss(self) -> None:
        """Record an LLM cache miss."""
        self.cache_misses.inc()

    # ── snapshot ────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of all metrics.

        Returns:
            Dict with 'latency' and 'counters' sub-dicts.
        """
        return {
            "latency": {
                name: getattr(self, attr).snapshot()
                for name, attr in self._HISTOGRAM_MAP.items()
            },
            "counters": {
                "validations_total": self.validations_total.value,
                "validations_succeeded": (
                    self.validations_succeeded.value
                ),
                "validations_failed": self.validations_failed.value,
                "llm_calls_total": self.llm_calls_total.value,
                "llm_calls_failed": self.llm_calls_failed.value,
                "llm_tokens_input": self.llm_tokens_input.value,
                "llm_tokens_output": self.llm_tokens_output.value,
                "llm_cost_usd": (
                    self.llm_cost_usd_millionths.value / 1_000_000
                ),
                "fallback_triggers": self.fallback_triggers.value,
                "cache_hits": self.cache_hits.value,
                "cache_misses": self.cache_misses.value,
                "output_validation_failures": (
                    self.output_validation_failures.value
                ),
                "circuit_breaker_trips": (
                    self.circuit_breaker_trips.value
                ),
                "hallucinations_detected": (
                    self.hallucinations_detected.value
                ),
            },
        }

    # ── reset (for testing) ─────────────────────────────────────

    def reset(self) -> None:
        """Reset all metrics to zero (for testing)."""
        for attr in self._HISTOGRAM_MAP.values():
            getattr(self, attr).reset()

        for counter_name in (
            "validations_total",
            "validations_succeeded",
            "validations_failed",
            "llm_calls_total",
            "llm_calls_failed",
            "llm_tokens_input",
            "llm_tokens_output",
            "llm_cost_usd_millionths",
            "fallback_triggers",
            "cache_hits",
            "cache_misses",
            "output_validation_failures",
            "circuit_breaker_trips",
            "hallucinations_detected",
        ):
            getattr(self, counter_name).reset()
