"""
File: telemetry.py
Purpose: Structured logging + lightweight metrics for the Incident Commander.
Dependencies: Standard library only (logging, time, threading).
Performance: O(1) per metric operation, no I/O blocking.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator


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
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = str(record.exc_info[1])
        return json.dumps(payload, default=str)


def get_logger(name: str = "incident_commander_agent") -> logging.Logger:
    """Get or create a structured JSON logger.

    Args:
        name: Logger name.

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
    """Collects latency and throughput metrics for the command pipeline."""

    def __init__(self) -> None:
        # ── latency histograms ──
        self.runbook_generation = _Histogram()
        self.blast_radius_calculation = _Histogram()
        self.priority_ranking = _Histogram()
        self.action_sequencing = _Histogram()
        self.rollback_planning = _Histogram()
        self.communication_building = _Histogram()
        self.prevention_advising = _Histogram()
        self.escalation_calculation = _Histogram()
        self.output_validation = _Histogram()
        self.pipeline_total = _Histogram()

        # ── throughput counters ──
        self.commands_total = _Counter()
        self.commands_succeeded = _Counter()
        self.commands_failed = _Counter()
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
        self.escalations_triggered = _Counter()

        self._logger = get_logger("incident_commander_agent.telemetry")

    _HISTOGRAM_MAP = {
        "runbook_generation": "runbook_generation",
        "blast_radius_calculation": "blast_radius_calculation",
        "priority_ranking": "priority_ranking",
        "action_sequencing": "action_sequencing",
        "rollback_planning": "rollback_planning",
        "communication_building": "communication_building",
        "prevention_advising": "prevention_advising",
        "escalation_calculation": "escalation_calculation",
        "output_validation": "output_validation",
        "pipeline_total": "pipeline_total",
    }

    @contextmanager
    def measure(self, layer: str) -> Generator[None, None, None]:
        """Context manager to time a pipeline layer.

        Args:
            layer: Layer name.

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

    def record_llm_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_per_1k_input: float = 0.00059,
        cost_per_1k_output: float = 0.00079,
        success: bool = True,
    ) -> None:
        """Record an LLM API call.

        Args:
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_per_1k_input: USD per 1K input tokens.
            cost_per_1k_output: USD per 1K output tokens.
            success: Whether the call succeeded.
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
        self.llm_cost_usd_millionths.inc(int(cost_usd * 1_000_000))

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of all metrics."""
        return {
            "latency": {
                name: getattr(self, attr).snapshot()
                for name, attr in self._HISTOGRAM_MAP.items()
            },
            "counters": {
                "commands_total": self.commands_total.value,
                "commands_succeeded": self.commands_succeeded.value,
                "commands_failed": self.commands_failed.value,
                "llm_calls_total": self.llm_calls_total.value,
                "fallback_triggers": self.fallback_triggers.value,
                "escalations_triggered": (
                    self.escalations_triggered.value
                ),
            },
        }

    def reset(self) -> None:
        """Reset all metrics to zero (for testing)."""
        for attr in self._HISTOGRAM_MAP.values():
            getattr(self, attr).reset()
        for name in (
            "commands_total", "commands_succeeded", "commands_failed",
            "llm_calls_total", "llm_calls_failed",
            "llm_tokens_input", "llm_tokens_output",
            "llm_cost_usd_millionths", "fallback_triggers",
            "cache_hits", "cache_misses",
            "output_validation_failures", "circuit_breaker_trips",
            "escalations_triggered",
        ):
            getattr(self, name).reset()
