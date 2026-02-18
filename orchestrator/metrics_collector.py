"""Prometheus metrics collection and export."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

from .config import OrchestratorConfig
from .schema import PipelineStatus, PipelineTelemetry
from .telemetry import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import prometheus_client; fall back to a no-op shim if unavailable
# so the orchestrator can still run without the optional dependency.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (  # type: ignore[import-untyped]
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except ImportError:  # pragma: no cover – optional dep
    _HAS_PROM = False


class MetricsCollector:
    """Aggregate pipeline and per-agent metrics.

    When ``config.enable_prometheus_metrics`` is ``True`` **and**
    ``prometheus_client`` is installed, real Prometheus metrics are used.
    Otherwise every method is a no-op.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._enabled = config.enable_prometheus_metrics and _HAS_PROM

        # In-memory accumulators (always active)
        self._agent_durations: Dict[str, float] = {}
        self._agent_statuses: Dict[str, str] = {}
        self._timeout_counts: Dict[str, int] = defaultdict(int)
        self._retry_counts: Dict[str, int] = defaultdict(int)
        self._llm_costs: Dict[str, float] = defaultdict(float)
        self._llm_calls: Dict[str, int] = defaultdict(int)
        self._cb_trips: int = 0
        self._pipeline_count: Dict[str, int] = defaultdict(int)

        # Prometheus objects (isolated registry per instance for test safety)
        if self._enabled:
            self._registry = CollectorRegistry()
            self._prom_agent_exec = Histogram(
                "agent_execution_seconds",
                "Agent execution time in seconds",
                labelnames=["agent_name", "status"],
                registry=self._registry,
            )
            self._prom_pipeline = Counter(
                "pipeline_executions_total",
                "Total pipeline executions",
                labelnames=["status"],
                registry=self._registry,
            )
            self._prom_agent_fail = Counter(
                "agent_failures_total",
                "Agent failure count",
                labelnames=["agent_name", "error_type"],
                registry=self._registry,
            )
            self._prom_cb_state = Gauge(
                "circuit_breaker_state",
                "Circuit breaker state (0=CLOSED,1=OPEN,2=HALF_OPEN)",
                labelnames=["agent_name"],
                registry=self._registry,
            )
            self._prom_timeouts = Counter(
                "timeout_violations_total",
                "Timeout violations",
                labelnames=["agent_name"],
                registry=self._registry,
            )
            self._prom_retries = Counter(
                "retry_attempts_total",
                "Retry attempts",
                labelnames=["agent_name"],
                registry=self._registry,
            )
            self._prom_llm_cost = Counter(
                "llm_cost_dollars_total",
                "LLM API cost in dollars",
                labelnames=["agent_name"],
                registry=self._registry,
            )

    # ---- recording --------------------------------------------------------

    def record_agent_execution(
        self, agent_name: str, duration: float, status: str
    ) -> None:
        """Record agent execution time.

        Args:
            agent_name: Agent identifier.
            duration: Execution time in seconds.
            status: ``"success"`` or ``"failed"``.
        """
        self._agent_durations[agent_name] = duration
        self._agent_statuses[agent_name] = status
        if self._enabled:
            self._prom_agent_exec.labels(
                agent_name=agent_name, status=status
            ).observe(duration)

    def record_pipeline_result(
        self, status: PipelineStatus, duration: float
    ) -> None:
        """Record completion of a pipeline run.

        Args:
            status: Final pipeline status.
            duration: Total elapsed seconds.
        """
        self._pipeline_count[status.value] += 1
        if self._enabled:
            self._prom_pipeline.labels(status=status.value).inc()

    def record_agent_failure(
        self, agent_name: str, error_type: str
    ) -> None:
        """Record an agent failure.

        Args:
            agent_name: Agent identifier.
            error_type: Canonical error type string.
        """
        if self._enabled:
            self._prom_agent_fail.labels(
                agent_name=agent_name, error_type=error_type
            ).inc()

    def record_circuit_breaker_state_change(
        self, agent_name: str, from_state: str, to_state: str
    ) -> None:
        """Record a circuit-breaker state transition.

        Args:
            agent_name: Agent identifier.
            from_state: Previous state label.
            to_state: New state label.
        """
        self._cb_trips += 1
        _state_vals = {"closed": 0, "open": 1, "half_open": 2}
        if self._enabled:
            self._prom_cb_state.labels(agent_name=agent_name).set(
                _state_vals.get(to_state, 0)
            )

    def record_timeout(self, agent_name: str) -> None:
        """Record a timeout violation.

        Args:
            agent_name: Agent identifier.
        """
        self._timeout_counts[agent_name] += 1
        if self._enabled:
            self._prom_timeouts.labels(agent_name=agent_name).inc()

    def record_retry(self, agent_name: str) -> None:
        """Record a retry attempt.

        Args:
            agent_name: Agent identifier.
        """
        self._retry_counts[agent_name] += 1
        if self._enabled:
            self._prom_retries.labels(agent_name=agent_name).inc()

    def record_llm_cost(self, agent_name: str, cost: float) -> None:
        """Record LLM API cost.

        Args:
            agent_name: Agent identifier.
            cost: Cost in dollars.
        """
        self._llm_costs[agent_name] += cost
        self._llm_calls[agent_name] += 1
        if self._enabled:
            self._prom_llm_cost.labels(agent_name=agent_name).inc(cost)

    # ---- export -----------------------------------------------------------

    def export_metrics(self) -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Multi-line string in Prometheus format, or ``""`` if disabled.
        """
        if not self._enabled:
            return ""
        return generate_latest(self._registry).decode("utf-8")

    def get_telemetry(self) -> PipelineTelemetry:
        """Build :class:`PipelineTelemetry` from accumulated data.

        Returns:
            Aggregate telemetry for the current pipeline run.
        """
        total_cost = sum(self._llm_costs.values())
        total_calls = sum(self._llm_calls.values())
        timeout_violations = sum(self._timeout_counts.values())

        return PipelineTelemetry(
            total_llm_cost=total_cost,
            total_llm_calls=total_calls,
            total_tokens=0,
            agent_latencies=dict(self._agent_durations),
            parallel_speedup=1.0,  # overridden by result_aggregator
            timeout_violations=timeout_violations,
            circuit_breaker_trips=self._cb_trips,
        )

    # ---- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Clear in-memory accumulators for a new pipeline run."""
        self._agent_durations.clear()
        self._agent_statuses.clear()
        self._timeout_counts.clear()
        self._retry_counts.clear()
        self._llm_costs.clear()
        self._llm_calls.clear()
        self._cb_trips = 0
