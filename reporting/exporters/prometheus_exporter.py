"""Prometheus exporter — push incident metrics to Prometheus."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
    )

    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _HAS_PROMETHEUS = False


class PrometheusExporter:
    """Export incident metrics to a Prometheus Pushgateway.

    If ``prometheus_client`` is not installed the exporter logs a
    warning and silently no-ops.

    Args:
        gateway_url: Pushgateway URL (e.g. ``"localhost:9091"``).
        job_name: Prometheus job label.
    """

    def __init__(
        self,
        gateway_url: str = "localhost:9091",
        job_name: str = "warroom_simulator",
    ) -> None:
        self._gateway = gateway_url
        self._job = job_name
        self._registry: Optional[Any] = None

        if _HAS_PROMETHEUS:
            self._registry = CollectorRegistry()
            self._incident_count = Counter(
                "warroom_incidents_total",
                "Total incidents processed",
                registry=self._registry,
            )
            self._duration_hist = Histogram(
                "warroom_incident_duration_seconds",
                "Pipeline duration histogram",
                buckets=(1, 2, 5, 10, 15, 30, 60),
                registry=self._registry,
            )
            self._cost_gauge = Gauge(
                "warroom_last_incident_cost",
                "Cost of last incident",
                registry=self._registry,
            )
            self._confidence_gauge = Gauge(
                "warroom_last_incident_confidence",
                "Root-cause confidence",
                registry=self._registry,
            )

    @property
    def available(self) -> bool:
        return _HAS_PROMETHEUS

    def export(
        self,
        *,
        duration: float = 0.0,
        total_cost: float = 0.0,
        confidence: float = 0.0,
    ) -> bool:
        """Record metrics and push to the gateway.

        Args:
            duration: Pipeline duration in seconds.
            total_cost: Total LLM cost.
            confidence: Root-cause confidence.

        Returns:
            ``True`` if the push succeeded, ``False`` otherwise.
        """
        if not _HAS_PROMETHEUS or self._registry is None:
            _logger.warning("prometheus_client not available; skipping export")
            return False

        try:
            self._incident_count.inc()
            self._duration_hist.observe(duration)
            self._cost_gauge.set(total_cost)
            self._confidence_gauge.set(confidence)

            push_to_gateway(
                self._gateway,
                job=self._job,
                registry=self._registry,
            )
            _logger.info("Pushed metrics to Prometheus gateway %s", self._gateway)
            return True
        except Exception:
            _logger.exception("Failed to push to Prometheus gateway")
            return False
