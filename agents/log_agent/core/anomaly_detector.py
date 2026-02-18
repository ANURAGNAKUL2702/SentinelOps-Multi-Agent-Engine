"""
File: core/anomaly_detector.py
Purpose: Statistical anomaly detection using z-scores and multi-metric correlation.
Dependencies: math (standard library)
Performance: <50ms for 1000 services

Provides z-score-based (3-sigma) anomaly detection on extracted signals.
No ML required — pure statistics.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from agents.log_agent.config import LogAgentConfig
from agents.log_agent.schema import (
    AnomalyDetectionResult,
    AnomalyResult,
    ServiceSignal,
    SignalExtractionResult,
)
from agents.log_agent.telemetry import get_logger

logger = get_logger("log_agent.anomaly_detector")


class AnomalyDetector:
    """Statistical anomaly detector using z-score analysis.

    Identifies services whose metrics deviate significantly from
    the population mean.  Uses the 3-sigma rule by default.

    Args:
        config: Agent configuration (thresholds).

    Example::

        detector = AnomalyDetector(LogAgentConfig())
        anomalies = detector.detect(extraction_result)
        for a in anomalies.anomalies:
            print(f"{a.service}.{a.metric}: z={a.z_score:.2f}")
    """

    # Metrics to evaluate for anomalies
    _METRICS = [
        ("error_percentage", lambda s: s.error_percentage),
        ("growth_rate_last_period", lambda s: s.growth_rate_last_period),
        ("error_count", lambda s: float(s.error_count)),
    ]

    def __init__(self, config: Optional[LogAgentConfig] = None) -> None:
        self._config = config or LogAgentConfig()
        self._threshold = self._config.thresholds.z_score_threshold

    def detect(
        self,
        extraction: SignalExtractionResult,
        correlation_id: str = "",
    ) -> AnomalyDetectionResult:
        """Run anomaly detection on extraction results.

        Args:
            extraction: Output of SignalExtractor.extract().
            correlation_id: Request correlation ID.

        Returns:
            AnomalyDetectionResult with flagged anomalies.
        """
        start = time.perf_counter()
        signals = extraction.service_signals
        anomalies: List[AnomalyResult] = []

        if len(signals) < 2:
            # Need at least 2 data points for meaningful z-scores
            elapsed_ms = (time.perf_counter() - start) * 1000
            return AnomalyDetectionResult(
                anomalies=[], detection_latency_ms=round(elapsed_ms, 2)
            )

        for metric_name, accessor in self._METRICS:
            values = [accessor(s) for s in signals]
            mean, std = self._mean_std(values)

            if std == 0.0:
                # No variance — no anomalies possible
                continue

            for sig in signals:
                val = accessor(sig)
                z = abs(val - mean) / std

                is_anomaly = z > self._threshold
                if is_anomaly:
                    anomalies.append(
                        AnomalyResult(
                            service=sig.service,
                            metric=metric_name,
                            value=round(val, 2),
                            z_score=round(z, 2),
                            threshold=self._threshold,
                            is_anomaly=True,
                        )
                    )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Anomaly detection completed: {len(anomalies)} anomalies "
            f"in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "anomaly_detection",
                "context": {
                    "anomaly_count": len(anomalies),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return AnomalyDetectionResult(
            anomalies=anomalies,
            detection_latency_ms=round(elapsed_ms, 2),
        )

    # ── statistics helpers ──────────────────────────────────────

    @staticmethod
    def _mean_std(values: List[float]) -> tuple[float, float]:
        """Compute mean and standard deviation.

        Args:
            values: List of numeric values.

        Returns:
            (mean, std) tuple.  Returns (0.0, 0.0) for empty input.

        Example::

            AnomalyDetector._mean_std([10, 20, 30])  # (20.0, 8.165)
            AnomalyDetector._mean_std([])              # (0.0, 0.0)
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean = sum(values) / n
        if n == 1:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in values) / n
        return mean, math.sqrt(variance)
