"""
File: core/anomaly_detector.py
Purpose: Anomaly type classification — spike, sustained, oscillating.
Dependencies: Standard library only
Performance: <1ms per metric, O(n) complexity

Implements Algorithm 3: Anomaly Type Classification
Uses timeseries shape + growth rate to classify anomaly patterns.
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AggregationResult,
    AnomalyDetectionResult,
    AnomalyResult,
    AnomalyType,
    MetricSignal,
    MetricsAnalysisInput,
    Severity,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.anomaly_detector")


class AnomalyDetector:
    """Anomaly type classifier for metric timeseries.

    Classifies anomalous metrics into spike, sustained, or oscillating
    patterns based on their timeseries shape.

    Also assigns severity using Algorithm 6 (zscore + deviation + threshold).

    Args:
        config: Agent configuration with statistical thresholds.

    Example::

        detector = AnomalyDetector(MetricsAgentConfig())
        result = detector.detect(aggregation_result, input_data)
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def detect(
        self,
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> AnomalyDetectionResult:
        """Detect anomaly types for all anomalous metrics.

        Args:
            aggregation: Aggregated metric signals.
            input_data: Original input for timeseries access.
            correlation_id: Request correlation ID.

        Returns:
            AnomalyDetectionResult with typed anomalies.
        """
        start = time.perf_counter()
        anomalies: List[AnomalyResult] = []

        for signal in aggregation.metric_signals:
            if not signal.is_anomalous:
                continue

            timeseries = input_data.metrics.get(signal.metric_name, [])
            anomaly_type = self.classify_anomaly_type(
                timeseries=timeseries,
                signal=signal,
            )
            severity = self.classify_severity(signal)
            reasoning = self._build_reasoning(signal, anomaly_type)

            anomalies.append(AnomalyResult(
                metric_name=signal.metric_name,
                anomaly_type=anomaly_type,
                severity=severity,
                reasoning=reasoning,
            ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Anomaly detection completed: "
            f"{len(anomalies)} anomalies in {elapsed_ms:.2f}ms",
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

    # ── Algorithm 3: Anomaly Type Classification ────────────────

    def classify_anomaly_type(
        self,
        timeseries: List[float],
        signal: MetricSignal,
    ) -> AnomalyType:
        """Classify the type of anomaly based on timeseries pattern.

        Rules:
            Spike: growth_rate > 50% AND current > mean + 2σ
            Sustained: last 3 values ALL > mean + 2σ
            Oscillating: alternating above/below threshold in last 3
            None: default

        Args:
            timeseries: Full timeseries for the metric.
            signal: Pre-computed metric signal.

        Returns:
            AnomalyType enum value.

        Example::

            >>> detector.classify_anomaly_type(
            ...     [45.2, 48.1, 52.3, 87.6, 95.2],
            ...     signal,
            ... )
            AnomalyType.SPIKE  # if growth > 50% and above 2σ
        """
        if len(timeseries) < 2:
            return AnomalyType.NONE

        mean = signal.baseline_mean
        stddev = signal.baseline_stddev
        sustained_sigma = self._config.statistical.sustained_sigma
        threshold_2sigma = mean + sustained_sigma * stddev
        spike_pct = self._config.statistical.spike_threshold_percent

        current = timeseries[-1]

        # Spike: sudden jump above threshold
        if signal.growth_rate > spike_pct and current > threshold_2sigma:
            return AnomalyType.SPIKE

        # Sustained: last 3 values all elevated
        if len(timeseries) >= 3:
            last_3 = timeseries[-3:]
            if all(v > threshold_2sigma for v in last_3):
                return AnomalyType.SUSTAINED

            # Oscillating: alternating pattern
            if len(last_3) == 3:
                above = [v > threshold_2sigma for v in last_3]
                if above[0] and not above[1] and above[2]:
                    return AnomalyType.OSCILLATING
                if not above[0] and above[1] and not above[2]:
                    return AnomalyType.OSCILLATING

        return AnomalyType.NONE

    # ── Algorithm 6: Severity Classification ────────────────────

    @staticmethod
    def classify_severity(signal: MetricSignal) -> Severity:
        """Classify severity based on zscore, deviation, and threshold.

        Rules:
            zscore > 5.0 OR deviation > 100% → critical
            zscore > 4.0 OR (zscore > 3.0 AND threshold_breached) → high
            3.0 < zscore ≤ 4.0 → medium
            2.0 < zscore ≤ 3.0 → low
            else → info

        Args:
            signal: Pre-computed metric signal.

        Returns:
            Severity enum value.

        Example::

            >>> AnomalyDetector.classify_severity(signal_with_zscore_4_52)
            Severity.HIGH
        """
        z = abs(signal.zscore)
        dev = abs(signal.deviation_percent)

        if z > 5.0 or dev > 100.0:
            return Severity.CRITICAL
        elif z > 4.0 or (z > 3.0 and signal.threshold_breached):
            return Severity.HIGH
        elif z > 3.0:
            return Severity.MEDIUM
        elif z > 2.0:
            return Severity.LOW
        else:
            return Severity.INFO

    # ── Reasoning Builder ───────────────────────────────────────

    @staticmethod
    def _build_reasoning(
        signal: MetricSignal,
        anomaly_type: AnomalyType,
    ) -> str:
        """Build human-readable reasoning for an anomaly.

        Args:
            signal: Pre-computed metric signal.
            anomaly_type: Classified anomaly type.

        Returns:
            Reasoning string.
        """
        parts = [
            f"{signal.metric_name} at {signal.current_value}"
        ]

        if signal.baseline_stddev > 0:
            parts.append(
                f"({abs(signal.zscore):.1f}σ above baseline of "
                f"{signal.baseline_mean})"
            )

        if signal.threshold_breached:
            threshold = signal.baseline_mean  # approximate
            parts.append(f"breached threshold")

        if anomaly_type == AnomalyType.SPIKE:
            parts.append(
                f"sudden {signal.growth_rate:.1f}% spike from previous"
            )
        elif anomaly_type == AnomalyType.SUSTAINED:
            parts.append("sustained elevation over last 3 periods")
        elif anomaly_type == AnomalyType.OSCILLATING:
            parts.append("oscillating pattern detected")

        return ", ".join(parts)
