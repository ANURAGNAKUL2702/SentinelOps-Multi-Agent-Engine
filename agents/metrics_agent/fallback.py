"""
File: fallback.py
Purpose: Deterministic rule-based classification engine.
Dependencies: Standard library only
Performance: <1ms, $0 cost, 100% reliability

Implements Algorithm 6 (severity) and Algorithm 7 (confidence).
Used as the default classification path and as fallback when LLM fails.
Same output schema as the LLM path for seamless switching.
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AggregationResult,
    AnomalousMetric,
    AnomalyDetectionResult,
    AnomalyResult,
    AnomalyType,
    ClassificationResult,
    CorrelationDetectionResult,
    CorrelationResult,
    MetricsAnalysisInput,
    Severity,
    SystemSummary,
    TrendType,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.fallback")


class RuleEngine:
    """Deterministic rule-based classification engine.

    Provides the same output as the LLM classifier but using
    purely deterministic rules.  Zero network calls, zero cost,
    always available.

    Args:
        config: Agent configuration.

    Example::

        engine = RuleEngine(MetricsAgentConfig())
        result = engine.classify(aggregation, anomalies, correlations, input_data)
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def classify(
        self,
        aggregation: AggregationResult,
        anomaly_result: Optional[AnomalyDetectionResult],
        correlation_result: Optional[CorrelationDetectionResult],
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> ClassificationResult:
        """Run rule-based classification on pre-computed signals.

        Args:
            aggregation: Aggregated metric signals.
            anomaly_result: Anomaly detection results (optional).
            correlation_result: Correlation detection results (optional).
            input_data: Original input.
            correlation_id: Request correlation ID.

        Returns:
            ClassificationResult with source='deterministic'.
        """
        start = time.perf_counter()

        # Build anomaly type lookup
        anomaly_map = {}
        if anomaly_result:
            for a in anomaly_result.anomalies:
                anomaly_map[a.metric_name] = a

        # Build anomalous metrics list
        anomalous_metrics: List[AnomalousMetric] = []
        for signal in aggregation.metric_signals:
            if not signal.is_anomalous:
                continue

            anomaly = anomaly_map.get(signal.metric_name)
            anomaly_type = anomaly.anomaly_type if anomaly else AnomalyType.NONE
            severity = anomaly.severity if anomaly else Severity.INFO
            reasoning = anomaly.reasoning if anomaly else ""

            anomalous_metrics.append(AnomalousMetric(
                metric_name=signal.metric_name,
                current_value=signal.current_value,
                previous_value=signal.previous_value,
                baseline_mean=signal.baseline_mean,
                baseline_stddev=signal.baseline_stddev,
                zscore=signal.zscore,
                deviation_percent=signal.deviation_percent,
                growth_rate=signal.growth_rate,
                is_anomalous=True,
                anomaly_type=anomaly_type,
                severity=severity,
                trend=signal.trend,
                threshold_breached=signal.threshold_breached,
                reasoning=reasoning,
            ))

        # Correlations
        correlations: List[CorrelationResult] = []
        if correlation_result:
            correlations = correlation_result.correlations

        # System summary
        severity_counts = {"critical": 0, "high": 0, "medium": 0}
        for am in anomalous_metrics:
            if am.severity == Severity.CRITICAL:
                severity_counts["critical"] += 1
            elif am.severity == Severity.HIGH:
                severity_counts["high"] += 1
            elif am.severity == Severity.MEDIUM:
                severity_counts["medium"] += 1

        resource_metrics = {"cpu_percent", "memory_percent", "disk_usage_percent"}
        resource_saturated = any(
            am.metric_name in resource_metrics and am.threshold_breached
            for am in anomalous_metrics
        )

        # Cascading: resource + latency both anomalous
        latency_metrics = {
            "api_latency_p50_ms", "api_latency_p95_ms", "api_latency_p99_ms"
        }
        has_resource_anomaly = any(
            am.metric_name in resource_metrics for am in anomalous_metrics
        )
        has_latency_anomaly = any(
            am.metric_name in latency_metrics for am in anomalous_metrics
        )
        cascading = has_resource_anomaly and has_latency_anomaly

        system_summary = SystemSummary(
            total_metrics_analyzed=aggregation.total_metrics_analyzed,
            total_anomalies_detected=len(anomalous_metrics),
            critical_anomalies=severity_counts["critical"],
            high_anomalies=severity_counts["high"],
            medium_anomalies=severity_counts["medium"],
            resource_saturation=resource_saturated,
            cascading_degradation=cascading,
        )

        # Algorithm 7: Confidence score
        confidence, reasoning = self.compute_confidence(
            anomaly_count=len(anomalous_metrics),
            correlation_count=len(correlations),
            max_zscore=max(
                (abs(am.zscore) for am in anomalous_metrics), default=0.0
            ),
            has_threshold_breach=any(
                am.threshold_breached for am in anomalous_metrics
            ),
            has_sudden_spike=any(
                am.trend == TrendType.SUDDEN_SPIKE
                for am in anomalous_metrics
            ),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Rule engine classification completed: "
            f"{len(anomalous_metrics)} anomalies, "
            f"confidence={confidence:.2f}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "classification",
                "context": {
                    "anomaly_count": len(anomalous_metrics),
                    "confidence": confidence,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return ClassificationResult(
            anomalous_metrics=anomalous_metrics,
            correlations=correlations,
            system_summary=system_summary,
            confidence_score=confidence,
            confidence_reasoning=reasoning,
            classification_source="deterministic",
            classification_latency_ms=round(elapsed_ms, 2),
        )

    # ── Algorithm 7: Confidence Score ───────────────────────────

    @staticmethod
    def compute_confidence(
        anomaly_count: int = 0,
        correlation_count: int = 0,
        max_zscore: float = 0.0,
        has_threshold_breach: bool = False,
        has_sudden_spike: bool = False,
    ) -> tuple[float, str]:
        """Compute confidence score using additive rule.

        Rules:
            base = 0.2
            + 0.2 if anomaly_count > 0
            + 0.2 if max_zscore > 4.0
            + 0.2 if correlation_count > 0
            + 0.1 if has_threshold_breach
            + 0.1 if has_sudden_spike
            clamp to [0.0, 1.0]

        Args:
            anomaly_count: Number of anomalous metrics.
            correlation_count: Number of significant correlations.
            max_zscore: Highest absolute z-score.
            has_threshold_breach: Any threshold breached.
            has_sudden_spike: Any sudden spike trend.

        Returns:
            Tuple of (confidence_score, reasoning_string).

        Example::

            >>> RuleEngine.compute_confidence(
            ...     anomaly_count=2, correlation_count=1,
            ...     max_zscore=4.52, has_threshold_breach=True,
            ... )
            (0.9, "High confidence due to: ...")
        """
        confidence = 0.2
        reasons = []

        if anomaly_count > 0:
            confidence += 0.2
            reasons.append(f"{anomaly_count} anomalies detected")

        if max_zscore > 4.0:
            confidence += 0.2
            reasons.append(f"strong anomaly (zscore {max_zscore:.1f})")

        if correlation_count > 0:
            confidence += 0.2
            reasons.append(
                f"{correlation_count} significant correlation(s)"
            )

        if has_threshold_breach:
            confidence += 0.1
            reasons.append("threshold breach")

        if has_sudden_spike:
            confidence += 0.1
            reasons.append("sudden spike detected")

        confidence = min(1.0, confidence)

        if reasons:
            reasoning = f"Confidence {confidence:.1f}: {', '.join(reasons)}"
        else:
            reasoning = "Base confidence only — no anomalies detected"

        return round(confidence, 2), reasoning
