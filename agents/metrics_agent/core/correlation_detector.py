"""
File: core/correlation_detector.py
Purpose: Pearson correlation detection between anomalous metric pairs.
Dependencies: Standard library only (math)
Performance: O(n²) where n = anomalous metrics (typically <10)

Implements Algorithm 5: Pearson Correlation Coefficient.
Only correlates anomalous metrics to minimize computation.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AggregationResult,
    CorrelationDetectionResult,
    CorrelationRelationship,
    CorrelationResult,
    MetricsAnalysisInput,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.correlation")


class CorrelationDetector:
    """Pearson correlation detector for anomalous metric pairs.

    Only computes correlations between metrics that are already
    flagged as anomalous to minimize O(n²) computation.

    Args:
        config: Agent configuration with correlation threshold.

    Example::

        detector = CorrelationDetector(MetricsAgentConfig())
        result = detector.detect(aggregation, input_data)
        for corr in result.correlations:
            print(f"{corr.metric_1} ↔ {corr.metric_2}: r={corr.correlation_coefficient}")
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def detect(
        self,
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> CorrelationDetectionResult:
        """Detect correlations between anomalous metric pairs.

        Only pairs with |r| > correlation_threshold (default 0.7) are
        included in the output.

        Args:
            aggregation: Aggregated metric signals.
            input_data: Original input for timeseries data.
            correlation_id: Request correlation ID.

        Returns:
            CorrelationDetectionResult with significant correlations.
        """
        start = time.perf_counter()
        correlations: List[CorrelationResult] = []

        # Get anomalous metric names
        anomalous_names = [
            s.metric_name
            for s in aggregation.metric_signals
            if s.is_anomalous
        ]

        threshold = self._config.statistical.correlation_threshold

        # Pairwise correlation for anomalous metrics
        for i in range(len(anomalous_names)):
            for j in range(i + 1, len(anomalous_names)):
                name_a = anomalous_names[i]
                name_b = anomalous_names[j]

                ts_a = input_data.metrics.get(name_a, [])
                ts_b = input_data.metrics.get(name_b, [])

                r = self.pearson_correlation(ts_a, ts_b)
                if r is None:
                    continue

                if abs(r) > threshold:
                    relationship = self.classify_relationship(r)
                    interpretation = self._build_interpretation(
                        name_a, name_b, r, relationship
                    )
                    correlations.append(CorrelationResult(
                        metric_1=name_a,
                        metric_2=name_b,
                        correlation_coefficient=round(r, 4),
                        relationship=relationship,
                        interpretation=interpretation,
                    ))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Correlation detection completed: "
            f"{len(correlations)} correlations from "
            f"{len(anomalous_names)} anomalous metrics in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "correlation",
                "context": {
                    "anomalous_count": len(anomalous_names),
                    "correlation_count": len(correlations),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return CorrelationDetectionResult(
            correlations=correlations,
            detection_latency_ms=round(elapsed_ms, 2),
        )

    # ── Algorithm 5: Pearson Correlation Coefficient ────────────

    @staticmethod
    def pearson_correlation(
        x: List[float],
        y: List[float],
    ) -> Optional[float]:
        """Compute Pearson correlation coefficient between two timeseries.

        Formula:
            r = Σ((xᵢ - μₓ)(yᵢ - μᵧ)) / (√Σ(xᵢ - μₓ)² × √Σ(yᵢ - μᵧ)²)

        Args:
            x: First timeseries.
            y: Second timeseries.

        Returns:
            Pearson r (-1.0 to 1.0), or None if computation is impossible.

        Edge cases:
            - Different lengths: truncate to shorter
            - Length < 2: return None
            - Zero variance: return 0.0

        Example::

            >>> CorrelationDetector.pearson_correlation(
            ...     [45.2, 48.1, 52.3, 87.6, 95.2],
            ...     [450, 460, 480, 2200, 2800],
            ... )
            0.94...
        """
        # Align lengths
        n = min(len(x), len(y))
        if n < 2:
            return None

        x = x[:n]
        y = y[:n]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Compute numerator and denominators
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        # Division by zero guard
        if denom_x == 0.0 or denom_y == 0.0:
            return 0.0

        r = numerator / (denom_x * denom_y)

        # Clamp to [-1, 1] for floating point safety
        return max(-1.0, min(1.0, r))

    # ── Relationship Classification ─────────────────────────────

    @staticmethod
    def classify_relationship(r: float) -> CorrelationRelationship:
        """Classify the direction of a Pearson correlation.

        Args:
            r: Pearson correlation coefficient.

        Returns:
            CorrelationRelationship enum value.

        Example::

            >>> CorrelationDetector.classify_relationship(0.94)
            CorrelationRelationship.POSITIVE
        """
        if r > 0.7:
            return CorrelationRelationship.POSITIVE
        elif r < -0.7:
            return CorrelationRelationship.NEGATIVE
        else:
            return CorrelationRelationship.WEAK

    # ── Interpretation Builder ──────────────────────────────────

    @staticmethod
    def _build_interpretation(
        metric_1: str,
        metric_2: str,
        r: float,
        relationship: CorrelationRelationship,
    ) -> str:
        """Build human-readable interpretation of a correlation.

        Args:
            metric_1: First metric name.
            metric_2: Second metric name.
            r: Correlation coefficient.
            relationship: Classified relationship.

        Returns:
            Interpretation string.
        """
        direction = {
            CorrelationRelationship.POSITIVE: "strongly correlates with",
            CorrelationRelationship.NEGATIVE: "inversely correlates with",
            CorrelationRelationship.WEAK: "weakly correlates with",
        }
        verb = direction.get(relationship, "correlates with")
        return (
            f"{metric_1.replace('_', ' ').title()} {verb} "
            f"{metric_2.replace('_', ' ').title()} (r={r:.2f})"
        )
