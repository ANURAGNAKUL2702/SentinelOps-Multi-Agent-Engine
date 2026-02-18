"""
File: core/metric_aggregator.py
Purpose: Deterministic metric signal extraction — Z-score, trends, thresholds.
Dependencies: Standard library only (math)
Performance: <1ms per 100 metrics, O(n) complexity

Implements:
  Algorithm 1: Z-Score Calculation (3-Sigma Rule)
  Algorithm 2: Growth Rate & Trend Classification
  Algorithm 4: Threshold Breach Detection

All edge cases handled: division by zero, empty timeseries, single values.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AggregationResult,
    MetricSignal,
    MetricsAnalysisInput,
    TrendType,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.aggregator")


class MetricAggregator:
    """Deterministic metric signal extractor.

    Processes each metric timeseries to compute Z-scores, growth rates,
    trends, threshold breaches, and deviation percentages.

    All calculations are deterministic — same input → same output.
    No LLM calls, no network I/O.

    Args:
        config: Agent configuration with thresholds and statistical params.

    Example::

        aggregator = MetricAggregator(MetricsAgentConfig())
        result = aggregator.aggregate(input_data)
        for sig in result.metric_signals:
            print(f"{sig.metric_name}: zscore={sig.zscore}")
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def aggregate(
        self,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> AggregationResult:
        """Extract signals from all metrics in the input.

        Args:
            input_data: Validated metrics analysis input.
            correlation_id: Request correlation ID for logging.

        Returns:
            AggregationResult with one MetricSignal per metric.

        Example::

            result = aggregator.aggregate(input_data)
            assert len(result.metric_signals) == len(input_data.metrics)
        """
        start = time.perf_counter()
        signals: List[MetricSignal] = []

        for metric_name, timeseries in input_data.metrics.items():
            signal = self._process_metric(
                metric_name=metric_name,
                timeseries=timeseries,
                input_data=input_data,
            )
            if signal is not None:
                signals.append(signal)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Metric aggregation completed: "
            f"{len(signals)} metrics in {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "deterministic",
                "context": {
                    "metric_count": len(signals),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return AggregationResult(
            metric_signals=signals,
            total_metrics_analyzed=len(signals),
            aggregation_latency_ms=round(elapsed_ms, 2),
        )

    def _process_metric(
        self,
        metric_name: str,
        timeseries: List[float],
        input_data: MetricsAnalysisInput,
    ) -> Optional[MetricSignal]:
        """Process a single metric timeseries.

        Computes zscore, growth_rate, trend, threshold_breached, deviation.

        Args:
            metric_name: Name of the metric.
            timeseries: Chronological values.
            input_data: Full input for baseline lookup.

        Returns:
            MetricSignal or None if timeseries is empty.
        """
        if not timeseries:
            return None

        current = timeseries[-1]
        previous = timeseries[-2] if len(timeseries) >= 2 else current

        # Baseline lookup
        baseline = input_data.baseline.get(metric_name)
        baseline_mean = baseline.mean if baseline else 0.0
        baseline_stddev = baseline.stddev if baseline else 0.0

        # Algorithm 1: Z-Score
        zscore = self.compute_zscore(current, baseline_mean, baseline_stddev)
        is_anomalous = abs(zscore) > self._config.statistical.zscore_threshold

        # Deviation percentage
        deviation_percent = self.compute_deviation_percent(
            current, baseline_mean
        )

        # Algorithm 2: Growth Rate & Trend
        growth_rate = self.compute_growth_rate(current, previous)
        trend = self.classify_trend(growth_rate)

        # Algorithm 4: Threshold Breach
        threshold_breached = self.check_threshold(metric_name, current)

        return MetricSignal(
            metric_name=metric_name,
            current_value=round(current, 4),
            previous_value=round(previous, 4),
            baseline_mean=round(baseline_mean, 4),
            baseline_stddev=round(baseline_stddev, 4),
            zscore=round(zscore, 2),
            deviation_percent=round(deviation_percent, 2),
            growth_rate=round(growth_rate, 2),
            is_anomalous=is_anomalous,
            trend=trend,
            threshold_breached=threshold_breached,
        )

    # ── Algorithm 1: Z-Score ────────────────────────────────────

    @staticmethod
    def compute_zscore(
        value: float,
        mean: float,
        stddev: float,
    ) -> float:
        """Compute Z-score using the 3-sigma rule.

        Formula: zscore = (value - mean) / stddev

        Args:
            value: Current metric value.
            mean: Historical baseline mean.
            stddev: Historical baseline standard deviation.

        Returns:
            Z-score (0.0 if stddev is 0 to avoid division by zero).

        Example::

            >>> MetricAggregator.compute_zscore(95.2, 50.0, 10.0)
            4.52
            >>> MetricAggregator.compute_zscore(50.0, 50.0, 0.0)
            0.0
        """
        if stddev == 0.0:
            return 0.0
        return (value - mean) / stddev

    # ── Algorithm 2: Growth Rate ────────────────────────────────

    @staticmethod
    def compute_growth_rate(
        current: float,
        previous: float,
    ) -> float:
        """Compute growth rate between two consecutive values.

        Formula: growth_rate = ((current - previous) / previous) × 100

        Args:
            current: Current value (vₙ).
            previous: Previous value (vₙ₋₁).

        Returns:
            Growth rate as a percentage.
            Returns 999.9 if previous=0 and current>0 (maximum spike).
            Returns 0.0 if both are 0.

        Example::

            >>> MetricAggregator.compute_growth_rate(95.2, 87.6)
            8.67...
            >>> MetricAggregator.compute_growth_rate(50.0, 0.0)
            999.9
        """
        if previous == 0.0:
            if current > 0.0:
                return 999.9
            return 0.0
        return ((current - previous) / abs(previous)) * 100

    # ── Algorithm 2 (cont): Trend Classification ────────────────

    def classify_trend(self, growth_rate: float) -> TrendType:
        """Classify trend based on growth rate.

        Classification rules:
            growth_rate > 50%  → sudden_spike
            20% ≤ growth_rate ≤ 50% → increasing
            -20% < growth_rate < 20% → stable
            growth_rate ≤ -20% → decreasing

        Args:
            growth_rate: Growth rate percentage.

        Returns:
            TrendType enum value.

        Example::

            >>> aggregator.classify_trend(63.79)
            TrendType.SUDDEN_SPIKE
        """
        spike = self._config.statistical.spike_threshold_percent
        inc = self._config.statistical.increasing_threshold_percent

        if growth_rate > spike:
            return TrendType.SUDDEN_SPIKE
        elif growth_rate >= inc:
            return TrendType.INCREASING
        elif growth_rate <= -inc:
            return TrendType.DECREASING
        else:
            return TrendType.STABLE

    # ── Algorithm 4: Threshold Breach ───────────────────────────

    def check_threshold(
        self,
        metric_name: str,
        current_value: float,
    ) -> bool:
        """Check if a metric exceeds its configured threshold.

        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.

        Returns:
            True if the threshold is breached.

        Example::

            >>> aggregator.check_threshold("cpu_percent", 95.2)
            True
            >>> aggregator.check_threshold("cpu_percent", 70.0)
            False
        """
        threshold = self._config.thresholds.get_threshold(metric_name)
        if threshold is None or threshold == 0.0:
            return False
        return current_value > threshold

    # ── Deviation Percentage ────────────────────────────────────

    @staticmethod
    def compute_deviation_percent(
        current: float,
        baseline_mean: float,
    ) -> float:
        """Compute how far current is from baseline as a percentage.

        Formula: ((current - mean) / mean) × 100

        Args:
            current: Current metric value.
            baseline_mean: Historical baseline mean.

        Returns:
            Deviation as a percentage (can be negative).
            Returns 0.0 if baseline_mean is 0.

        Example::

            >>> MetricAggregator.compute_deviation_percent(95.2, 50.0)
            90.4
        """
        if baseline_mean == 0.0:
            return 0.0
        return ((current - baseline_mean) / abs(baseline_mean)) * 100
