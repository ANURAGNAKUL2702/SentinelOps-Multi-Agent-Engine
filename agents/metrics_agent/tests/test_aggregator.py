"""
File: tests/test_aggregator.py
Purpose: Unit tests for each core algorithm in isolation.

Tests per algorithm:
  1. Z-Score computation
  2. Growth Rate computation
  3. Anomaly Type classification
  4. Threshold Breach detection
  5. Pearson Correlation
  6. Severity Classification
  7. Confidence Score
"""

from __future__ import annotations

import unittest

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.core.anomaly_detector import AnomalyDetector
from agents.metrics_agent.core.correlation_detector import CorrelationDetector
from agents.metrics_agent.core.metric_aggregator import MetricAggregator
from agents.metrics_agent.fallback import RuleEngine
from agents.metrics_agent.schema import (
    AggregationResult,
    AnomalyType,
    BaselineStats,
    CorrelationRelationship,
    MetricSignal,
    MetricsAnalysisInput,
    Severity,
    TrendType,
)


class TestZScoreAlgorithm(unittest.TestCase):
    """Algorithm 1: Z-Score computation."""

    def setUp(self) -> None:
        self.agg = MetricAggregator(MetricsAgentConfig())

    def test_basic_zscore(self) -> None:
        """(95.2 - 50) / 10 = 4.52."""
        z = self.agg.compute_zscore(95.2, 50.0, 10.0)
        self.assertAlmostEqual(z, 4.52, places=1)

    def test_zero_stddev_returns_zero(self) -> None:
        """stddev=0 → z-score=0.0 (no division by zero)."""
        z = self.agg.compute_zscore(100.0, 50.0, 0.0)
        self.assertEqual(z, 0.0)

    def test_negative_zscore(self) -> None:
        """Value below mean → negative z-score."""
        z = self.agg.compute_zscore(20.0, 50.0, 10.0)
        self.assertAlmostEqual(z, -3.0, places=1)

    def test_exact_mean_zscore_zero(self) -> None:
        """Value at mean → z-score=0."""
        z = self.agg.compute_zscore(50.0, 50.0, 10.0)
        self.assertAlmostEqual(z, 0.0, places=5)


class TestGrowthRateAlgorithm(unittest.TestCase):
    """Algorithm 2: Growth Rate computation."""

    def setUp(self) -> None:
        self.agg = MetricAggregator(MetricsAgentConfig())

    def test_basic_growth(self) -> None:
        """((95.2 - 87.6) / 87.6) * 100 ≈ 8.68%."""
        rate = self.agg.compute_growth_rate(95.2, 87.6)
        self.assertAlmostEqual(rate, 8.68, places=1)

    def test_zero_previous_capped(self) -> None:
        """previous=0 → growth_rate=999.9."""
        rate = self.agg.compute_growth_rate(100.0, 0.0)
        self.assertEqual(rate, 999.9)

    def test_decrease(self) -> None:
        """Decrease → negative growth rate."""
        rate = self.agg.compute_growth_rate(50.0, 100.0)
        self.assertAlmostEqual(rate, -50.0, places=1)


class TestTrendClassification(unittest.TestCase):
    """Algorithm 2b: Trend classification from growth rate."""

    def setUp(self) -> None:
        self.agg = MetricAggregator(MetricsAgentConfig())

    def test_sudden_spike(self) -> None:
        """growth > 50% → sudden_spike."""
        self.assertEqual(
            self.agg.classify_trend(60.0), TrendType.SUDDEN_SPIKE
        )

    def test_increasing(self) -> None:
        """20% < growth ≤ 50% → increasing."""
        self.assertEqual(
            self.agg.classify_trend(30.0), TrendType.INCREASING
        )

    def test_stable(self) -> None:
        """-20% ≤ growth ≤ 20% → stable."""
        self.assertEqual(
            self.agg.classify_trend(5.0), TrendType.STABLE
        )

    def test_decreasing(self) -> None:
        """growth < -20% → decreasing."""
        self.assertEqual(
            self.agg.classify_trend(-30.0), TrendType.DECREASING
        )


class TestThresholdBreach(unittest.TestCase):
    """Algorithm 4: Threshold breach detection."""

    def setUp(self) -> None:
        self.agg = MetricAggregator(MetricsAgentConfig())

    def test_cpu_above_threshold(self) -> None:
        """cpu_percent=95 > 80 → breached."""
        breached = self.agg.check_threshold("cpu_percent", 95.0)
        self.assertTrue(breached)

    def test_cpu_below_threshold(self) -> None:
        """cpu_percent=50 < 80 → not breached."""
        breached = self.agg.check_threshold("cpu_percent", 50.0)
        self.assertFalse(breached)

    def test_unknown_metric_not_breached(self) -> None:
        """Unknown metric → always False."""
        breached = self.agg.check_threshold("unknown_metric", 9999.0)
        self.assertFalse(breached)

    def test_memory_threshold(self) -> None:
        """memory_percent=95 > 90 → breached."""
        breached = self.agg.check_threshold("memory_percent", 95.0)
        self.assertTrue(breached)


class TestAnomalyTypeClassification(unittest.TestCase):
    """Algorithm 3: Anomaly type classification."""

    def setUp(self) -> None:
        self.detector = AnomalyDetector(MetricsAgentConfig())

    def test_spike_detection(self) -> None:
        """High growth + value > mean+2σ → spike."""
        signal = MetricSignal(
            metric_name="cpu_percent",
            current_value=95.0,
            previous_value=50.0,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=4.5,
            deviation_percent=90.0,
            growth_rate=90.0,
            is_anomalous=True,
            trend=TrendType.SUDDEN_SPIKE,
            threshold_breached=True,
        )
        ts = [45.0, 48.0, 50.0, 50.0, 95.0]
        anomaly_type = self.detector.classify_anomaly_type(ts, signal)
        self.assertEqual(anomaly_type, AnomalyType.SPIKE)

    def test_sustained_detection(self) -> None:
        """Last 3 all > mean+2σ → sustained."""
        signal = MetricSignal(
            metric_name="cpu_percent",
            current_value=85.0,
            previous_value=82.0,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=3.5,
            deviation_percent=70.0,
            growth_rate=3.6,
            is_anomalous=True,
            trend=TrendType.STABLE,
            threshold_breached=True,
        )
        ts = [50.0, 55.0, 80.0, 82.0, 85.0]
        anomaly_type = self.detector.classify_anomaly_type(ts, signal)
        self.assertEqual(anomaly_type, AnomalyType.SUSTAINED)


class TestSeverityClassification(unittest.TestCase):
    """Algorithm 6: Severity classification."""

    def setUp(self) -> None:
        self.detector = AnomalyDetector(MetricsAgentConfig())

    def test_critical_zscore(self) -> None:
        """zscore > 5 → critical."""
        signal = MetricSignal(
            metric_name="cpu_percent",
            current_value=110.0,
            previous_value=50.0,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=6.0,
            deviation_percent=120.0,
            growth_rate=120.0,
            is_anomalous=True,
            trend=TrendType.SUDDEN_SPIKE,
            threshold_breached=True,
        )
        severity = self.detector.classify_severity(signal)
        self.assertEqual(severity, Severity.CRITICAL)

    def test_high_severity(self) -> None:
        """4 < zscore ≤ 5 → high (or zscore>3 + threshold)."""
        signal = MetricSignal(
            metric_name="cpu_percent",
            current_value=95.0,
            previous_value=50.0,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=4.5,
            deviation_percent=90.0,
            growth_rate=90.0,
            is_anomalous=True,
            trend=TrendType.SUDDEN_SPIKE,
            threshold_breached=True,
        )
        severity = self.detector.classify_severity(signal)
        self.assertEqual(severity, Severity.HIGH)

    def test_medium_severity(self) -> None:
        """3 < zscore ≤ 4 → medium."""
        signal = MetricSignal(
            metric_name="some_metric",
            current_value=80.0,
            previous_value=70.0,
            baseline_mean=50.0,
            baseline_stddev=10.0,
            zscore=3.5,
            deviation_percent=60.0,
            growth_rate=14.0,
            is_anomalous=True,
            trend=TrendType.STABLE,
            threshold_breached=False,
        )
        severity = self.detector.classify_severity(signal)
        self.assertEqual(severity, Severity.MEDIUM)


class TestPearsonCorrelation(unittest.TestCase):
    """Algorithm 5: Pearson correlation."""

    def setUp(self) -> None:
        self.detector = CorrelationDetector(MetricsAgentConfig())

    def test_perfect_positive(self) -> None:
        """x=y → r=1.0."""
        r = self.detector.pearson_correlation(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_perfect_negative(self) -> None:
        """x = -y → r=-1.0."""
        r = self.detector.pearson_correlation(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        )
        self.assertAlmostEqual(r, -1.0, places=5)

    def test_weak_relationship(self) -> None:
        """Random-ish data → |r| < 0.7."""
        r = self.detector.pearson_correlation(
            [1.0, 3.0, 2.0, 5.0, 4.0],
            [5.0, 1.0, 4.0, 2.0, 3.0],
        )
        self.assertLess(abs(r), 0.7)

    def test_short_series_returns_none(self) -> None:
        """Length < 2 → None."""
        r = self.detector.pearson_correlation([1.0], [1.0])
        self.assertIsNone(r)

    def test_zero_variance_returns_zero(self) -> None:
        """Constant series → r=0.0."""
        r = self.detector.pearson_correlation(
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(r, 0.0)

    def test_classify_positive(self) -> None:
        """r > 0.7 → positive."""
        rel = self.detector.classify_relationship(0.85)
        self.assertEqual(rel, CorrelationRelationship.POSITIVE)

    def test_classify_negative(self) -> None:
        """r < -0.7 → negative."""
        rel = self.detector.classify_relationship(-0.85)
        self.assertEqual(rel, CorrelationRelationship.NEGATIVE)

    def test_classify_weak(self) -> None:
        """|r| < 0.7 → weak."""
        rel = self.detector.classify_relationship(0.3)
        self.assertEqual(rel, CorrelationRelationship.WEAK)


class TestConfidenceScore(unittest.TestCase):
    """Algorithm 7: Confidence score computation."""

    def test_base_only(self) -> None:
        """No anomalies → base confidence 0.2."""
        conf, _ = RuleEngine.compute_confidence()
        self.assertAlmostEqual(conf, 0.2, places=2)

    def test_all_factors(self) -> None:
        """All factors → 0.2+0.2+0.2+0.2+0.1+0.1 = 1.0."""
        conf, _ = RuleEngine.compute_confidence(
            anomaly_count=2,
            correlation_count=1,
            max_zscore=5.0,
            has_threshold_breach=True,
            has_sudden_spike=True,
        )
        self.assertAlmostEqual(conf, 1.0, places=2)

    def test_partial_factors(self) -> None:
        """Anomalies + zscore → 0.2 + 0.2 + 0.2 = 0.6."""
        conf, _ = RuleEngine.compute_confidence(
            anomaly_count=1,
            max_zscore=4.5,
        )
        self.assertAlmostEqual(conf, 0.6, places=2)


class TestAggregatorIntegration(unittest.TestCase):
    """Integration test for the full aggregation pipeline."""

    def setUp(self) -> None:
        self.agg = MetricAggregator(MetricsAgentConfig())

    def test_full_aggregation(self) -> None:
        """Full aggregation of multiple metrics."""
        input_data = MetricsAnalysisInput(
            service="test-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [45.0, 50.0, 55.0, 85.0, 95.0],
                "memory_percent": [60.0, 62.0, 61.0, 63.0, 64.0],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
                "memory_percent": BaselineStats(mean=62.0, stddev=5.0),
            },
        )
        result = self.agg.aggregate(input_data)

        self.assertEqual(result.total_metrics_analyzed, 2)
        self.assertEqual(len(result.metric_signals), 2)

        cpu = next(
            s for s in result.metric_signals
            if s.metric_name == "cpu_percent"
        )
        self.assertTrue(cpu.is_anomalous)
        self.assertGreater(abs(cpu.zscore), 3.0)

        mem = next(
            s for s in result.metric_signals
            if s.metric_name == "memory_percent"
        )
        self.assertFalse(mem.is_anomalous)


if __name__ == "__main__":
    unittest.main()
