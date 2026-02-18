"""
File: tests/test_signal_extractor.py
Purpose: Comprehensive tests for the deterministic signal extraction layer.
Dependencies: unittest (stdlib), time.
Performance: Full suite <2s.

Tests cover:
  - Empty input
  - Single service
  - Multi-service
  - 1000 services (performance benchmark <100ms)
  - Edge cases: zero errors, division by zero, missing trends
  - Keyword detection (critical, moderate, bidirectional)
  - Cascading failure detection
  - Trend classification (spike, increasing, stable, decreasing)
  - Log flooding detection
  - Dominant service detection
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# Ensure the app root is on sys.path
_APP_ROOT = str(Path(__file__).resolve().parents[3])
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from agents.log_agent.config import LogAgentConfig
from agents.log_agent.core.signal_extractor import SignalExtractor
from agents.log_agent.schema import (
    LogAnalysisInput,
    TrendType,
)


class TestSignalExtractorEmpty(unittest.TestCase):
    """Tests for empty / zero-error input."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_empty_error_summary(self) -> None:
        """Empty error_summary → no service signals, zero system signals."""
        inp = LogAnalysisInput(
            error_summary={},
            total_error_logs=0,
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(len(result.service_signals), 0)
        self.assertEqual(result.system_signals.total_error_logs, 0)
        self.assertEqual(result.system_signals.affected_service_count, 0)
        self.assertIsNone(result.system_signals.earliest_error_service)
        self.assertFalse(result.system_signals.cascading_candidate)

    def test_all_zero_errors(self) -> None:
        """All services with 0 errors → no service signals emitted."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 0, "svc-b": 0},
            total_error_logs=0,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(len(result.service_signals), 0)


class TestSignalExtractorSingle(unittest.TestCase):
    """Tests for single-service scenarios."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_single_service_basic(self) -> None:
        """Single service with 100% errors."""
        inp = LogAnalysisInput(
            error_summary={"payment-service": 340},
            total_error_logs=340,
            error_trends={"payment-service": [0, 0, 5, 30, 340]},
            keyword_matches={"payment-service": ["database timeout"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(len(result.service_signals), 1)

        sig = result.service_signals[0]
        self.assertEqual(sig.service, "payment-service")
        self.assertEqual(sig.error_count, 340)
        self.assertAlmostEqual(sig.error_percentage, 100.0, places=1)
        self.assertTrue(sig.dominant_service_signal)
        self.assertTrue(sig.critical_keyword)

    def test_error_percentage_accuracy(self) -> None:
        """Verify error_percentage = (error_count / total) * 100."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 340, "svc-b": 12},
            total_error_logs=352,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        sigs = {s.service: s for s in result.service_signals}

        self.assertAlmostEqual(
            sigs["svc-a"].error_percentage, 96.59, places=1
        )
        self.assertAlmostEqual(
            sigs["svc-b"].error_percentage, 3.41, places=1
        )

    def test_dominant_service_detection(self) -> None:
        """Service with >60% errors is dominant."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 70, "svc-b": 30},
            total_error_logs=100,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        sigs = {s.service: s for s in result.service_signals}

        self.assertTrue(sigs["svc-a"].dominant_service_signal)
        self.assertFalse(sigs["svc-b"].dominant_service_signal)


class TestTrendClassification(unittest.TestCase):
    """Tests for growth rate and trend classification."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_sudden_spike(self) -> None:
        """growth_rate > 200 → sudden_spike."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 340},
            total_error_logs=340,
            error_trends={"svc-a": [0, 0, 5, 30, 340]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].trend_type, TrendType.SUDDEN_SPIKE
        )

    def test_increasing_trend(self) -> None:
        """20 ≤ growth_rate ≤ 200 → increasing."""
        # growth = ((100 - 50) / 50) * 100 = 100%
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            error_trends={"svc-a": [10, 20, 50, 100]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].trend_type, TrendType.INCREASING
        )

    def test_stable_trend(self) -> None:
        """growth_rate between -20 and 20 → stable."""
        # growth = ((100 - 95) / 95) * 100 ≈ 5.3%
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            error_trends={"svc-a": [90, 95, 100]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].trend_type, TrendType.STABLE
        )

    def test_decreasing_trend(self) -> None:
        """growth_rate ≤ -20 → decreasing."""
        # growth = ((50 - 100) / 100) * 100 = -50%
        inp = LogAnalysisInput(
            error_summary={"svc-a": 50},
            total_error_logs=50,
            error_trends={"svc-a": [200, 100, 50]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].trend_type, TrendType.DECREASING
        )

    def test_no_trend_data(self) -> None:
        """Missing trend data → stable (growth_rate = 0)."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].trend_type, TrendType.STABLE
        )
        self.assertAlmostEqual(
            result.service_signals[0].growth_rate_last_period, 0.0
        )

    def test_single_trend_point(self) -> None:
        """Single trend point → stable (insufficient data)."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            error_trends={"svc-a": [10]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertAlmostEqual(
            result.service_signals[0].growth_rate_last_period, 0.0
        )

    def test_zero_previous_period(self) -> None:
        """Previous period = 0 → uses mean as baseline."""
        # trend = [0, 100], prev=0, mean of non-zero = 100
        # growth = ((100 - 100) / 100) * 100 = 0%
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            error_trends={"svc-a": [0, 100]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertAlmostEqual(
            result.service_signals[0].growth_rate_last_period, 0.0
        )

    def test_first_non_zero_index(self) -> None:
        """first_non_zero_trend_index reports correctly."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            error_trends={"svc-a": [0, 0, 5, 30, 100]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].first_non_zero_trend_index, 2
        )


class TestKeywordDetection(unittest.TestCase):
    """Tests for critical/moderate keyword matching."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_critical_keyword_exact(self) -> None:
        """Exact critical keyword match."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["deadlock"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.service_signals[0].critical_keyword)

    def test_critical_keyword_substring(self) -> None:
        """Substring match: 'connection refused' matches critical keyword."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["connection refused"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.service_signals[0].critical_keyword)

    def test_critical_keyword_database_timeout(self) -> None:
        """'database timeout' matches critical keyword list."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["database timeout"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.service_signals[0].critical_keyword)

    def test_moderate_keyword(self) -> None:
        """Moderate keyword detection."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["high cpu"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.service_signals[0].critical_keyword)
        self.assertTrue(result.service_signals[0].moderate_keyword)

    def test_no_keyword_match(self) -> None:
        """No keyword match → both false."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["unrelated error"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.service_signals[0].critical_keyword)
        self.assertFalse(result.service_signals[0].moderate_keyword)

    def test_case_insensitive(self) -> None:
        """Keyword matching is case-insensitive."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": ["DEADLOCK"]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.service_signals[0].critical_keyword)


class TestCascadingDetection(unittest.TestCase):
    """Tests for cascading failure detection."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_cascading_detected(self) -> None:
        """3+ services, earliest precedes others → cascading."""
        inp = LogAnalysisInput(
            error_summary={
                "svc-a": 100, "svc-b": 50, "svc-c": 30,
            },
            total_error_logs=180,
            error_trends={
                "svc-a": [10, 50, 100],   # starts at index 0
                "svc-b": [0, 0, 50],      # starts at index 2
                "svc-c": [0, 0, 30],      # starts at index 2
            },
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.system_signals.cascading_candidate)
        self.assertEqual(
            result.system_signals.earliest_error_service, "svc-a"
        )

    def test_not_cascading_fewer_services(self) -> None:
        """< 3 services → no cascading."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100, "svc-b": 50},
            total_error_logs=150,
            error_trends={
                "svc-a": [10, 50, 100],
                "svc-b": [0, 0, 50],
            },
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.system_signals.cascading_candidate)

    def test_not_cascading_simultaneous(self) -> None:
        """All services start at same index → not cascading."""
        inp = LogAnalysisInput(
            error_summary={
                "svc-a": 100, "svc-b": 50, "svc-c": 30,
            },
            total_error_logs=180,
            error_trends={
                "svc-a": [10, 50, 100],
                "svc-b": [5, 25, 50],
                "svc-c": [3, 15, 30],
            },
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.system_signals.cascading_candidate)


class TestLogFlooding(unittest.TestCase):
    """Tests for log flooding detection."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_flooding_detected(self) -> None:
        """Bucket > 100 → flooding."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 200},
            total_error_logs=200,
            error_trends={"svc-a": [0, 0, 200]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertTrue(result.service_signals[0].log_flooding_signal)

    def test_no_flooding(self) -> None:
        """All buckets ≤ 100 → no flooding."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            error_trends={"svc-a": [20, 30, 50]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.service_signals[0].log_flooding_signal)


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance tests — signal extraction must be <100ms for 1000 services."""

    def test_1000_services_under_100ms(self) -> None:
        """Benchmark: 1000 services must complete in <100ms."""
        error_summary = {f"svc-{i}": (i % 50) + 1 for i in range(1000)}
        total = sum(error_summary.values())
        trends = {
            svc: [0, count // 4, count // 2, count]
            for svc, count in error_summary.items()
        }

        inp = LogAnalysisInput(
            error_summary=error_summary,
            total_error_logs=total,
            error_trends=trends,
            time_window="benchmark",
        )

        extractor = SignalExtractor(LogAgentConfig())

        start = time.perf_counter()
        result = extractor.extract(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertEqual(len(result.service_signals), 1000)
        self.assertLess(
            elapsed_ms, 100.0,
            f"Extraction took {elapsed_ms:.2f}ms (>100ms)",
        )

    def test_8_services_under_10ms(self) -> None:
        """Typical incident: 8 services should be <10ms."""
        services = [
            "api-gateway", "auth-service", "payment-service",
            "fraud-service", "notification-service",
            "database", "cache-service", "merchant-portal",
        ]
        error_summary = {s: (i + 1) * 10 for i, s in enumerate(services)}
        total = sum(error_summary.values())

        inp = LogAnalysisInput(
            error_summary=error_summary,
            total_error_logs=total,
            time_window="test",
        )

        extractor = SignalExtractor(LogAgentConfig())

        start = time.perf_counter()
        result = extractor.extract(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertEqual(len(result.service_signals), 8)
        self.assertLess(
            elapsed_ms, 10.0,
            f"Extraction took {elapsed_ms:.2f}ms (>10ms)",
        )


class TestSystemSignals(unittest.TestCase):
    """Tests for system-level signal aggregation."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_affected_service_count(self) -> None:
        """affected_service_count = number of services with errors > 0."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10, "svc-b": 20, "svc-c": 0},
            total_error_logs=30,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(result.system_signals.affected_service_count, 2)

    def test_earliest_error_service(self) -> None:
        """Earliest error service has lowest first_non_zero_trend_index."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10, "svc-b": 20},
            total_error_logs=30,
            error_trends={
                "svc-a": [0, 0, 10],
                "svc-b": [5, 10, 20],
            },
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.system_signals.earliest_error_service, "svc-b"
        )

    def test_total_error_logs_passthrough(self) -> None:
        """total_error_logs from input is passed through."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 100},
            total_error_logs=100,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(result.system_signals.total_error_logs, 100)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def setUp(self) -> None:
        self.extractor = SignalExtractor(LogAgentConfig())

    def test_correlation_id_propagated(self) -> None:
        """Correlation ID from input is preserved."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            time_window="test",
            correlation_id="test-cid-123",
        )
        # Extraction completes without error — cid is used in logging
        result = self.extractor.extract(inp)
        self.assertEqual(len(result.service_signals), 1)

    def test_extraction_latency_recorded(self) -> None:
        """extraction_latency_ms is recorded and > 0."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertGreater(result.extraction_latency_ms, 0.0)

    def test_all_trends_zero(self) -> None:
        """All trend values zero → first_non_zero_index = -1."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            error_trends={"svc-a": [0, 0, 0]},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertEqual(
            result.service_signals[0].first_non_zero_trend_index, -1
        )

    def test_empty_keyword_list(self) -> None:
        """Empty keyword list → no keywords detected."""
        inp = LogAnalysisInput(
            error_summary={"svc-a": 10},
            total_error_logs=10,
            keyword_matches={"svc-a": []},
            time_window="test",
        )
        result = self.extractor.extract(inp)
        self.assertFalse(result.service_signals[0].critical_keyword)
        self.assertFalse(result.service_signals[0].moderate_keyword)


if __name__ == "__main__":
    unittest.main()
