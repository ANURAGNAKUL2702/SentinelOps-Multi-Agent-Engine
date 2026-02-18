"""Tests for reporting.analysis.trend_analyzer."""

from __future__ import annotations

import pytest

from reporting.analysis.trend_analyzer import TrendAnalyzer


@pytest.fixture
def analyzer() -> TrendAnalyzer:
    return TrendAnalyzer(stability_threshold=0.01)


class TestTrendAnalyzer:
    def test_empty_values(self, analyzer: TrendAnalyzer) -> None:
        result = analyzer.analyze_metric([], metric_name="test")
        assert result.direction == "stable"
        assert result.data_points == 0

    def test_single_value(self, analyzer: TrendAnalyzer) -> None:
        result = analyzer.analyze_metric([5.0], metric_name="test")
        assert result.direction == "stable"
        assert result.data_points == 1

    def test_increasing_trend(self, analyzer: TrendAnalyzer) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.analyze_metric(values, metric_name="cost")
        assert result.direction == "increasing"
        assert result.slope > 0
        assert result.r_squared > 0.9

    def test_decreasing_trend(self, analyzer: TrendAnalyzer) -> None:
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = analyzer.analyze_metric(values, metric_name="latency")
        assert result.direction == "decreasing"
        assert result.slope < 0

    def test_stable_trend(self, analyzer: TrendAnalyzer) -> None:
        values = [5.0, 5.001, 4.999, 5.0, 5.002]
        result = analyzer.analyze_metric(values, metric_name="accuracy")
        assert result.direction == "stable"

    def test_analyze_incidents(self, analyzer: TrendAnalyzer) -> None:
        incidents = [
            {"duration": 5.0, "total_cost": 0.01, "confidence": 0.8, "validation_accuracy": 0.9},
            {"duration": 6.0, "total_cost": 0.02, "confidence": 0.85, "validation_accuracy": 0.85},
            {"duration": 7.0, "total_cost": 0.03, "confidence": 0.9, "validation_accuracy": 0.8},
        ]
        results = analyzer.analyze_incidents(incidents)
        assert "duration" in results
        assert "total_cost" in results
        assert results["duration"].direction == "increasing"

    def test_r_squared_perfect_fit(self, analyzer: TrendAnalyzer) -> None:
        values = [2.0, 4.0, 6.0, 8.0]
        result = analyzer.analyze_metric(values)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)
