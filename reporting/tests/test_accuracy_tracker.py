"""Tests for reporting.analysis.accuracy_tracker."""

from __future__ import annotations

import pytest

from reporting.analysis.accuracy_tracker import AccuracyTracker


@pytest.fixture
def tracker() -> AccuracyTracker:
    return AccuracyTracker(threshold=0.7)


class TestAccuracyTracker:
    def test_empty(self, tracker: AccuracyTracker) -> None:
        result = tracker.analyze([])
        assert result.total_incidents == 0
        assert result.accuracy_rate == 0.0

    def test_all_accurate(self, tracker: AccuracyTracker) -> None:
        incidents = [
            {"validation_accuracy": 0.9, "confidence": 0.8, "severity": "P1_HIGH"},
            {"validation_accuracy": 0.85, "confidence": 0.7, "severity": "P2_MEDIUM"},
        ]
        result = tracker.analyze(incidents)
        assert result.accuracy_rate == 1.0
        assert result.accurate_count == 2

    def test_mixed_accuracy(self, tracker: AccuracyTracker) -> None:
        incidents = [
            {"validation_accuracy": 0.9, "confidence": 0.8, "severity": "P1_HIGH"},
            {"validation_accuracy": 0.3, "confidence": 0.5, "severity": "P2_MEDIUM"},
        ]
        result = tracker.analyze(incidents)
        assert result.accuracy_rate == 0.5
        assert result.accurate_count == 1
        assert result.validated_count == 2

    def test_no_validated(self, tracker: AccuracyTracker) -> None:
        incidents = [
            {"validation_accuracy": 0.0, "confidence": 0.5, "severity": "P1_HIGH"},
        ]
        result = tracker.analyze(incidents)
        assert result.validated_count == 0
        assert result.accuracy_rate == 0.0

    def test_accuracy_by_severity(self, tracker: AccuracyTracker) -> None:
        incidents = [
            {"validation_accuracy": 0.9, "confidence": 0.9, "severity": "P0_CRITICAL"},
            {"validation_accuracy": 0.5, "confidence": 0.4, "severity": "P0_CRITICAL"},
            {"validation_accuracy": 0.8, "confidence": 0.7, "severity": "P1_HIGH"},
        ]
        result = tracker.analyze(incidents)
        assert "P0_CRITICAL" in result.accuracy_by_severity
        assert result.accuracy_by_severity["P0_CRITICAL"] == 0.5

    def test_confidence_vs_accuracy(self, tracker: AccuracyTracker) -> None:
        incidents = [
            {"validation_accuracy": 0.9, "confidence": 0.85, "severity": "P1"},
        ]
        result = tracker.analyze(incidents)
        assert len(result.confidence_vs_accuracy) == 4  # 4 buckets
