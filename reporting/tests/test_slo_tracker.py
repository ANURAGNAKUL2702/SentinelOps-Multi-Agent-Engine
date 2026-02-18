"""Tests for reporting.analysis.slo_tracker."""

from __future__ import annotations

import pytest

from reporting.analysis.slo_tracker import SLOTracker


class TestSLOTracker:
    def test_all_slos_met(self) -> None:
        tracker = SLOTracker(slo_definitions={
            "latency": 10.0,
            "error_rate": 0.01,
        })
        result = tracker.evaluate({"latency": 5.0, "error_rate": 0.005})
        assert result.met_slos == 2
        assert result.overall_compliance == 1.0

    def test_some_slos_breached(self) -> None:
        tracker = SLOTracker(slo_definitions={
            "latency": 10.0,
            "error_rate": 0.01,
        })
        result = tracker.evaluate({"latency": 15.0, "error_rate": 0.005})
        assert result.met_slos == 1
        assert result.overall_compliance == 0.5

    def test_all_slos_breached(self) -> None:
        tracker = SLOTracker(slo_definitions={
            "latency": 10.0,
            "error_rate": 0.01,
        })
        result = tracker.evaluate({"latency": 15.0, "error_rate": 0.05})
        assert result.met_slos == 0
        assert result.overall_compliance == 0.0

    def test_missing_actual(self) -> None:
        tracker = SLOTracker(slo_definitions={"latency": 10.0})
        result = tracker.evaluate({})  # missing key defaults to 0.0
        assert result.met_slos == 1  # 0.0 <= 10.0

    def test_default_slos(self) -> None:
        tracker = SLOTracker()
        assert len(tracker._slos) >= 1

    def test_result_margins(self) -> None:
        tracker = SLOTracker(slo_definitions={"latency": 10.0})
        result = tracker.evaluate({"latency": 12.0})
        assert result.results[0].margin == pytest.approx(2.0)
        assert not result.results[0].compliant

    def test_exact_boundary(self) -> None:
        tracker = SLOTracker(slo_definitions={"latency": 10.0})
        result = tracker.evaluate({"latency": 10.0})
        assert result.results[0].compliant is True
