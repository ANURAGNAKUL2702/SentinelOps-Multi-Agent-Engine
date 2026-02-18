"""Tests for reporting.analysis.mttr_calculator."""

from __future__ import annotations

import pytest

from reporting.analysis.mttr_calculator import MTTRCalculator


@pytest.fixture
def calculator() -> MTTRCalculator:
    return MTTRCalculator(target_minutes=15.0)


class TestMTTRCalculator:
    def test_empty(self, calculator: MTTRCalculator) -> None:
        result = calculator.calculate([])
        assert result.mean_minutes == 0.0
        assert result.sample_count == 0
        assert result.target_compliance == 1.0

    def test_single_duration(self, calculator: MTTRCalculator) -> None:
        result = calculator.calculate([600.0])  # 10 min
        assert result.mean_minutes == pytest.approx(10.0)
        assert result.median_minutes == pytest.approx(10.0)
        assert result.sample_count == 1
        assert result.within_target == 1

    def test_multiple_durations(self, calculator: MTTRCalculator) -> None:
        durations = [300.0, 600.0, 900.0]  # 5, 10, 15 min
        result = calculator.calculate(durations)
        assert result.mean_minutes == pytest.approx(10.0)
        assert result.median_minutes == pytest.approx(10.0)
        assert result.sample_count == 3

    def test_all_within_target(self, calculator: MTTRCalculator) -> None:
        durations = [60.0, 120.0, 300.0]  # 1, 2, 5 min
        result = calculator.calculate(durations)
        assert result.within_target == 3
        assert result.target_compliance == 1.0

    def test_some_exceed_target(self, calculator: MTTRCalculator) -> None:
        durations = [60.0, 600.0, 1200.0]  # 1, 10, 20 min
        result = calculator.calculate(durations)
        assert result.within_target == 2  # 1 min and 10 min within 15
        assert result.target_compliance == pytest.approx(2.0 / 3.0, abs=0.001)

    def test_percentiles(self, calculator: MTTRCalculator) -> None:
        durations = [float(i * 60) for i in range(1, 101)]  # 1-100 min
        result = calculator.calculate(durations)
        assert result.p95_minutes > result.median_minutes
        assert result.p99_minutes > result.p95_minutes

    def test_min_max(self, calculator: MTTRCalculator) -> None:
        durations = [120.0, 600.0, 1800.0]  # 2, 10, 30 min
        result = calculator.calculate(durations)
        assert result.min_minutes == pytest.approx(2.0)
        assert result.max_minutes == pytest.approx(30.0)
