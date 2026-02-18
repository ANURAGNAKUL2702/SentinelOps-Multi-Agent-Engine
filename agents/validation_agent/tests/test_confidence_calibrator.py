"""Tests for confidence_calibrator.py — Algorithm 2."""

from __future__ import annotations

import pytest

from agents.validation_agent.core.confidence_calibrator import (
    calculate_calibration,
)


class TestCalculateCalibration:
    """Tests for calibration error and curve calculation."""

    def test_perfect_calibration(self) -> None:
        """Confidence matches accuracy exactly."""
        error, bins = calculate_calibration(0.9, 0.9)
        assert error == 0.0
        assert len(bins) == 10

    def test_overconfident(self) -> None:
        """Confidence=0.9 but accuracy=0.3 → large error."""
        error, bins = calculate_calibration(0.9, 0.3)
        assert error == 0.6

    def test_underconfident(self) -> None:
        """Confidence=0.2 but accuracy=0.8 → large error."""
        error, bins = calculate_calibration(0.2, 0.8)
        assert error == 0.6

    def test_empty_history(self) -> None:
        """No history — single sample calibration."""
        error, bins = calculate_calibration(0.5, 0.5, history=None)
        assert error == 0.0
        # Only one bin should have count > 0
        populated = [b for b in bins if b.count > 0]
        assert len(populated) == 1

    def test_with_history(self) -> None:
        """History samples improve calibration estimate."""
        history = [
            (0.9, 0.85),
            (0.8, 0.75),
            (0.7, 0.7),
            (0.5, 0.55),
        ]
        error, bins = calculate_calibration(0.6, 0.6, history=history)
        # With reasonably calibrated history, error should be low
        assert error < 0.15

    def test_single_sample(self) -> None:
        """Single sample: error = |confidence - accuracy|."""
        error, bins = calculate_calibration(0.8, 0.6)
        assert abs(error - 0.2) < 0.001

    def test_bins_cover_range(self) -> None:
        """All bins should cover [0, 1]."""
        _, bins = calculate_calibration(0.5, 0.5)
        assert bins[0].bin_start == 0.0
        assert bins[-1].bin_end == 1.0

    def test_clamping(self) -> None:
        """Values outside [0, 1] should be clamped."""
        error, _ = calculate_calibration(1.5, -0.5)
        # Clamped to (1.0, 0.0) → error = 1.0
        assert error == 1.0

    def test_custom_bins(self) -> None:
        """Custom number of bins."""
        _, bins = calculate_calibration(0.5, 0.5, num_bins=5)
        assert len(bins) == 5
