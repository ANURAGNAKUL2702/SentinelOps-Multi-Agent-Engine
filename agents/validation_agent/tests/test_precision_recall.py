"""Tests for precision_recall.py — Algorithm 3."""

from __future__ import annotations

import pytest

from agents.validation_agent.core.precision_recall import (
    calculate_batch_precision_recall,
    calculate_precision_recall,
)


class TestCalculatePrecisionRecall:
    """Tests for single-sample precision/recall/F1."""

    def test_true_positive(self) -> None:
        """Correct with high confidence → P=R=F1=1.0."""
        p, r, f1 = calculate_precision_recall(True, 0.9)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_false_positive(self) -> None:
        """Wrong with high confidence → all 0."""
        p, r, f1 = calculate_precision_recall(False, 0.9)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_correct_low_confidence(self) -> None:
        """Correct but low confidence → underconfident."""
        p, r, f1 = calculate_precision_recall(True, 0.3)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_wrong_low_confidence(self) -> None:
        """Wrong and low confidence → correctly uncertain."""
        p, r, f1 = calculate_precision_recall(False, 0.3)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_at_threshold(self) -> None:
        """Exactly at threshold = high confidence."""
        p, r, f1 = calculate_precision_recall(True, 0.7)
        assert p == 1.0
        assert r == 1.0

    def test_custom_threshold(self) -> None:
        """Custom threshold changes classification."""
        # Below higher threshold
        p, r, f1 = calculate_precision_recall(True, 0.7, threshold=0.8)
        assert p == 0.0  # Below threshold → not a positive

        # Above lower threshold
        p, r, f1 = calculate_precision_recall(True, 0.5, threshold=0.5)
        assert p == 1.0

    def test_threshold_variations(self) -> None:
        """Varying thresholds affect classification."""
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            p, r, f1 = calculate_precision_recall(True, 0.85, threshold)
            if 0.85 >= threshold:
                assert p == 1.0
            else:
                assert p == 0.0


class TestBatchPrecisionRecall:
    """Tests for batch precision/recall/F1."""

    def test_empty_batch(self) -> None:
        p, r, f1 = calculate_batch_precision_recall([])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_all_correct_high_confidence(self) -> None:
        """Perfect batch → P=R=F1=1.0."""
        samples = [(True, 0.9), (True, 0.8), (True, 0.95)]
        p, r, f1 = calculate_batch_precision_recall(samples)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_all_wrong_high_confidence(self) -> None:
        """All wrong with high confidence → P=0, no TPs."""
        samples = [(False, 0.9), (False, 0.8)]
        p, r, f1 = calculate_batch_precision_recall(samples)
        assert p == 0.0
        assert f1 == 0.0

    def test_mixed_batch(self) -> None:
        """Mixed results: TP=2, FP=1, FN=1."""
        samples = [
            (True, 0.9),   # TP
            (True, 0.8),   # TP
            (False, 0.85), # FP
            (True, 0.3),   # FN (correct but low confidence)
        ]
        p, r, f1 = calculate_batch_precision_recall(samples)
        # P = 2/(2+1) = 0.6667, R = 2/(2+1) = 0.6667
        assert abs(p - 0.6667) < 0.01
        assert abs(r - 0.6667) < 0.01

    def test_high_precision_low_recall(self) -> None:
        """Many correct-but-low-confidence → low recall."""
        samples = [
            (True, 0.9),   # TP
            (True, 0.3),   # FN
            (True, 0.2),   # FN
            (True, 0.1),   # FN
        ]
        p, r, f1 = calculate_batch_precision_recall(samples)
        assert p == 1.0
        assert r == 0.25

    def test_division_by_zero(self) -> None:
        """No positives predicted or actual → safe 0.0."""
        samples = [(False, 0.3), (False, 0.2)]
        p, r, f1 = calculate_batch_precision_recall(samples)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0
