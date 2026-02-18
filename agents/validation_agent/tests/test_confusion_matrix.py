"""Tests for confusion_matrix.py — Algorithm 4."""

from __future__ import annotations

import pytest

from agents.validation_agent.core.confusion_matrix import (
    build_confusion_matrix,
    merge_confusion_matrices,
)


class TestBuildConfusionMatrix:
    """Tests for single-prediction confusion matrix."""

    def test_correct_prediction(self) -> None:
        """Correct → TP=1, FP=0, FN=0."""
        cm = build_confusion_matrix(
            "crash", "crash", ["crash", "network", "resource"]
        )
        assert cm.tp == 1
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.tn == 2  # other two classes

    def test_incorrect_prediction(self) -> None:
        """Wrong → TP=0, FP=1, FN=1."""
        cm = build_confusion_matrix(
            "network", "crash", ["crash", "network", "resource"]
        )
        assert cm.tp == 0
        assert cm.fp == 1
        assert cm.fn == 1
        assert cm.tn == 1

    def test_matrix_shape(self) -> None:
        """Matrix should be NxN."""
        classes = ["a", "b", "c", "d", "e"]
        cm = build_confusion_matrix("a", "a", classes)
        assert len(cm.matrix) == 5
        for row in cm.matrix:
            assert len(row) == 5

    def test_matrix_diagonal(self) -> None:
        """Correct prediction → on diagonal."""
        cm = build_confusion_matrix("b", "b", ["a", "b", "c"])
        assert cm.matrix[1][1] == 1  # b is index 1

    def test_matrix_off_diagonal(self) -> None:
        """Wrong prediction → off diagonal."""
        cm = build_confusion_matrix("a", "c", ["a", "b", "c"])
        assert cm.matrix[2][0] == 1  # actual=c(2), predicted=a(0)

    def test_empty_classes(self) -> None:
        cm = build_confusion_matrix("a", "b", [])
        assert cm.tp == 0
        assert cm.matrix == []

    def test_unknown_class_added(self) -> None:
        """Classes not in list get appended."""
        cm = build_confusion_matrix("x", "y", ["a", "b"])
        assert "x" in cm.classes
        assert "y" in cm.classes

    def test_binary_classification(self) -> None:
        """Two classes only."""
        cm = build_confusion_matrix("pos", "pos", ["pos", "neg"])
        assert cm.tp == 1
        assert cm.tn == 1
        assert len(cm.matrix) == 2


class TestMergeConfusionMatrices:
    """Tests for merging multiple confusion matrices."""

    def test_empty_merge(self) -> None:
        cm = merge_confusion_matrices([])
        assert cm.tp == 0
        assert cm.matrix == []

    def test_merge_two(self) -> None:
        cm1 = build_confusion_matrix("a", "a", ["a", "b"])
        cm2 = build_confusion_matrix("b", "a", ["a", "b"])
        merged = merge_confusion_matrices([cm1, cm2])
        assert merged.tp == 1  # only one correct
        assert merged.fp == 1
        assert merged.fn == 1

    def test_merge_preserves_classes(self) -> None:
        cm1 = build_confusion_matrix("a", "a", ["a", "b"])
        cm2 = build_confusion_matrix("c", "c", ["c", "d"])
        merged = merge_confusion_matrices([cm1, cm2])
        assert set(merged.classes) == {"a", "b", "c", "d"}

    def test_merge_all_correct(self) -> None:
        matrices = [
            build_confusion_matrix("a", "a", ["a", "b", "c"]),
            build_confusion_matrix("b", "b", ["a", "b", "c"]),
            build_confusion_matrix("c", "c", ["a", "b", "c"]),
        ]
        merged = merge_confusion_matrices(matrices)
        assert merged.tp == 3
        assert merged.fp == 0
