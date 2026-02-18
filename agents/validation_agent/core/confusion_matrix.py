"""
File: core/confusion_matrix.py
Purpose: Build confusion matrix for root cause classification.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 4: Multi-class confusion matrix.
Rows = actual, columns = predicted. Tracks TP/FP/TN/FN per class.
"""

from __future__ import annotations

from typing import List

from agents.validation_agent.schema import ConfusionMatrixResult


def build_confusion_matrix(
    predicted: str,
    actual: str,
    all_classes: List[str],
) -> ConfusionMatrixResult:
    """Build a confusion matrix for a single prediction.

    For the predicted class:
      - TP if predicted == actual
      - FP if predicted != actual (predicted class got a false positive)
    For other classes:
      - TN for classes that are neither predicted nor actual
      - FN for the actual class if it wasn't predicted

    Args:
        predicted: Predicted class label (root cause).
        actual: Actual class label (ground truth).
        all_classes: All possible class labels.

    Returns:
        ConfusionMatrixResult with TP/FP/TN/FN and full matrix.
    """
    if not all_classes:
        return ConfusionMatrixResult(
            tp=0, fp=0, tn=0, fn=0, matrix=[], classes=[]
        )

    # Ensure both classes are in the list
    classes = list(all_classes)
    for c in (predicted, actual):
        if c and c not in classes:
            classes.append(c)

    n = len(classes)

    # Initialize N×N matrix of zeros
    matrix = [[0] * n for _ in range(n)]

    # Find indices
    pred_idx = classes.index(predicted) if predicted in classes else -1
    actual_idx = classes.index(actual) if actual in classes else -1

    # Populate matrix: matrix[actual_row][predicted_col] += 1
    if actual_idx >= 0 and pred_idx >= 0:
        matrix[actual_idx][pred_idx] += 1

    # Compute TP/FP/TN/FN (binary perspective: "is it this class?")
    correct = predicted == actual and predicted != ""

    if correct:
        tp = 1
        fp = 0
        fn = 0
        tn = n - 1 if n > 1 else 0  # all other classes are TN
    else:
        tp = 0
        fp = 1 if predicted else 0  # predicted class is a false positive
        fn = 1 if actual else 0     # actual class is a false negative
        tn = max(0, n - 2) if (predicted and actual) else max(0, n - 1)

    return ConfusionMatrixResult(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        matrix=matrix,
        classes=classes,
    )


def merge_confusion_matrices(
    matrices: List[ConfusionMatrixResult],
) -> ConfusionMatrixResult:
    """Merge multiple single-prediction confusion matrices.

    Args:
        matrices: List of ConfusionMatrixResult objects.

    Returns:
        Aggregated ConfusionMatrixResult.
    """
    if not matrices:
        return ConfusionMatrixResult(
            tp=0, fp=0, tn=0, fn=0, matrix=[], classes=[]
        )

    # Collect all classes
    all_classes: List[str] = []
    for m in matrices:
        for c in m.classes:
            if c not in all_classes:
                all_classes.append(c)

    n = len(all_classes)
    merged_matrix = [[0] * n for _ in range(n)]

    # Sum up individual matrices
    for m in matrices:
        for row_idx, row_class in enumerate(m.classes):
            for col_idx, col_class in enumerate(m.classes):
                if row_class in all_classes and col_class in all_classes:
                    new_row = all_classes.index(row_class)
                    new_col = all_classes.index(col_class)
                    if row_idx < len(m.matrix) and col_idx < len(m.matrix[row_idx]):
                        merged_matrix[new_row][new_col] += m.matrix[row_idx][col_idx]

    total_tp = sum(m.tp for m in matrices)
    total_fp = sum(m.fp for m in matrices)
    total_tn = sum(m.tn for m in matrices)
    total_fn = sum(m.fn for m in matrices)

    return ConfusionMatrixResult(
        tp=total_tp,
        fp=total_fp,
        tn=total_tn,
        fn=total_fn,
        matrix=merged_matrix,
        classes=all_classes,
    )
