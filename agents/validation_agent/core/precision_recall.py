"""
File: core/precision_recall.py
Purpose: Precision, recall, and F1 score calculation.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 3:
  TP: verdict correct AND high confidence (>= threshold)
  FP: verdict incorrect AND high confidence
  FN: verdict correct AND low confidence (below threshold) — missed opportunities
      OR verdict incorrect AND low confidence
  TN: not applicable in single-sample mode

For single-sample evaluation, returns binary P/R/F1.
For batch evaluation, aggregates across samples.
"""

from __future__ import annotations

from typing import List, Tuple


def calculate_precision_recall(
    verdict_correct: bool,
    confidence: float,
    threshold: float = 0.7,
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 for a single prediction.

    Args:
        verdict_correct: Whether the verdict matches ground truth.
        confidence: Predicted confidence score.
        threshold: Confidence threshold for positive classification.

    Returns:
        Tuple of (precision, recall, f1_score).
    """
    high_confidence = confidence >= threshold

    if verdict_correct and high_confidence:
        # True positive: correct with high confidence
        return 1.0, 1.0, 1.0
    elif not verdict_correct and high_confidence:
        # False positive: wrong but high confidence
        return 0.0, 0.0, 0.0
    elif verdict_correct and not high_confidence:
        # True but low confidence — underconfident
        # Precision undefined (no positives predicted), recall=0
        return 0.0, 0.0, 0.0
    else:
        # Incorrect and low confidence — correctly uncertain
        return 0.0, 0.0, 0.0


def calculate_batch_precision_recall(
    samples: List[Tuple[bool, float]],
    threshold: float = 0.7,
) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1 over a batch of predictions.

    Args:
        samples: List of (verdict_correct, confidence) tuples.
        threshold: Confidence threshold for positive classification.

    Returns:
        Tuple of (precision, recall, f1_score).
    """
    if not samples:
        return 0.0, 0.0, 0.0

    tp = 0
    fp = 0
    fn = 0

    for correct, confidence in samples:
        high_confidence = confidence >= threshold
        if correct and high_confidence:
            tp += 1
        elif not correct and high_confidence:
            fp += 1
        elif correct and not high_confidence:
            fn += 1
        # TN: not correct and low confidence — correctly uncertain

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)
