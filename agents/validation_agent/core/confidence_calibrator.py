"""
File: core/confidence_calibrator.py
Purpose: Calculate calibration error between predicted confidence and actual accuracy.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 2: Calibration curves — bin confidences and track
actual accuracy per bin. Perfect calibration = confidence 0.9
means 90% of predictions correct.
"""

from __future__ import annotations

from typing import List, Tuple

from agents.validation_agent.schema import CalibrationBin


def calculate_calibration(
    confidence: float,
    accuracy: float,
    history: List[Tuple[float, float]] | None = None,
    num_bins: int = 10,
) -> Tuple[float, List[CalibrationBin]]:
    """Calculate calibration error and calibration curve.

    The calibration error is |predicted_confidence - actual_accuracy|.
    When history is provided, builds a calibration curve across bins.

    Args:
        confidence: Predicted confidence for current verdict.
        accuracy: Actual accuracy score for current verdict.
        history: List of (confidence, accuracy) tuples from past predictions.
        num_bins: Number of bins for the calibration curve.

    Returns:
        Tuple of (calibration_error, calibration_curve).
    """
    # Current sample calibration error
    confidence = max(0.0, min(1.0, confidence))
    accuracy = max(0.0, min(1.0, accuracy))
    calibration_error = abs(confidence - accuracy)

    # Build calibration curve from history + current sample
    all_samples: List[Tuple[float, float]] = []
    if history:
        for conf, acc in history:
            all_samples.append((
                max(0.0, min(1.0, conf)),
                max(0.0, min(1.0, acc)),
            ))
    all_samples.append((confidence, accuracy))

    # Create bins
    bin_width = 1.0 / num_bins
    bins: List[CalibrationBin] = []

    for i in range(num_bins):
        bin_start = round(i * bin_width, 4)
        bin_end = round((i + 1) * bin_width, 4)

        # Collect samples in this bin
        bin_confidences: List[float] = []
        bin_accuracies: List[float] = []

        for conf, acc in all_samples:
            if bin_start <= conf < bin_end or (
                i == num_bins - 1 and conf == bin_end
            ):
                bin_confidences.append(conf)
                bin_accuracies.append(acc)

        count = len(bin_confidences)
        avg_conf = (
            sum(bin_confidences) / count if count > 0 else 0.0
        )
        avg_acc = (
            sum(bin_accuracies) / count if count > 0 else 0.0
        )

        bins.append(CalibrationBin(
            bin_start=bin_start,
            bin_end=bin_end,
            avg_confidence=round(avg_conf, 4),
            avg_accuracy=round(avg_acc, 4),
            count=count,
        ))

    # If we have enough history, compute mean calibration error
    # across populated bins for a more robust estimate
    if history and len(all_samples) > 1:
        populated_bins = [b for b in bins if b.count > 0]
        if populated_bins:
            total_error = sum(
                abs(b.avg_confidence - b.avg_accuracy) * b.count
                for b in populated_bins
            )
            total_count = sum(b.count for b in populated_bins)
            calibration_error = total_error / total_count if total_count else 0.0

    return round(calibration_error, 4), bins
