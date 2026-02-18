"""Accuracy tracking — monitor root-cause accuracy over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class AccuracySummary:
    """Immutable accuracy summary."""

    total_incidents: int
    validated_count: int
    accurate_count: int
    accuracy_rate: float
    avg_confidence: float
    confidence_vs_accuracy: List[Dict[str, float]]
    accuracy_by_severity: Dict[str, float]


class AccuracyTracker:
    """Track the accuracy of root-cause identification.

    An incident is considered *accurate* when its
    ``validation_accuracy`` is ≥ *threshold*.

    Args:
        threshold: Minimum validation accuracy to count as accurate.
            Defaults to ``0.7`` (70 %).
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self._threshold = threshold

    def analyze(
        self,
        incidents: List[Dict[str, Any]],
    ) -> AccuracySummary:
        """Compute accuracy statistics over a list of incident dicts.

        Each dict is expected to have ``validation_accuracy``,
        ``confidence``, and ``severity`` fields.

        Args:
            incidents: List of incident dictionaries (as returned by the
                repository layer).

        Returns:
            A frozen :class:`AccuracySummary`.
        """
        if not incidents:
            return AccuracySummary(
                total_incidents=0,
                validated_count=0,
                accurate_count=0,
                accuracy_rate=0.0,
                avg_confidence=0.0,
                confidence_vs_accuracy=[],
                accuracy_by_severity={},
            )

        validated = [
            i for i in incidents if i.get("validation_accuracy", 0.0) > 0
        ]
        accurate = [
            i for i in validated
            if float(i.get("validation_accuracy", 0.0)) >= self._threshold
        ]

        accuracy_rate = (
            len(accurate) / len(validated) if validated else 0.0
        )

        confidences = [float(i.get("confidence", 0.0)) for i in incidents]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Confidence-vs-accuracy buckets: [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]
        conf_acc: List[Dict[str, float]] = []
        for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            bucket = [
                i for i in validated
                if lo <= float(i.get("confidence", 0.0)) < (hi + (0.01 if hi == 1.0 else 0))
            ]
            if bucket:
                acc = sum(
                    1 for b in bucket
                    if float(b.get("validation_accuracy", 0.0)) >= self._threshold
                ) / len(bucket)
            else:
                acc = 0.0
            conf_acc.append({
                "confidence_range": f"{lo:.0%}-{hi:.0%}",
                "accuracy": round(acc, 4),
                "count": float(len(bucket)),
            })

        # Per-severity accuracy
        severity_groups: Dict[str, List[Dict[str, Any]]] = {}
        for i in validated:
            sev = str(i.get("severity", "unknown"))
            severity_groups.setdefault(sev, []).append(i)

        accuracy_by_severity: Dict[str, float] = {}
        for sev, group in severity_groups.items():
            acc_count = sum(
                1 for g in group
                if float(g.get("validation_accuracy", 0.0)) >= self._threshold
            )
            accuracy_by_severity[sev] = round(acc_count / len(group), 4) if group else 0.0

        return AccuracySummary(
            total_incidents=len(incidents),
            validated_count=len(validated),
            accurate_count=len(accurate),
            accuracy_rate=round(accuracy_rate, 4),
            avg_confidence=round(avg_confidence, 4),
            confidence_vs_accuracy=conf_acc,
            accuracy_by_severity=accuracy_by_severity,
        )
