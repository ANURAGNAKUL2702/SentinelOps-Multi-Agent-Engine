"""SLO tracker — monitor compliance against service-level objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class SLOResult:
    """Result for a single SLO check."""

    name: str
    target: float
    actual: float
    compliant: bool
    margin: float  # actual - target


@dataclass(frozen=True)
class SLOSummary:
    """Immutable SLO compliance summary."""

    results: List[SLOResult]
    overall_compliance: float  # fraction of SLOs met
    total_slos: int
    met_slos: int


class SLOTracker:
    """Track whether operational metrics meet pre-defined SLO targets.

    Args:
        slo_definitions: Mapping from SLO name to target value.
            Each target is compared against the actual value; the SLO
            is met when the actual value is **at or below** the target
            (i.e. lower-is-better semantics, suitable for latencies,
            error rates, etc.).  For ratio metrics where higher-is-better,
            negate both target and actual before passing.
    """

    def __init__(
        self,
        slo_definitions: Optional[Dict[str, float]] = None,
    ) -> None:
        self._slos: Dict[str, float] = slo_definitions or {
            "resolution_time_minutes": 30.0,
            "detection_time_minutes": 5.0,
            "accuracy_rate": -0.70,  # negated → higher is better
            "pipeline_duration_seconds": 10.0,
        }

    def evaluate(
        self,
        actuals: Dict[str, float],
    ) -> SLOSummary:
        """Evaluate actual metrics against SLO targets.

        Args:
            actuals: Dictionary mapping SLO name → measured value.
                Keys must match the SLO definitions.

        Returns:
            A frozen :class:`SLOSummary`.
        """
        results: List[SLOResult] = []
        met = 0
        for name, target in self._slos.items():
            actual = actuals.get(name, 0.0)
            compliant = actual <= target
            results.append(
                SLOResult(
                    name=name,
                    target=target,
                    actual=actual,
                    compliant=compliant,
                    margin=round(actual - target, 6),
                )
            )
            if compliant:
                met += 1

        total = len(results)
        return SLOSummary(
            results=results,
            overall_compliance=round(met / total, 4) if total else 1.0,
            total_slos=total,
            met_slos=met,
        )

