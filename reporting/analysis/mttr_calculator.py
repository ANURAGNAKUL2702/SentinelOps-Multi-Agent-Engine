"""MTTR / MTTD calculator — percentiles and historical tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class MTTRSummary:
    """Immutable MTTR / MTTD analysis summary."""

    mean_minutes: float
    median_minutes: float
    p95_minutes: float
    p99_minutes: float
    min_minutes: float
    max_minutes: float
    sample_count: int
    within_target: int
    target_compliance: float


class MTTRCalculator:
    """Compute Mean-Time-To-Resolve statistics.

    All durations are supplied in **seconds** and results are
    expressed in **minutes**.

    Args:
        target_minutes: MTTR target used for compliance calculation.
            Defaults to ``15.0`` minutes.
    """

    def __init__(self, target_minutes: float = 15.0) -> None:
        self._target = target_minutes

    def calculate(self, durations_seconds: List[float]) -> MTTRSummary:
        """Calculate MTTR statistics from raw durations.

        Args:
            durations_seconds: List of incident durations in seconds.

        Returns:
            A frozen :class:`MTTRSummary`.
        """
        if not durations_seconds:
            return MTTRSummary(
                mean_minutes=0.0,
                median_minutes=0.0,
                p95_minutes=0.0,
                p99_minutes=0.0,
                min_minutes=0.0,
                max_minutes=0.0,
                sample_count=0,
                within_target=0,
                target_compliance=1.0,
            )

        mins = sorted(d / 60.0 for d in durations_seconds)
        n = len(mins)
        mean_m = sum(mins) / n
        within = sum(1 for m in mins if m <= self._target)

        return MTTRSummary(
            mean_minutes=round(mean_m, 4),
            median_minutes=round(self._percentile(mins, 50), 4),
            p95_minutes=round(self._percentile(mins, 95), 4),
            p99_minutes=round(self._percentile(mins, 99), 4),
            min_minutes=round(mins[0], 4),
            max_minutes=round(mins[-1], 4),
            sample_count=n,
            within_target=within,
            target_compliance=round(within / n, 4),
        )

    @staticmethod
    def _percentile(sorted_vals: List[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])
