"""Trend analysis — detect trends in incident metrics over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class TrendResult:
    """Immutable result of trend analysis."""

    metric_name: str
    direction: str  # "increasing" | "decreasing" | "stable"
    slope: float
    r_squared: float
    data_points: int
    values: List[float]
    timestamps: List[str]


class TrendAnalyzer:
    """Detect trends using simple linear regression.

    Args:
        stability_threshold: Absolute slope below which the trend
            is labelled *stable*.  Defaults to ``0.01``.
    """

    def __init__(self, stability_threshold: float = 0.01) -> None:
        self._stability = stability_threshold

    def analyze_metric(
        self,
        values: List[float],
        timestamps: Optional[List[datetime]] = None,
        metric_name: str = "metric",
    ) -> TrendResult:
        """Analyse a single metric's trend.

        Args:
            values: Ordered metric values.
            timestamps: Optional corresponding timestamps.
            metric_name: Human-readable metric label.

        Returns:
            A frozen :class:`TrendResult`.
        """
        if len(values) < 2:
            return TrendResult(
                metric_name=metric_name,
                direction="stable",
                slope=0.0,
                r_squared=0.0,
                data_points=len(values),
                values=values,
                timestamps=[
                    t.isoformat() if t else "" for t in (timestamps or [])
                ],
            )

        xs = list(range(len(values)))
        slope, intercept = self._linreg(xs, values)
        r_sq = self._r_squared(xs, values, slope, intercept)

        if abs(slope) < self._stability:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        ts_strs = [
            t.isoformat() if t else "" for t in (timestamps or [])
        ]

        return TrendResult(
            metric_name=metric_name,
            direction=direction,
            slope=round(slope, 6),
            r_squared=round(r_sq, 4),
            data_points=len(values),
            values=values,
            timestamps=ts_strs,
        )

    def analyze_incidents(
        self,
        incidents: List[Dict[str, Any]],
    ) -> Dict[str, TrendResult]:
        """Analyse trends across a collection of incidents.

        Returns trend results for ``duration``, ``total_cost``,
        ``confidence``, and ``validation_accuracy``.

        Args:
            incidents: List of incident dicts ordered chronologically.

        Returns:
            Dictionary mapping metric name to :class:`TrendResult`.
        """
        metrics = ("duration", "total_cost", "confidence", "validation_accuracy")
        results: Dict[str, TrendResult] = {}
        timestamps = [
            i.get("started_at") or i.get("created_at") for i in incidents
        ]
        for m in metrics:
            vals = [float(i.get(m, 0.0)) for i in incidents]
            results[m] = self.analyze_metric(vals, timestamps, metric_name=m)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float]:
        """Simple OLS linear regression: returns *(slope, intercept)*."""
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)
        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0, (sum_y / n if n else 0.0)
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    @staticmethod
    def _r_squared(
        xs: List[float],
        ys: List[float],
        slope: float,
        intercept: float,
    ) -> float:
        """Coefficient of determination (R²)."""
        n = len(ys)
        if n == 0:
            return 0.0
        mean_y = sum(ys) / n
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        if ss_tot == 0:
            return 1.0
        return 1.0 - ss_res / ss_tot
