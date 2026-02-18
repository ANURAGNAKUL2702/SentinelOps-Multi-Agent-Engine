"""Cost analysis — breakdown, outlier detection, trend calculation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class CostSummary:
    """Immutable summary of cost analysis."""

    total_cost: float
    avg_cost: float
    max_cost: float
    min_cost: float
    std_dev: float
    cost_by_agent: Dict[str, float]
    outlier_threshold: float
    outliers: List[Dict[str, Any]]
    cost_per_token: float


class CostAnalyzer:
    """Analyse costs across pipeline runs.

    Args:
        outlier_multiplier: Number of standard deviations above the mean
            that defines an outlier.  Defaults to ``2.0``.
    """

    def __init__(self, outlier_multiplier: float = 2.0) -> None:
        self._outlier_mult = outlier_multiplier

    def analyze(
        self,
        cost_records: List[Dict[str, Any]],
    ) -> CostSummary:
        """Perform cost analysis on a list of cost records.

        Each record is expected to have ``agent_name``, ``cost``, and
        ``tokens_input`` / ``tokens_output`` keys (matching the schema used
        by the repository layer).

        Args:
            cost_records: List of cost record dicts.

        Returns:
            A frozen :class:`CostSummary` data-class.
        """
        if not cost_records:
            return CostSummary(
                total_cost=0.0,
                avg_cost=0.0,
                max_cost=0.0,
                min_cost=0.0,
                std_dev=0.0,
                cost_by_agent={},
                outlier_threshold=0.0,
                outliers=[],
                cost_per_token=0.0,
            )

        costs = [float(r.get("cost", 0.0)) for r in cost_records]
        total = sum(costs)
        avg = total / len(costs)
        std = statistics.pstdev(costs) if len(costs) > 1 else 0.0
        threshold = avg + self._outlier_mult * std

        cost_by_agent: Dict[str, float] = {}
        total_tokens = 0
        for rec in cost_records:
            agent = str(rec.get("agent_name", "unknown"))
            cost_by_agent[agent] = cost_by_agent.get(agent, 0.0) + float(
                rec.get("cost", 0.0)
            )
            total_tokens += int(rec.get("tokens_input", 0)) + int(
                rec.get("tokens_output", 0)
            )

        outliers = [r for r in cost_records if float(r.get("cost", 0.0)) > threshold]

        cost_per_token = total / total_tokens if total_tokens > 0 else 0.0

        return CostSummary(
            total_cost=total,
            avg_cost=avg,
            max_cost=max(costs),
            min_cost=min(costs),
            std_dev=std,
            cost_by_agent=cost_by_agent,
            outlier_threshold=threshold,
            outliers=outliers,
            cost_per_token=cost_per_token,
        )

    def compare_runs(
        self,
        current: CostSummary,
        previous: CostSummary,
    ) -> Dict[str, float]:
        """Compare two cost summaries and return deltas.

        Args:
            current: Current run summary.
            previous: Previous run summary.

        Returns:
            Dictionary with ``total_delta``, ``avg_delta``,
            ``pct_change`` keys.
        """
        total_delta = current.total_cost - previous.total_cost
        pct = (
            (total_delta / previous.total_cost * 100.0)
            if previous.total_cost > 0
            else 0.0
        )
        return {
            "total_delta": total_delta,
            "avg_delta": current.avg_cost - previous.avg_cost,
            "pct_change": pct,
        }
