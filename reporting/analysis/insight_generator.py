"""Insight generator — rule-based insights from incident data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class Insight:
    """A single generated insight."""

    category: str  # "performance" | "cost" | "reliability" | "accuracy"
    severity: str  # "info" | "warning" | "critical"
    title: str
    description: str
    recommendation: str


class InsightGenerator:
    """Generate human-readable insights from analysis results.

    All rules are deterministic; no LLM calls are made.
    """

    def generate(
        self,
        *,
        avg_duration: float = 0.0,
        target_duration: float = 10.0,
        total_cost: float = 0.0,
        accuracy_rate: float = 0.0,
        accuracy_threshold: float = 0.7,
        slo_compliance: float = 1.0,
        anomaly_count: int = 0,
        outlier_count: int = 0,
        trend_direction: str = "stable",
        mttr_minutes: float = 0.0,
        mttr_target: float = 15.0,
        common_root_causes: Optional[List[tuple]] = None,
    ) -> List[Insight]:
        """Generate insights based on aggregated metrics.

        Args:
            avg_duration: Average pipeline duration in seconds.
            target_duration: Target duration SLO in seconds.
            total_cost: Total LLM cost.
            accuracy_rate: Root-cause accuracy rate [0, 1].
            accuracy_threshold: Minimum acceptable accuracy.
            slo_compliance: SLO compliance ratio [0, 1].
            anomaly_count: Number of performance anomalies.
            outlier_count: Number of cost outliers.
            trend_direction: Duration trend direction.
            mttr_minutes: Mean time to resolve in minutes.
            mttr_target: MTTR target in minutes.
            common_root_causes: Top root causes list.

        Returns:
            List of :class:`Insight` objects.
        """
        insights: List[Insight] = []

        # --- Performance insights ---
        if avg_duration > target_duration:
            pct_over = ((avg_duration - target_duration) / target_duration) * 100
            insights.append(Insight(
                category="performance",
                severity="warning" if pct_over < 50 else "critical",
                title="Pipeline duration exceeds target",
                description=(
                    f"Average duration {avg_duration:.1f}s is "
                    f"{pct_over:.0f}% above the {target_duration:.0f}s target."
                ),
                recommendation="Investigate slow agents and consider parallelization.",
            ))

        if anomaly_count > 0:
            insights.append(Insight(
                category="performance",
                severity="warning",
                title="Performance anomalies detected",
                description=f"{anomaly_count} agent(s) have anomalous latencies.",
                recommendation="Review anomalous agents for regressions.",
            ))

        if trend_direction == "increasing":
            insights.append(Insight(
                category="performance",
                severity="info",
                title="Duration trend is increasing",
                description="Pipeline duration is trending upward over recent runs.",
                recommendation="Monitor closely; may indicate degradation.",
            ))

        # --- Cost insights ---
        if outlier_count > 0:
            insights.append(Insight(
                category="cost",
                severity="warning",
                title="Cost outliers detected",
                description=f"{outlier_count} run(s) had abnormally high costs.",
                recommendation="Check for excessive token usage or retries.",
            ))

        if total_cost > 0.10:
            insights.append(Insight(
                category="cost",
                severity="info",
                title="Elevated total cost",
                description=f"Total cost is ${total_cost:.4f}.",
                recommendation="Review per-agent token usage for optimization.",
            ))

        # --- Accuracy insights ---
        if accuracy_rate < accuracy_threshold:
            gap = accuracy_threshold - accuracy_rate
            insights.append(Insight(
                category="accuracy",
                severity="critical" if gap > 0.2 else "warning",
                title="Root-cause accuracy below threshold",
                description=(
                    f"Accuracy rate {accuracy_rate:.0%} is below the "
                    f"{accuracy_threshold:.0%} threshold."
                ),
                recommendation="Review hypothesis and validation agent prompts.",
            ))

        # --- Reliability insights ---
        if slo_compliance < 1.0:
            insights.append(Insight(
                category="reliability",
                severity="warning" if slo_compliance >= 0.8 else "critical",
                title="SLO compliance below 100%",
                description=f"SLO compliance is {slo_compliance:.0%}.",
                recommendation="Investigate incidents that breached SLOs.",
            ))

        if mttr_minutes > mttr_target:
            insights.append(Insight(
                category="reliability",
                severity="warning",
                title="MTTR exceeds target",
                description=(
                    f"Mean time to resolve {mttr_minutes:.1f} min exceeds "
                    f"the {mttr_target:.0f} min target."
                ),
                recommendation="Streamline the resolution workflow.",
            ))

        # --- Root-cause pattern ---
        if common_root_causes and len(common_root_causes) >= 1:
            top_cause, top_count = common_root_causes[0]
            if top_count >= 3:
                insights.append(Insight(
                    category="reliability",
                    severity="info",
                    title="Recurring root cause detected",
                    description=(
                        f"'{top_cause}' has occurred {top_count} times."
                    ),
                    recommendation="Consider proactive fixes for this failure mode.",
                ))

        return insights
