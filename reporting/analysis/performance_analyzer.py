"""Performance analysis — latency, throughput, anomaly detection."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List

from ..telemetry import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class PerformanceSummary:
    """Immutable performance summary."""

    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    min_latency: float
    std_dev: float
    anomalies: List[Dict[str, Any]]
    agent_latencies: Dict[str, float]
    parallel_speedup: float


class PerformanceAnalyzer:
    """Analyse execution performance of pipeline agents.

    Args:
        z_threshold: Z-score threshold for anomaly detection.
            Values above this are flagged.  Defaults to ``2.0``.
    """

    def __init__(self, z_threshold: float = 2.0) -> None:
        self._z_threshold = z_threshold

    def analyze(
        self,
        agent_executions: List[Dict[str, Any]],
        total_duration: float = 0.0,
    ) -> PerformanceSummary:
        """Analyse a list of agent execution records.

        Each record should contain ``agent_name`` and ``duration`` keys.

        Args:
            agent_executions: Agent execution dicts.
            total_duration: Overall pipeline execution duration.

        Returns:
            A frozen :class:`PerformanceSummary`.
        """
        if not agent_executions:
            return PerformanceSummary(
                avg_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                max_latency=0.0,
                min_latency=0.0,
                std_dev=0.0,
                anomalies=[],
                agent_latencies={},
                parallel_speedup=1.0,
            )

        durations = [float(e.get("duration", 0.0)) for e in agent_executions]
        durations.sort()

        avg = statistics.mean(durations)
        std = statistics.pstdev(durations) if len(durations) > 1 else 0.0

        agent_latencies: Dict[str, float] = {}
        for e in agent_executions:
            agent_latencies[str(e.get("agent_name", "unknown"))] = float(
                e.get("duration", 0.0)
            )

        anomalies = self._detect_anomalies(agent_executions, avg, std)

        sequential_total = sum(durations)
        parallel_speedup = (
            sequential_total / total_duration
            if total_duration > 0
            else 1.0
        )

        return PerformanceSummary(
            avg_latency=avg,
            p50_latency=self._percentile(durations, 50),
            p95_latency=self._percentile(durations, 95),
            p99_latency=self._percentile(durations, 99),
            max_latency=max(durations),
            min_latency=min(durations),
            std_dev=std,
            anomalies=anomalies,
            agent_latencies=agent_latencies,
            parallel_speedup=round(parallel_speedup, 2),
        )

    def _detect_anomalies(
        self,
        executions: List[Dict[str, Any]],
        mean: float,
        std: float,
    ) -> List[Dict[str, Any]]:
        """Flag executions whose duration is *z_threshold* σ above the mean."""
        if std == 0:
            return []
        anomalies: List[Dict[str, Any]] = []
        for e in executions:
            dur = float(e.get("duration", 0.0))
            z = (dur - mean) / std
            if z > self._z_threshold:
                anomalies.append({
                    "agent_name": e.get("agent_name", "unknown"),
                    "duration": dur,
                    "z_score": round(z, 2),
                    "expected_max": round(mean + self._z_threshold * std, 4),
                })
        return anomalies

    @staticmethod
    def _percentile(sorted_vals: List[float], pct: float) -> float:
        """Calculate *pct*-th percentile from a sorted list."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])
