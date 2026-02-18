"""
File: core/impact_assessor.py
Purpose: Algorithm 8 — Quantify blast radius using dep graph + metrics.
Dependencies: Schema models only.
Performance: <2ms, O(V + E) BFS.

Performs BFS from the root cause service in the dependency graph,
counts reachable services, aggregates metrics severity, and returns
ImpactAssessment.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    ImpactAssessment,
    RootCauseAgentInput,
    Severity,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.impact_assessor")


class ImpactAssessor:
    """Quantifies incident impact using dependency graph BFS + metrics.

    Pipeline::

        root_cause_service → BFS in dep_graph → count reachable
        + aggregate metrics severity → ImpactAssessment

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def assess(
        self,
        input_data: RootCauseAgentInput,
        root_cause_service: str = "",
        correlation_id: str = "",
    ) -> ImpactAssessment:
        """Compute impact assessment.

        Args:
            input_data: Root cause agent input with dep graph.
            root_cause_service: The identified root cause service.
            correlation_id: Request correlation ID.

        Returns:
            ImpactAssessment with blast radius and severity.
        """
        graph = input_data.dependency_findings.impact_graph
        affected = self._bfs_reachable(graph, root_cause_service)

        # Also include dep agent's affected_services
        for svc in input_data.dependency_findings.affected_services:
            affected.add(svc)

        # Remove the root cause service itself from "affected"
        affected.discard(root_cause_service)

        # Severity score from metrics anomalies
        severity_score = self._compute_severity(
            input_data, affected
        )

        # Cascading if blast radius > 2.
        is_cascading = len(affected) > 2

        # Use dep agent's blast radius if available and larger
        blast_radius = max(
            len(affected),
            input_data.dependency_findings.blast_radius,
        )

        logger.debug(
            f"Impact assessed: {len(affected)} affected, "
            f"severity={severity_score:.2f}, cascading={is_cascading}",
            extra={
                "correlation_id": correlation_id,
                "layer": "impact_assessment",
            },
        )

        return ImpactAssessment(
            affected_services=sorted(list(affected)),
            affected_count=len(affected),
            severity_score=round(severity_score, 4),
            blast_radius=blast_radius,
            is_cascading=is_cascading,
        )

    def _bfs_reachable(
        self,
        graph: Dict[str, List[str]],
        start: str,
    ) -> Set[str]:
        """BFS to find all services reachable from start.

        Args:
            graph: Adjacency list (service → downstream services).
            start: Starting service.

        Returns:
            Set of reachable services (excluding start).
        """
        if not start or start not in graph:
            return set()

        visited: Set[str] = set()
        queue: deque = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        visited.discard(start)
        return visited

    def _compute_severity(
        self,
        input_data: RootCauseAgentInput,
        affected: Set[str],
    ) -> float:
        """Aggregate severity score from metrics anomalies.

        Args:
            input_data: Root cause agent input.
            affected: Set of affected services.

        Returns:
            Severity score 0.0-1.0.
        """
        severity_weights = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }

        scores: List[float] = []
        for anom in input_data.metrics_findings.anomalies:
            svc = anom.get("service", "")
            sev = anom.get("severity", "medium")
            if svc in affected or not affected:
                weight = severity_weights.get(sev, 0.5)
                scores.append(weight)

        if not scores:
            # Fall back to blast radius based severity
            if len(affected) >= 5:
                return 0.9
            elif len(affected) >= 3:
                return 0.7
            elif len(affected) >= 1:
                return 0.5
            return 0.2

        return min(1.0, sum(scores) / max(len(scores), 1))
