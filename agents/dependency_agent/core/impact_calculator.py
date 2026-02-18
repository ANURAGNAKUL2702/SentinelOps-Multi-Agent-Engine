"""
File: core/impact_calculator.py
Purpose: Calculate blast radius, criticality scores, and SPOF detection.
Dependencies: Standard library only
Performance: O(V+E) for blast radius BFS, <20ms for 1000-node graphs

Implements:
  Algorithm 3: Blast Radius via BFS (upstream + downstream)
  Algorithm 5: Criticality Score (weighted formula, 0–1 range)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    BlastRadius,
    CriticalPathResult,
    DependencyAnalysisInput,
    DownstreamDependencies,
    GraphData,
    ImpactAnalysisResult,
    ServiceEdge,
    UpstreamDependencies,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.impact_calculator")


class ImpactCalculator:
    """Calculate blast radius and criticality for each service.

    For every service node, performs BFS on the adjacency list
    (downstream) and reverse adjacency list (upstream) to determine
    blast radius. Criticality score blends multiple signals into
    a 0–1 float.

    Args:
        config: Agent configuration with thresholds and weights.

    Example::

        calc = ImpactCalculator(DependencyAgentConfig())
        result = calc.calculate(input_data, graph_data)
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def calculate(
        self,
        input_data: DependencyAnalysisInput,
        graph: GraphData,
        critical_path: Optional[CriticalPathResult] = None,
        correlation_id: str = "",
    ) -> Dict[str, ImpactAnalysisResult]:
        """Calculate impact analysis for every service.

        Args:
            input_data: Raw input with service/edge definitions.
            graph: Pre-built graph adjacency data.
            critical_path: Optional critical path result for scoring.
            correlation_id: Request correlation ID.

        Returns:
            Dict mapping service_name → ImpactAnalysisResult.
        """
        start = time.perf_counter()
        results: Dict[str, ImpactAnalysisResult] = {}

        critical_path_services: Set[str] = set()
        if critical_path and critical_path.path:
            critical_path_services = set(critical_path.path)

        for node in input_data.service_graph.nodes:
            svc = node.service_name
            blast = self._blast_radius(svc, graph)
            upstream = self._upstream_deps(svc, graph)
            downstream = self._downstream_deps(svc, graph)
            score = self._criticality_score(
                svc, graph, blast, critical_path_services
            )
            is_spof = self._is_spof(node, graph)

            results[svc] = ImpactAnalysisResult(
                blast_radius=blast,
                upstream_dependencies=upstream,
                downstream_dependencies=downstream,
                criticality_score=score,
                is_single_point_of_failure=is_spof,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Impact calculation complete: "
            f"services={len(results)}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "impact_calculation",
                "context": {
                    "service_count": len(results),
                    "spof_count": sum(
                        1 for r in results.values()
                        if r.is_single_point_of_failure
                    ),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return results

    # ── Algorithm 3: Blast Radius BFS ───────────────────────────

    def _blast_radius(
        self, service: str, graph: GraphData,
        max_depth: int = 5,
    ) -> BlastRadius:
        """BFS on adjacency list to find all affected downstream services.

        Depth-limited BFS to keep per-service cost bounded at O(branching^depth).

        Args:
            service: Name of the failing service.
            graph: Pre-built graph data.
            max_depth: Maximum BFS depth to explore (default 5).

        Returns:
            BlastRadius with directly and transitively affected services.
        """
        directly_affected: List[str] = list(
            graph.adjacency_list.get(service, [])
        )

        # Depth-limited BFS from direct neighbors for transitive impact
        indirectly_affected: Set[str] = set()
        visited: Set[str] = {service}
        # queue entries: (node, depth)
        queue: deque[tuple[str, int]] = deque()

        for d in directly_affected:
            visited.add(d)
            queue.append((d, 1))

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = graph.adjacency_list.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    indirectly_affected.add(neighbor)
                    queue.append((neighbor, depth + 1))

        total = len(directly_affected) + len(indirectly_affected)

        # Calculate affected request rate
        affected_rate = self._calculate_affected_rate(
            service, graph
        )

        return BlastRadius(
            directly_affected_services=sorted(directly_affected),
            indirectly_affected_services=sorted(indirectly_affected),
            total_affected_count=total,
            affected_request_rate_per_sec=affected_rate,
        )

    def _upstream_deps(
        self, service: str, graph: GraphData,
        max_depth: int = 5,
    ) -> UpstreamDependencies:
        """Find all upstream dependencies using reverse adjacency BFS.

        Upstream = services that depend ON this service (callers).
        Depth-limited to keep per-service cost bounded.

        Args:
            service: Target service.
            graph: Pre-built graph data.
            max_depth: Maximum BFS depth (default 5).

        Returns:
            UpstreamDependencies with service list and count.
        """
        all_upstream: Set[str] = set()
        visited: Set[str] = {service}
        initial = graph.reverse_adjacency.get(service, [])
        queue: deque[tuple[str, int]] = deque()

        for s in initial:
            visited.add(s)
            all_upstream.add(s)
            queue.append((s, 1))

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            parents = graph.reverse_adjacency.get(current, [])
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    all_upstream.add(p)
                    queue.append((p, depth + 1))

        return UpstreamDependencies(
            services_depending_on_failed=sorted(all_upstream),
            count=len(all_upstream),
        )

    def _downstream_deps(
        self, service: str, graph: GraphData,
        max_depth: int = 5,
    ) -> DownstreamDependencies:
        """Find all downstream dependencies using adjacency BFS.

        Downstream = services this service depends ON (callees).
        Depth-limited to keep per-service cost bounded.

        Args:
            service: Target service.
            graph: Pre-built graph data.
            max_depth: Maximum BFS depth (default 5).

        Returns:
            DownstreamDependencies with service list and count.
        """
        all_downstream: Set[str] = set()
        visited: Set[str] = {service}
        initial = graph.adjacency_list.get(service, [])
        queue: deque[tuple[str, int]] = deque()

        for s in initial:
            visited.add(s)
            all_downstream.add(s)
            queue.append((s, 1))

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            children = graph.adjacency_list.get(current, [])
            for c in children:
                if c not in visited:
                    visited.add(c)
                    all_downstream.add(c)
                    queue.append((c, depth + 1))

        return DownstreamDependencies(
            services_failed_depends_on=sorted(all_downstream),
            count=len(all_downstream),
        )

    # ── Algorithm 5: Criticality Score ──────────────────────────

    def _criticality_score(
        self,
        service: str,
        graph: GraphData,
        blast: BlastRadius,
        critical_path_services: Set[str],
    ) -> float:
        """Calculate criticality score for a service (0 to 1).

        Formula:
          score = min(upstream/U, W_u) + min(downstream/D, W_d)
                + min(blast/B, W_b) + (W_cp if on critical path)

        Where W_u=0.3, W_d=0.2, W_b=0.3, W_cp=0.2 (configurable).

        Args:
            service: Service name.
            graph: Pre-built graph data.
            blast: Pre-calculated blast radius.
            critical_path_services: Set of services on critical path.

        Returns:
            Criticality score clamped to [0.0, 1.0].
        """
        weights = self._config.criticality

        upstream_count = graph.in_degree.get(service, 0)
        downstream_count = graph.out_degree.get(service, 0)

        upstream_component = min(
            upstream_count / max(weights.upstream_divisor, 1.0),
            weights.upstream_weight,
        )
        downstream_component = min(
            downstream_count / max(weights.downstream_divisor, 1.0),
            weights.downstream_weight,
        )
        blast_component = min(
            blast.total_affected_count
            / max(weights.blast_radius_divisor, 1.0),
            weights.blast_radius_weight,
        )
        critical_path_component = (
            weights.critical_path_weight
            if service in critical_path_services
            else 0.0
        )

        score = (
            upstream_component
            + downstream_component
            + blast_component
            + critical_path_component
        )

        return round(min(max(score, 0.0), 1.0), 4)

    # ── SPOF Detection ──────────────────────────────────────────

    def _is_spof(self, node, graph: GraphData) -> bool:
        """Detect if a service is a Single Point of Failure.

        A SPOF has instance_count == 1 AND dependents >=
        spof_min_dependents (default: 3).

        Args:
            node: ServiceNode object.
            graph: Pre-built graph data.

        Returns:
            True if the service is identified as a SPOF.
        """
        if node.instance_count is None or node.instance_count > 1:
            return False

        threshold = self._config.thresholds.spof_min_dependents
        dependents = graph.in_degree.get(node.service_name, 0)

        return dependents >= threshold

    def _calculate_affected_rate(
        self, service: str, graph: GraphData
    ) -> float:
        """Sum request rates from edges originating at this service.

        Args:
            service: Service name.
            graph: Pre-built graph data.

        Returns:
            Total outbound request rate per second.
        """
        total = 0.0
        edges = graph.edge_map.get(service, [])
        for edge in edges:
            total += edge.request_rate_per_sec
        return round(total, 2)
