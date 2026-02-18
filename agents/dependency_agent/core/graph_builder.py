"""
File: core/graph_builder.py
Purpose: Build directed dependency graph + detect cycles.
Dependencies: Standard library only
Performance: O(V+E) complexity, <10ms for 1000 services

Implements:
  Algorithm 1: Build Directed Graph (adjacency list, reverse adjacency,
               in-degree, out-degree)
  Algorithm 2: Detect Cycles (DFS with recursion stack)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    DependencyAnalysisInput,
    GraphBuildResult,
    GraphData,
    ServiceEdge,
    ServiceNode,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.graph_builder")


class GraphBuilder:
    """Builds a directed dependency graph from service nodes and edges.

    Constructs adjacency lists, computes in/out-degree, detects cycles
    using DFS, and calculates max graph depth via BFS.

    Args:
        config: Agent configuration with performance limits.

    Example::

        builder = GraphBuilder(DependencyAgentConfig())
        result = builder.build(input_data)
        print(result.has_cycles)
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def build(
        self,
        input_data: DependencyAnalysisInput,
        correlation_id: str = "",
    ) -> GraphBuildResult:
        """Build the directed graph from input data.

        Args:
            input_data: Input containing service_graph with nodes and edges.
            correlation_id: Request correlation ID.

        Returns:
            GraphBuildResult with adjacency lists, cycle info, and depth.
        """
        start = time.perf_counter()

        nodes = input_data.service_graph.nodes
        edges = input_data.service_graph.edges

        # ── Algorithm 1: Build directed graph ───────────────────
        graph = self._build_adjacency(nodes, edges)

        # ── Algorithm 2: Detect cycles (DFS) ────────────────────
        has_cycles, cycle_paths = self._detect_cycles(graph)

        if has_cycles:
            logger.warning(
                "Dependency cycle detected",
                extra={
                    "correlation_id": correlation_id,
                    "layer": "graph_build",
                    "context": {
                        "cycle_paths": cycle_paths,
                        "cycle_count": len(cycle_paths),
                    },
                },
            )

        # ── Calculate max depth (BFS from roots) ────────────────
        max_depth = self._calculate_max_depth(graph)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Graph built: {len(nodes)} nodes, {len(edges)} edges, "
            f"cycles={has_cycles}, depth={max_depth}, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "graph_build",
                "context": {
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "has_cycles": has_cycles,
                    "max_depth": max_depth,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return GraphBuildResult(
            graph=graph,
            total_services=len(nodes),
            total_dependencies=len(edges),
            has_cycles=has_cycles,
            cycle_paths=cycle_paths,
            max_depth=max_depth,
            build_latency_ms=round(elapsed_ms, 2),
        )

    # ── Algorithm 1: Build Adjacency Lists ──────────────────────

    def _build_adjacency(
        self,
        nodes: List[ServiceNode],
        edges: List[ServiceEdge],
    ) -> GraphData:
        """Build adjacency list, reverse adjacency, in/out degree.

        Complexity: O(V + E)

        Args:
            nodes: List of service nodes.
            edges: List of dependency edges.

        Returns:
            GraphData with all adjacency structures populated.
        """
        adjacency: Dict[str, List[str]] = {}
        reverse: Dict[str, List[str]] = {}
        in_deg: Dict[str, int] = {}
        out_deg: Dict[str, int] = {}
        node_map: Dict[str, ServiceNode] = {}
        edge_map: Dict[str, List[ServiceEdge]] = {}

        # Initialize for all nodes
        for node in nodes:
            name = node.service_name
            adjacency[name] = []
            reverse[name] = []
            in_deg[name] = 0
            out_deg[name] = 0
            node_map[name] = node
            edge_map[name] = []

        # Process edges
        for edge in edges:
            src = edge.source
            tgt = edge.target

            # Ensure nodes exist (handle edges referencing unknown nodes)
            for svc in (src, tgt):
                if svc not in adjacency:
                    adjacency[svc] = []
                    reverse[svc] = []
                    in_deg[svc] = 0
                    out_deg[svc] = 0

            adjacency[src].append(tgt)
            reverse[tgt].append(src)
            out_deg[src] += 1
            in_deg[tgt] += 1
            edge_map.setdefault(src, []).append(edge)

        return GraphData(
            adjacency_list=adjacency,
            reverse_adjacency=reverse,
            in_degree=in_deg,
            out_degree=out_deg,
            node_map=node_map,
            edge_map=edge_map,
        )

    # ── Algorithm 2: Detect Cycles (DFS) ────────────────────────

    def _detect_cycles(
        self, graph: GraphData
    ) -> Tuple[bool, List[List[str]]]:
        """Detect cycles using DFS with recursion stack.

        Complexity: O(V + E)

        Args:
            graph: Built graph data.

        Returns:
            Tuple of (has_cycle, list_of_cycle_paths).
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycle_paths: List[List[str]] = []
        path: List[str] = []
        max_depth = self._config.performance.max_graph_depth

        def dfs(node: str, depth: int = 0) -> bool:
            if depth > max_depth:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.adjacency_list.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, depth + 1):
                        return True
                elif neighbor in rec_stack:
                    # Back edge found — extract cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycle_paths.append(cycle)
                    return True

            path.pop()
            rec_stack.discard(node)
            return False

        has_cycle = False
        for node in graph.adjacency_list:
            if node not in visited:
                if dfs(node):
                    has_cycle = True
                    # Reset for next component (find all cycles)
                    rec_stack.clear()
                    path.clear()

        return has_cycle, cycle_paths

    # ── Max Depth Calculation (BFS from roots) ──────────────────

    def _calculate_max_depth(self, graph: GraphData) -> int:
        """Calculate maximum dependency depth via BFS from root nodes.

        Root nodes have in_degree == 0 (no dependencies).

        Complexity: O(V + E)

        Args:
            graph: Built graph data.

        Returns:
            Maximum depth (0 if empty graph).
        """
        if not graph.adjacency_list:
            return 0

        roots = [
            n for n, deg in graph.in_degree.items() if deg == 0
        ]

        # If no roots (all in cycle), pick any node
        if not roots:
            roots = [next(iter(graph.adjacency_list))]

        max_depth = 0
        visited: Set[str] = set()

        for root in roots:
            queue: deque[Tuple[str, int]] = deque([(root, 1)])
            visited.add(root)

            while queue:
                node, depth = queue.popleft()
                max_depth = max(max_depth, depth)

                if depth >= self._config.performance.max_graph_depth:
                    continue

                for neighbor in graph.adjacency_list.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return max_depth
