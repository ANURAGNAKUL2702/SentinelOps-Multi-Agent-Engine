"""Execution DAG — build, validate, topological-sort, and cycle-detect."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from .schema import AgentNode, CycleDetectedError


class ExecutionDAG:
    """Directed acyclic graph of agent execution dependencies.

    Nodes are :class:`AgentNode` instances keyed by *agent_name*.
    Edges ``(u, v)`` mean *u* must complete before *v* starts.

    :meth:`topological_sort` returns a list of *stages* — each stage
    is a list of agents that can execute in parallel (in-degree 0 at
    that point in the Kahn walk).
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, AgentNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self._adj: Dict[str, List[str]] = defaultdict(list)
        self._in_degree: Dict[str, int] = defaultdict(int)

    # ---- mutation ---------------------------------------------------------

    def add_node(
        self,
        agent_name: str,
        agent_instance: Any = None,
        timeout: float = 10.0,
    ) -> None:
        """Register an agent node in the DAG.

        Args:
            agent_name: Unique agent identifier.
            agent_instance: The agent object (callable).
            timeout: Per-agent timeout in seconds.

        Raises:
            ValueError: If *agent_name* is already registered.
        """
        if agent_name in self.nodes:
            raise ValueError(f"Duplicate node: {agent_name}")
        self.nodes[agent_name] = AgentNode(
            name=agent_name,
            agent_instance=agent_instance,
            timeout=timeout,
        )
        # Ensure entry in in_degree even with 0 incoming edges
        if agent_name not in self._in_degree:
            self._in_degree[agent_name] = 0

    def add_edge(self, from_agent: str, to_agent: str) -> None:
        """Add a dependency edge: *from_agent* must finish before *to_agent*.

        Args:
            from_agent: Upstream agent name.
            to_agent: Downstream agent name.

        Raises:
            ValueError: If either node is not in the DAG.
        """
        if from_agent not in self.nodes:
            raise ValueError(f"Unknown source node: {from_agent}")
        if to_agent not in self.nodes:
            raise ValueError(f"Unknown target node: {to_agent}")
        self.edges.append((from_agent, to_agent))
        self._adj[from_agent].append(to_agent)
        self._in_degree[to_agent] += 1

    # ---- queries ----------------------------------------------------------

    def get_dependencies(self, agent_name: str) -> List[str]:
        """Return agents that *agent_name* depends on (predecessors).

        Args:
            agent_name: Agent to query.

        Returns:
            List of predecessor agent names.
        """
        return [u for (u, v) in self.edges if v == agent_name]

    def detect_cycles(self) -> Optional[List[str]]:
        """Return a list of nodes forming a cycle, or ``None`` if acyclic.

        Uses DFS with WHITE/GRAY/BLACK colouring.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {n: WHITE for n in self.nodes}
        parent: Dict[str, Optional[str]] = {n: None for n in self.nodes}

        def _dfs(node: str) -> Optional[List[str]]:
            colour[node] = GRAY
            for neighbour in self._adj.get(node, []):
                if colour[neighbour] == GRAY:
                    # back-edge → reconstruct cycle
                    cycle = [neighbour, node]
                    cur = parent[node]
                    while cur is not None and cur != neighbour:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(neighbour)
                    cycle.reverse()
                    return cycle
                if colour[neighbour] == WHITE:
                    parent[neighbour] = node
                    result = _dfs(neighbour)
                    if result is not None:
                        return result
            colour[node] = BLACK
            return None

        for node in self.nodes:
            if colour[node] == WHITE:
                result = _dfs(node)
                if result is not None:
                    return result
        return None

    def topological_sort(self) -> List[List[str]]:
        """Return stages (Kahn's algorithm).

        Each stage is a list of agent names that can execute in parallel.
        The stages respect dependency ordering.

        Returns:
            Ordered list of stages.

        Raises:
            CycleDetectedError: If the DAG contains a cycle.
        """
        if not self.nodes:
            return []

        in_deg = dict(self._in_degree)
        # ensure every node present
        for n in self.nodes:
            in_deg.setdefault(n, 0)

        queue: deque[str] = deque(n for n, d in in_deg.items() if d == 0)
        stages: List[List[str]] = []
        visited = 0

        while queue:
            stage = list(queue)
            queue.clear()
            stages.append(stage)
            visited += len(stage)
            for node in stage:
                for neighbour in self._adj.get(node, []):
                    in_deg[neighbour] -= 1
                    if in_deg[neighbour] == 0:
                        queue.append(neighbour)

        if visited != len(self.nodes):
            cycle = self.detect_cycles()
            raise CycleDetectedError(cycle or ["<unknown>"])

        return stages

    def validate(self) -> None:
        """Validate the DAG: no cycles and all edge endpoints exist.

        Raises:
            CycleDetectedError: If cycles exist.
            ValueError: If edge references a missing node.
        """
        for u, v in self.edges:
            if u not in self.nodes:
                raise ValueError(f"Edge references missing node: {u}")
            if v not in self.nodes:
                raise ValueError(f"Edge references missing node: {v}")
        cycles = self.detect_cycles()
        if cycles:
            raise CycleDetectedError(cycles)
