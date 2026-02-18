"""
File: core/action_sequencer.py
Purpose: Algorithm 5 – Topological sort of action items with cycle detection.
Dependencies: Standard library only + schema.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from agents.incident_commander_agent.schema import ActionItem


class CycleError(ValueError):
    """Raised when action dependencies contain a cycle."""


def _build_graph(
    actions: List[ActionItem],
) -> tuple[Dict[str, ActionItem], Dict[str, Set[str]], Dict[str, int]]:
    """Build adjacency list and in-degree map from action dependencies.

    Returns:
        Tuple of (id→action map, adjacency, in-degree).
    """
    by_id: Dict[str, ActionItem] = {a.action_id: a for a in actions}
    adj: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = {a.action_id: 0 for a in actions}

    for action in actions:
        for dep in action.dependencies:
            if dep in by_id:
                adj[dep].add(action.action_id)
                in_degree[action.action_id] = (
                    in_degree.get(action.action_id, 0) + 1
                )

    return by_id, dict(adj), in_degree


def _topo_sort(
    by_id: Dict[str, ActionItem],
    adj: Dict[str, Set[str]],
    in_degree: Dict[str, int],
) -> List[str]:
    """Kahn's algorithm for topological sort.

    Args:
        by_id: Mapping action_id → ActionItem.
        adj: Adjacency list (dep → set of dependants).
        in_degree: In-degree per action_id.

    Returns:
        Topologically sorted list of action_ids.

    Raises:
        CycleError: If the dependency graph has a cycle.
    """
    queue: deque[str] = deque()
    for aid, deg in in_degree.items():
        if deg == 0:
            queue.append(aid)

    result: List[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj.get(node, set()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(by_id):
        remaining = set(by_id) - set(result)
        raise CycleError(
            f"Dependency cycle detected among: {sorted(remaining)}"
        )

    return result


def sequence_actions(
    actions: List[ActionItem],
) -> List[ActionItem]:
    """Sequence actions respecting dependency order.

    Uses Kahn's topological sort. Actions with no dependencies
    come first; dependent actions are deferred until prerequisites
    are complete.

    Args:
        actions: List of action items with dependencies.

    Returns:
        New list sorted in safe execution order.

    Raises:
        CycleError: If a circular dependency is detected.
    """
    if not actions:
        return []

    by_id, adj, in_degree = _build_graph(actions)
    ordered_ids = _topo_sort(by_id, adj, in_degree)

    return [by_id[aid] for aid in ordered_ids]
