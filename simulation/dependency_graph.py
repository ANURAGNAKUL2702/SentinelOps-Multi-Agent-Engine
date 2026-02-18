"""
dependency_graph.py — Service dependency topology for the fintech platform.

Provides:
  • Direct downstream dependencies for every service.
  • Upstream (reverse) look-ups.
  • Cascade / blast-radius calculation when a service fails.
"""

from __future__ import annotations
from typing import Dict, List, Set

from simulation.services import get_service_names


# ── topology ────────────────────────────────────────────────────────
# key  → depends on → list of downstream services
# "api-gateway" depends on ["auth-service", "payment-service", "merchant-portal"]

DEPENDENCY_MAP: Dict[str, List[str]] = {
    "api-gateway":          ["auth-service", "payment-service", "merchant-portal"],
    "auth-service":         ["database", "cache-service"],
    "payment-service":      ["database", "fraud-service", "notification-service"],
    "fraud-service":        ["database"],
    "notification-service": ["database"],
    "merchant-portal":      ["api-gateway"],
    "cache-service":        [],
    "database":             [],
}


# ── public API ──────────────────────────────────────────────────────

def get_dependencies(service: str) -> List[str]:
    """Return the direct downstream dependencies of *service*."""
    return list(DEPENDENCY_MAP.get(service, []))


def get_upstream_services(service: str) -> List[str]:
    """Return every service that directly depends on *service*."""
    return [svc for svc, deps in DEPENDENCY_MAP.items() if service in deps]


def get_all_upstream_services(service: str) -> Set[str]:
    """Recursively walk the graph **upward** to find every service
    that would be affected if *service* goes down (blast radius)."""
    affected: Set[str] = set()
    queue = [service]
    while queue:
        current = queue.pop(0)
        for upstream in get_upstream_services(current):
            if upstream not in affected:
                affected.add(upstream)
                queue.append(upstream)
    return affected


def get_cascade_impact(failed_service: str) -> Dict[str, object]:
    """Return a structured blast-radius report.

    Example output when *database* fails::

        {
            "failed_service": "database",
            "directly_affected": ["auth-service", "payment-service", ...],
            "indirectly_affected": ["api-gateway", "merchant-portal"],
            "total_affected": 6,
            "all_affected": [...]
        }
    """
    all_affected = get_all_upstream_services(failed_service)
    direct = set(get_upstream_services(failed_service))
    indirect = all_affected - direct

    return {
        "failed_service": failed_service,
        "directly_affected": sorted(direct),
        "indirectly_affected": sorted(indirect),
        "total_affected": len(all_affected),
        "all_affected": sorted(all_affected),
    }


def get_full_dependency_map() -> Dict[str, List[str]]:
    """Return a copy of the entire dependency adjacency list."""
    return {k: list(v) for k, v in DEPENDENCY_MAP.items()}
