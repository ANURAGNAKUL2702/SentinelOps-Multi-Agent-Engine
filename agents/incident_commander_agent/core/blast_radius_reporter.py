"""
File: core/blast_radius_reporter.py
Purpose: Algorithm 3 – Calculate blast radius from affected services.
Dependencies: Standard library only + schema + config.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from typing import List, Optional, Set

from agents.incident_commander_agent.config import BlastRadiusConfig
from agents.incident_commander_agent.schema import (
    BlastRadius,
    CausalLink,
)


_CUSTOMER_FACING_KEYWORDS: Set[str] = {
    "api", "gateway", "frontend", "web", "mobile",
    "auth", "login", "checkout", "payment", "cdn",
}


def _is_customer_facing(service: str) -> bool:
    """Return True if the service name suggests customer exposure."""
    normalized = service.lower().replace("-", "_")
    return any(k in normalized for k in _CUSTOMER_FACING_KEYWORDS)


def _availability_impact(
    affected_count: int,
    total_services: int,
) -> float:
    """Calculate availability impact as fraction 0.0–1.0."""
    if total_services <= 0:
        return 0.0
    return min(1.0, affected_count / total_services)


def calculate_blast_radius(
    affected_services: List[str],
    causal_chain: List[CausalLink] | None = None,
    total_known_services: int = 10,
    config: BlastRadiusConfig | None = None,
) -> BlastRadius:
    """Quantify the blast radius of an incident.

    Args:
        affected_services: Service names involved in the incident.
        causal_chain: Optional causal chain to discover more services.
        total_known_services: Total services in the system.
        config: Blast radius configuration overrides.

    Returns:
        BlastRadius with estimated impact metrics.
    """
    cfg = config or BlastRadiusConfig()
    chain = causal_chain or []

    # Merge services from explicit list + causal chain
    services: Set[str] = set(affected_services)
    for link in chain:
        services.add(link.cause)
        services.add(link.effect)

    service_list = sorted(services)
    count = len(service_list)

    customer_facing = any(_is_customer_facing(s) for s in service_list)

    estimated_users = count * cfg.avg_users_per_service
    revenue = count * cfg.revenue_per_minute_per_service
    availability = _availability_impact(count, total_known_services)

    return BlastRadius(
        affected_services=service_list,
        affected_service_count=count,
        estimated_users=estimated_users,
        revenue_impact_per_minute=round(revenue, 2),
        availability_impact=round(availability, 4),
        is_customer_facing=customer_facing,
    )
