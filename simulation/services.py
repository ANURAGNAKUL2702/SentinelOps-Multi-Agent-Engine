"""
services.py — Service definitions for the fintech payment platform.

Pure data definitions. No logic.
Each service carries metadata: criticality, whether it's external,
and a human-readable description.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ServiceDefinition:
    """Immutable descriptor for a single platform service."""
    name: str
    is_critical: bool = False
    is_external: bool = False
    description: str = ""
    team_owner: str = "platform"
    tier: int = 2  # 1 = highest priority, 3 = lowest


# ── service catalogue ───────────────────────────────────────────────
SERVICES: List[ServiceDefinition] = [
    ServiceDefinition(
        name="api-gateway",
        is_critical=True,
        description="Public entry-point — routes all inbound traffic.",
        team_owner="platform",
        tier=1,
    ),
    ServiceDefinition(
        name="auth-service",
        is_critical=True,
        description="Handles authentication, token validation, and SSO.",
        team_owner="identity",
        tier=1,
    ),
    ServiceDefinition(
        name="payment-service",
        is_critical=True,
        description="Core payment processing — charges, refunds, settlements.",
        team_owner="payments",
        tier=1,
    ),
    ServiceDefinition(
        name="fraud-service",
        is_critical=True,
        description="Real-time fraud scoring and transaction blocking.",
        team_owner="risk",
        tier=1,
    ),
    ServiceDefinition(
        name="notification-service",
        is_critical=False,
        description="Email / SMS / push notifications to merchants and customers.",
        team_owner="comms",
        tier=2,
    ),
    ServiceDefinition(
        name="database",
        is_critical=True,
        description="Primary PostgreSQL cluster — source of truth for all transactional data.",
        team_owner="data-infra",
        tier=1,
    ),
    ServiceDefinition(
        name="cache-service",
        is_critical=False,
        description="Redis layer for session caching and rate limiting.",
        team_owner="platform",
        tier=2,
    ),
    ServiceDefinition(
        name="merchant-portal",
        is_critical=False,
        is_external=True,
        description="Web dashboard for merchant self-service.",
        team_owner="merchant-experience",
        tier=2,
    ),
]


# ── public helpers ──────────────────────────────────────────────────

def get_all_services() -> List[ServiceDefinition]:
    """Return the full service catalogue."""
    return list(SERVICES)


def get_service_names() -> List[str]:
    """Return a flat list of service names."""
    return [s.name for s in SERVICES]


def get_critical_services() -> List[ServiceDefinition]:
    """Return only tier-1 / critical services."""
    return [s for s in SERVICES if s.is_critical]


def get_service(name: str) -> ServiceDefinition | None:
    """Look up a single service by name (returns None if unknown)."""
    return next((s for s in SERVICES if s.name == name), None)
