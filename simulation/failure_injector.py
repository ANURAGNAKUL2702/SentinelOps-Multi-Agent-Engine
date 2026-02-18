"""
failure_injector.py — Incident injection engine.

Each injector function:
  1. Builds a *failure_plan* (service → profile mapping).
  2. Returns a structured *scenario* dict that the simulation
     orchestrator feeds into metrics_engine and log_engine.

This keeps the injection logic in one place while the actual metric /
log generation stays in its own module.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from simulation.dependency_graph import (
    get_cascade_impact,
    get_dependencies,
    get_upstream_services,
)
from simulation.services import get_service_names


# ── severity helpers ────────────────────────────────────────────────

def _auto_severity(affected_count: int, critical_service: bool) -> str:
    """Derive a SEV level from blast-radius size."""
    if critical_service and affected_count >= 4:
        return "SEV-1"
    if critical_service or affected_count >= 3:
        return "SEV-2"
    if affected_count >= 1:
        return "SEV-3"
    return "SEV-4"


# ── individual injectors ───────────────────────────────────────────

def inject_memory_leak(
    service: str = "payment-service",
) -> Dict[str, Any]:
    """Simulate a gradual memory leak on *service*.

    The target service gets the ``memory_leak`` failure profile.
    Upstream dependents get the ``network_latency`` profile (they see
    slowdowns as the leaking service degrades).
    """
    impact = get_cascade_impact(service)
    failure_plan: Dict[str, str] = {service: "memory_leak"}

    # upstream services experience knock-on latency
    for upstream in impact["directly_affected"]:
        failure_plan[upstream] = "network_latency"

    return {
        "scenario": "memory_leak",
        "root_cause": f"{service} memory leak",
        "root_service": service,
        "failure_plan": failure_plan,
        "severity": _auto_severity(
            impact["total_affected"], critical_service=True
        ),
        "blast_radius": impact,
    }


def inject_cpu_spike(
    service: str = "fraud-service",
) -> Dict[str, Any]:
    """Simulate a sudden CPU spike on *service*.

    Upstream dependents see elevated latency.
    """
    impact = get_cascade_impact(service)
    failure_plan: Dict[str, str] = {service: "cpu_spike"}

    for upstream in impact["directly_affected"]:
        failure_plan[upstream] = "network_latency"

    return {
        "scenario": "cpu_spike",
        "root_cause": f"{service} CPU spike",
        "root_service": service,
        "failure_plan": failure_plan,
        "severity": _auto_severity(
            impact["total_affected"], critical_service=True
        ),
        "blast_radius": impact,
    }


def inject_database_timeout(
    service: str = "database",
) -> Dict[str, Any]:
    """Simulate the primary database becoming unresponsive.

    Every service that depends on the DB (directly or transitively)
    receives a ``database_timeout`` or ``network_latency`` profile.
    """
    impact = get_cascade_impact(service)
    failure_plan: Dict[str, str] = {service: "database_timeout"}

    for dep in impact["directly_affected"]:
        failure_plan[dep] = "database_timeout"
    for dep in impact["indirectly_affected"]:
        failure_plan[dep] = "network_latency"

    return {
        "scenario": "database_timeout",
        "root_cause": f"{service} connection timeout",
        "root_service": service,
        "failure_plan": failure_plan,
        "severity": "SEV-1",  # DB outage is always SEV-1
        "blast_radius": impact,
    }


def inject_network_latency(
    service: str = "api-gateway",
) -> Dict[str, Any]:
    """Simulate network degradation affecting *service*.

    Downstream dependencies of the service also experience latency.
    """
    impact = get_cascade_impact(service)
    downstream = get_dependencies(service)

    failure_plan: Dict[str, str] = {service: "network_latency"}
    for dep in downstream:
        failure_plan[dep] = "network_latency"

    return {
        "scenario": "network_latency",
        "root_cause": f"{service} network degradation",
        "root_service": service,
        "failure_plan": failure_plan,
        "severity": _auto_severity(
            len(downstream) + impact["total_affected"],
            critical_service=True,
        ),
        "blast_radius": impact,
    }


# ── scenario catalogue ─────────────────────────────────────────────

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "memory_leak": {
        "injector": inject_memory_leak,
        "default_service": "payment-service",
        "description": "Gradual heap exhaustion on a payment-critical path.",
    },
    "cpu_spike": {
        "injector": inject_cpu_spike,
        "default_service": "fraud-service",
        "description": "Sudden CPU saturation on the fraud scoring engine.",
    },
    "database_timeout": {
        "injector": inject_database_timeout,
        "default_service": "database",
        "description": "Primary database becomes unresponsive.",
    },
    "network_latency": {
        "injector": inject_network_latency,
        "default_service": "api-gateway",
        "description": "Upstream network degradation at the edge.",
    },
}


def get_available_scenarios() -> List[str]:
    """Return the names of all registered failure scenarios."""
    return list(SCENARIOS.keys())


def inject_scenario(
    scenario: str | None = None,
    service: str | None = None,
) -> Dict[str, Any]:
    """High-level entry point: pick a scenario (or random) and run it.

    Parameters
    ----------
    scenario : str | None
        One of the keys in ``SCENARIOS``.  ``None`` → pick at random.
    service : str | None
        Override the target service.  ``None`` → use scenario default.

    Returns
    -------
    dict
        Structured scenario descriptor including ``failure_plan``,
        ``root_cause``, ``severity``, and ``blast_radius``.
    """
    if scenario is None:
        scenario = random.choice(list(SCENARIOS.keys()))

    entry = SCENARIOS.get(scenario)
    if entry is None:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {get_available_scenarios()}"
        )

    target = service or entry["default_service"]
    return entry["injector"](service=target)
