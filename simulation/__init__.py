"""
simulation — Fintech payment-platform incident simulator.

Orchestrates the full simulation flow:
  1. Load service definitions
  2. Build dependency graph
  3. Choose / inject a failure scenario
  4. Generate degraded + healthy metrics
  5. Generate correlated log entries
  6. Return a single structured incident payload
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from simulation.services import get_service_names, get_all_services
from simulation.dependency_graph import get_full_dependency_map, get_cascade_impact
from simulation.metrics_engine import generate_metrics
from simulation.log_engine import generate_logs
from simulation.failure_injector import inject_scenario, get_available_scenarios


def run_simulation(
    scenario: str | None = None,
    target_service: str | None = None,
    duration_minutes: int = 30,
    metrics_interval_seconds: int = 60,
    log_interval_seconds: int = 30,
    start_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Execute a complete incident simulation and return structured data.

    Parameters
    ----------
    scenario : str | None
        Failure scenario name (e.g. ``"memory_leak"``).
        ``None`` → random.
    target_service : str | None
        Service to inject the failure into.
        ``None`` → scenario default.
    duration_minutes : int
        Length of the simulated observation window.
    metrics_interval_seconds : int
        Gap between metric data points.
    log_interval_seconds : int
        Average gap between log lines per service.
    start_time : datetime | None
        Anchor timestamp.  Defaults to *now* (UTC).

    Returns
    -------
    dict
        ``{ services, dependencies, metrics, logs,
             root_cause, severity, scenario, blast_radius }``
    """
    start = start_time or datetime.now(timezone.utc)

    # 1 → load services
    services = get_service_names()

    # 2 → build dependency graph
    dependencies = get_full_dependency_map()

    # 3-4 → choose scenario & inject failure
    incident = inject_scenario(scenario=scenario, service=target_service)
    failure_plan = incident["failure_plan"]

    # 5 → generate metrics (healthy + degraded)
    metrics = generate_metrics(
        services=services,
        duration_minutes=duration_minutes,
        interval_seconds=metrics_interval_seconds,
        failure_plan=failure_plan,
        start_time=start,
    )

    # 6 → generate logs (correlated with failure)
    logs = generate_logs(
        services=services,
        duration_minutes=duration_minutes,
        interval_seconds=log_interval_seconds,
        failure_plan=failure_plan,
        start_time=start,
    )

    # 7 → assemble final payload
    return {
        "services": services,
        "dependencies": dependencies,
        "metrics": metrics,
        "logs": logs,
        "root_cause": incident["root_cause"],
        "severity": incident["severity"],
        "scenario": incident["scenario"],
        "blast_radius": incident["blast_radius"],
    }
