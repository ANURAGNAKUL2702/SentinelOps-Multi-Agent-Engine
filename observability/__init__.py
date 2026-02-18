"""
observability — Monitoring / observability layer.

Wires up MetricsStore, LogStore, and QueryEngine.
Provides a convenience function to hydrate the stores from
simulation output so agents never touch raw simulation data.
"""

from __future__ import annotations

from typing import Any, Dict

from observability.metrics_store import MetricsStore
from observability.log_store import LogStore
from observability.query_engine import QueryEngine


def build_observability_from_simulation(
    simulation_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Hydrate the observability layer from a simulation payload.

    Parameters
    ----------
    simulation_output : dict
        The dict returned by ``simulation.run_simulation()``.

    Returns
    -------
    dict
        ``{ metrics_store, log_store, query_engine }`` — ready for
        agents to query.
    """
    metrics_store = MetricsStore()
    log_store = LogStore()

    metrics_store.store(simulation_output["metrics"])
    log_store.store(simulation_output["logs"])

    query_engine = QueryEngine(metrics_store, log_store)

    return {
        "metrics_store": metrics_store,
        "log_store": log_store,
        "query_engine": query_engine,
    }
