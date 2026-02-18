"""
metrics_store.py — Time-series metrics storage and retrieval.

Acts as a mini Prometheus: accepts metric records from the simulation,
stores them in memory, and exposes deterministic query methods.

This module does NOT:
  • Detect root causes
  • Analyze trends
  • Inject failures
  • Modify ingested data

It only stores and retrieves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class MetricsStore:
    """In-memory time-series metrics store."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._by_service: Dict[str, List[Dict[str, Any]]] = {}

    # ── ingestion ───────────────────────────────────────────────

    def store(self, metrics: List[Dict[str, Any]]) -> None:
        """Ingest a batch of metric records from the simulation.

        Each record must contain at minimum:
          timestamp, service, cpu_percent, memory_percent,
          latency_ms, error_rate
        """
        for record in metrics:
            self._records.append(record)
            svc = record["service"]
            self._by_service.setdefault(svc, []).append(record)

    def clear(self) -> None:
        """Wipe all stored metrics."""
        self._records.clear()
        self._by_service.clear()

    # ── basic queries ───────────────────────────────────────────

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def services(self) -> List[str]:
        """Return the list of services that have metric data."""
        return sorted(self._by_service.keys())

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every stored metric record."""
        return list(self._records)

    def get_by_service(self, service: str) -> List[Dict[str, Any]]:
        """Return all metric records for a single service."""
        return list(self._by_service.get(service, []))

    def get_between(
        self,
        start: str,
        end: str,
        service: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return metrics within a timestamp range (ISO-8601 strings).

        Optionally filter by service.
        """
        source = self._by_service.get(service, []) if service else self._records
        return [r for r in source if start <= r["timestamp"] <= end]

    def get_latest_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return the most recent metric record for every service."""
        snapshot: Dict[str, Dict[str, Any]] = {}
        for svc, records in self._by_service.items():
            if records:
                snapshot[svc] = records[-1]
        return snapshot

    # ── threshold queries ───────────────────────────────────────

    def get_services_with_high_cpu(
        self, threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """Return the latest record of each service whose most recent
        CPU reading exceeds *threshold*."""
        results = []
        for svc, records in self._by_service.items():
            if records and records[-1]["cpu_percent"] > threshold:
                results.append(records[-1])
        return sorted(results, key=lambda r: r["cpu_percent"], reverse=True)

    def get_services_with_high_memory(
        self, threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """Return the latest record of each service whose most recent
        memory reading exceeds *threshold*."""
        results = []
        for svc, records in self._by_service.items():
            if records and records[-1]["memory_percent"] > threshold:
                results.append(records[-1])
        return sorted(results, key=lambda r: r["memory_percent"], reverse=True)

    def get_services_with_high_latency(
        self, threshold: float = 500.0
    ) -> List[Dict[str, Any]]:
        """Return the latest record of each service whose most recent
        latency exceeds *threshold* ms."""
        results = []
        for svc, records in self._by_service.items():
            if records and records[-1]["latency_ms"] > threshold:
                results.append(records[-1])
        return sorted(results, key=lambda r: r["latency_ms"], reverse=True)

    def get_services_with_high_error_rate(
        self, threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Return the latest record of each service whose most recent
        error rate exceeds *threshold* %."""
        results = []
        for svc, records in self._by_service.items():
            if records and records[-1]["error_rate"] > threshold:
                results.append(records[-1])
        return sorted(results, key=lambda r: r["error_rate"], reverse=True)

    # ── time-series helpers ─────────────────────────────────────

    def get_metric_trend(
        self, service: str, metric: str
    ) -> List[Dict[str, Any]]:
        """Return ``[{timestamp, value}, ...]`` for a single metric
        of a single service — convenient for trend inspection."""
        records = self._by_service.get(service, [])
        return [
            {"timestamp": r["timestamp"], "value": r.get(metric)}
            for r in records
            if metric in r
        ]

    def get_average(self, service: str, metric: str) -> Optional[float]:
        """Return the arithmetic mean of *metric* across all stored
        records for *service*."""
        records = self._by_service.get(service, [])
        values = [r[metric] for r in records if metric in r]
        if not values:
            return None
        return round(sum(values) / len(values), 2)
