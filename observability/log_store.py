"""
log_store.py — Structured log storage and filtering.

Acts as a mini Elastic Stack: accepts log records from the simulation,
stores them immutably, and exposes deterministic query methods.

This module does NOT:
  • Guess root causes
  • Modify ingested data
  • Inject failures

Logs are immutable once stored.  Only retrieval and filtering.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class LogStore:
    """In-memory structured log store."""

    def __init__(self) -> None:
        self._logs: List[Dict[str, Any]] = []
        self._by_service: Dict[str, List[Dict[str, Any]]] = {}
        self._by_level: Dict[str, List[Dict[str, Any]]] = {}

    # ── ingestion ───────────────────────────────────────────────

    def store(self, logs: List[Dict[str, Any]]) -> None:
        """Ingest a batch of log records from the simulation.

        Each record must contain at minimum:
          timestamp, service, level, message
        """
        for log in logs:
            self._logs.append(log)
            svc = log["service"]
            lvl = log["level"]
            self._by_service.setdefault(svc, []).append(log)
            self._by_level.setdefault(lvl, []).append(log)

    def clear(self) -> None:
        """Wipe all stored logs."""
        self._logs.clear()
        self._by_service.clear()
        self._by_level.clear()

    # ── basic queries ───────────────────────────────────────────

    @property
    def total_logs(self) -> int:
        return len(self._logs)

    @property
    def services(self) -> List[str]:
        """Return the list of services that have log data."""
        return sorted(self._by_service.keys())

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every stored log record."""
        return list(self._logs)

    def get_by_service(self, service: str) -> List[Dict[str, Any]]:
        """Return all log records for a single service."""
        return list(self._by_service.get(service, []))

    def get_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Return all log records matching *level* (INFO/WARNING/ERROR)."""
        return list(self._by_level.get(level.upper(), []))

    def get_errors(self) -> List[Dict[str, Any]]:
        """Shortcut: return all ERROR-level logs."""
        return self.get_by_level("ERROR")

    def get_warnings(self) -> List[Dict[str, Any]]:
        """Shortcut: return all WARNING-level logs."""
        return self.get_by_level("WARNING")

    def get_between(
        self,
        start: str,
        end: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return logs within a timestamp range (ISO-8601 strings).

        Optionally filter by service and/or level.
        """
        source = self._logs
        if service:
            source = self._by_service.get(service, [])

        results = [r for r in source if start <= r["timestamp"] <= end]

        if level:
            results = [r for r in results if r["level"] == level.upper()]

        return results

    # ── error analysis ──────────────────────────────────────────

    def get_error_count_by_service(self) -> Dict[str, int]:
        """Return ``{service: error_count}`` for every service,
        sorted descending by count."""
        counter: Counter = Counter()
        for log in self._by_level.get("ERROR", []):
            counter[log["service"]] += 1
        return dict(counter.most_common())

    def get_top_error_services(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return the top *n* services generating the most ERROR logs,
        as ``[(service, count), ...]``."""
        counter: Counter = Counter()
        for log in self._by_level.get("ERROR", []):
            counter[log["service"]] += 1
        return counter.most_common(n)

    def get_error_frequency_over_time(
        self, service: str, bucket_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Split the service's log timeline into *bucket_count* equal
        windows and return the ERROR count per bucket.

        Useful for spotting error spikes over time.
        """
        svc_logs = self._by_service.get(service, [])
        if not svc_logs:
            return []

        timestamps = [l["timestamp"] for l in svc_logs]
        ts_min, ts_max = min(timestamps), max(timestamps)

        # build buckets by splitting the timestamp range
        errors_only = [l for l in svc_logs if l["level"] == "ERROR"]
        if not errors_only or ts_min == ts_max:
            return [{"bucket": 0, "start": ts_min, "end": ts_max, "error_count": len(errors_only)}]

        # simple approach: divide error timestamps into equal-sized buckets
        total = len(svc_logs)
        bucket_size = max(total // bucket_count, 1)

        buckets: List[Dict[str, Any]] = []
        for i in range(bucket_count):
            start_idx = i * bucket_size
            end_idx = min((i + 1) * bucket_size, total)
            if start_idx >= total:
                break
            window = svc_logs[start_idx:end_idx]
            err_count = sum(1 for l in window if l["level"] == "ERROR")
            buckets.append({
                "bucket": i,
                "start": window[0]["timestamp"],
                "end": window[-1]["timestamp"],
                "error_count": err_count,
            })

        return buckets

    # ── search ──────────────────────────────────────────────────

    def search_messages(
        self,
        keyword: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return logs whose message contains *keyword* (case-insensitive).

        Optionally filter by service and/or level.
        """
        keyword_lower = keyword.lower()
        source = self._logs
        if service:
            source = self._by_service.get(service, [])

        results = [
            l for l in source
            if keyword_lower in l["message"].lower()
        ]

        if level:
            results = [r for r in results if r["level"] == level.upper()]

        return results
