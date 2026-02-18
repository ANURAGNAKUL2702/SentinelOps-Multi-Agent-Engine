"""
query_engine.py — High-level observability queries for AI agents.

Built on top of MetricsStore + LogStore.  Provides rule-based
detection and filtered views so agents never need to write raw
filtering logic or loop through raw data.

This module does NOT:
  • Use AI / ML
  • Decide the final root cause
  • Inject failures
  • Modify any data

It provides deterministic, rule-based insights only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from observability.metrics_store import MetricsStore
from observability.log_store import LogStore


class QueryEngine:
    """Smart query layer that agents call instead of touching raw stores."""

    def __init__(
        self, metrics_store: MetricsStore, log_store: LogStore
    ) -> None:
        self._metrics = metrics_store
        self._logs = log_store

    # ── memory anomalies ────────────────────────────────────────

    def get_services_with_abnormal_memory(
        self, threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """Return services whose latest memory exceeds *threshold*,
        enriched with trend data (start vs end average)."""
        high_mem = self._metrics.get_services_with_high_memory(threshold)
        results = []
        for record in high_mem:
            svc = record["service"]
            trend = self._metrics.get_metric_trend(svc, "memory_percent")
            values = [p["value"] for p in trend if p["value"] is not None]
            early = values[:5]
            late = values[-5:]
            results.append({
                "service": svc,
                "current_memory": record["memory_percent"],
                "avg_start": round(sum(early) / max(len(early), 1), 2),
                "avg_end": round(sum(late) / max(len(late), 1), 2),
                "trend": "increasing" if late and early and
                         (sum(late)/len(late)) > (sum(early)/len(early)) * 1.3
                         else "stable",
                "related_errors": len(
                    self._logs.search_messages("memory", service=svc, level="ERROR")
                    + self._logs.search_messages("heap", service=svc, level="ERROR")
                    + self._logs.search_messages("oom", service=svc, level="ERROR")
                ),
            })
        return results

    # ── error rate anomalies ────────────────────────────────────

    def get_services_with_high_error_rate(
        self, threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Return services whose latest error rate exceeds *threshold*,
        enriched with error log counts."""
        high_err = self._metrics.get_services_with_high_error_rate(threshold)
        error_counts = self._logs.get_error_count_by_service()
        results = []
        for record in high_err:
            svc = record["service"]
            results.append({
                "service": svc,
                "current_error_rate": record["error_rate"],
                "total_error_logs": error_counts.get(svc, 0),
                "latest_latency_ms": record["latency_ms"],
            })
        return results

    # ── latency spikes ──────────────────────────────────────────

    def get_latency_spikes(
        self, threshold: float = 500.0
    ) -> List[Dict[str, Any]]:
        """Return services with latency above *threshold* ms,
        enriched with trend direction."""
        high_lat = self._metrics.get_services_with_high_latency(threshold)
        results = []
        for record in high_lat:
            svc = record["service"]
            trend = self._metrics.get_metric_trend(svc, "latency_ms")
            values = [p["value"] for p in trend if p["value"] is not None]
            early = values[:5]
            late = values[-5:]
            results.append({
                "service": svc,
                "current_latency_ms": record["latency_ms"],
                "avg_start_latency": round(sum(early) / max(len(early), 1), 2),
                "avg_end_latency": round(sum(late) / max(len(late), 1), 2),
                "trend": "spiking" if late and early and
                         (sum(late)/len(late)) > (sum(early)/len(early)) * 2
                         else "elevated",
                "related_errors": len(
                    self._logs.search_messages("timeout", service=svc, level="ERROR")
                    + self._logs.search_messages("connection", service=svc, level="ERROR")
                    + self._logs.search_messages("socket", service=svc, level="ERROR")
                ),
            })
        return results

    # ── CPU anomalies ───────────────────────────────────────────

    def get_services_with_high_cpu(
        self, threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """Return services whose latest CPU exceeds *threshold*,
        enriched with trend data."""
        high_cpu = self._metrics.get_services_with_high_cpu(threshold)
        results = []
        for record in high_cpu:
            svc = record["service"]
            trend = self._metrics.get_metric_trend(svc, "cpu_percent")
            values = [p["value"] for p in trend if p["value"] is not None]
            early = values[:5]
            late = values[-5:]
            results.append({
                "service": svc,
                "current_cpu": record["cpu_percent"],
                "avg_start": round(sum(early) / max(len(early), 1), 2),
                "avg_end": round(sum(late) / max(len(late), 1), 2),
                "trend": "spiking" if late and early and
                         (sum(late)/len(late)) > (sum(early)/len(early)) * 2
                         else "elevated",
                "related_errors": len(
                    self._logs.search_messages("cpu", service=svc, level="ERROR")
                    + self._logs.search_messages("thread", service=svc, level="ERROR")
                    + self._logs.search_messages("timeout", service=svc, level="ERROR")
                ),
            })
        return results

    # ── top error generators ────────────────────────────────────

    def get_top_error_generators(
        self, n: int = 5
    ) -> List[Dict[str, Any]]:
        """Return the top *n* services generating the most ERROR logs,
        enriched with sample messages and metric snapshot."""
        top = self._logs.get_top_error_services(n)
        snapshot = self._metrics.get_latest_snapshot()
        results = []
        for svc, count in top:
            # grab up to 3 unique error messages as samples
            svc_errors = self._logs.search_messages("", service=svc, level="ERROR")
            sample_msgs = list(dict.fromkeys(
                l["message"] for l in svc_errors
            ))[:3]

            entry: Dict[str, Any] = {
                "service": svc,
                "error_count": count,
                "sample_messages": sample_msgs,
            }
            if svc in snapshot:
                m = snapshot[svc]
                entry["latest_cpu"] = m["cpu_percent"]
                entry["latest_memory"] = m["memory_percent"]
                entry["latest_latency"] = m["latency_ms"]
                entry["latest_error_rate"] = m["error_rate"]
            results.append(entry)
        return results

    # ── database impact ─────────────────────────────────────────

    def get_services_impacted_by_database_failure(self) -> List[Dict[str, Any]]:
        """Return services showing signs of database trouble:
        logs mentioning DB keywords + elevated latency or error rate."""
        db_keywords = ["database", "connection timeout", "deadlock",
                       "pool exhausted", "connection refused", "sql"]
        snapshot = self._metrics.get_latest_snapshot()
        impacted = []

        for svc in self._logs.services:
            total_db_hits = 0
            for kw in db_keywords:
                total_db_hits += len(
                    self._logs.search_messages(kw, service=svc)
                )
            if total_db_hits == 0:
                continue

            entry: Dict[str, Any] = {
                "service": svc,
                "db_related_log_count": total_db_hits,
            }
            if svc in snapshot:
                m = snapshot[svc]
                entry["latest_latency"] = m["latency_ms"]
                entry["latest_error_rate"] = m["error_rate"]
            impacted.append(entry)

        impacted.sort(key=lambda e: e["db_related_log_count"], reverse=True)
        return impacted

    # ── full health summary ─────────────────────────────────────

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Return a high-level snapshot of the entire system's health.

        This is the first thing an agent should call to orient itself.
        """
        snapshot = self._metrics.get_latest_snapshot()
        error_counts = self._logs.get_error_count_by_service()
        total_errors = sum(error_counts.values())
        total_logs = self._logs.total_logs

        # count services in distress (any metric above threshold)
        services_in_distress = set()
        for svc, m in snapshot.items():
            if (m["cpu_percent"] > 80 or m["memory_percent"] > 80
                    or m["latency_ms"] > 500 or m["error_rate"] > 5):
                services_in_distress.add(svc)

        return {
            "total_services": len(snapshot),
            "services_healthy": len(snapshot) - len(services_in_distress),
            "services_in_distress": sorted(services_in_distress),
            "total_metric_points": self._metrics.total_records,
            "total_log_entries": total_logs,
            "total_error_logs": total_errors,
            "error_rate_overall": round(
                (total_errors / max(total_logs, 1)) * 100, 2
            ),
            "top_error_service": (
                max(error_counts, key=error_counts.get)
                if error_counts else None
            ),
        }
