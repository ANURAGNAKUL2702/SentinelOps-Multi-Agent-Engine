"""Incident repository — CRUD, queries, and aggregations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select, delete
from sqlalchemy.orm import Session

from .connection import DatabaseConnection
from .models import AgentExecution, CostRecord, Incident, Metric
from ..telemetry import get_logger

_logger = get_logger(__name__)


class IncidentRepository:
    """High-level data-access layer over the incident database.

    Args:
        connection: A :class:`DatabaseConnection` instance.
    """

    def __init__(self, connection: DatabaseConnection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_incident(
        self,
        *,
        correlation_id: str,
        started_at: Optional[datetime] = None,
        detected_at: Optional[datetime] = None,
        resolved_at: Optional[datetime] = None,
        duration: float = 0.0,
        root_cause: str = "",
        confidence: float = 0.0,
        severity: str = "P2_MEDIUM",
        status: str = "resolved",
        total_cost: float = 0.0,
        total_tokens: int = 0,
        total_llm_calls: int = 0,
        affected_services_count: int = 0,
        validation_accuracy: float = 0.0,
        scenario_name: str = "",
        failure_type: str = "",
        pipeline_status: str = "success",
        agent_executions: Optional[List[Dict[str, Any]]] = None,
        cost_records: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Insert an incident (and related child records) into the database.

        Args:
            correlation_id: Unique correlation ID from the pipeline.
            started_at: Incident start time.
            detected_at: Detection time.
            resolved_at: Resolution time.
            duration: Pipeline execution duration in seconds.
            root_cause: Identified root cause.
            confidence: Root cause confidence.
            severity: Severity label.
            status: Incident status.
            total_cost: Total LLM cost.
            total_tokens: Total tokens used.
            total_llm_calls: Total LLM API calls.
            affected_services_count: Number of affected services.
            validation_accuracy: Validation accuracy score.
            scenario_name: Scenario/incident name.
            failure_type: Type of failure.
            pipeline_status: Pipeline outcome (success/failed/etc).
            agent_executions: Optional list of agent execution dicts.
            cost_records: Optional list of cost record dicts.
            metrics: Optional list of metric dicts.

        Returns:
            The auto-generated incident ID.
        """
        with self._conn.session() as sess:
            incident = Incident(
                correlation_id=correlation_id,
                started_at=started_at,
                detected_at=detected_at,
                resolved_at=resolved_at,
                duration=duration,
                root_cause=root_cause,
                confidence=confidence,
                severity=severity,
                status=status,
                total_cost=total_cost,
                total_tokens=total_tokens,
                total_llm_calls=total_llm_calls,
                affected_services_count=affected_services_count,
                validation_accuracy=validation_accuracy,
                scenario_name=scenario_name,
                failure_type=failure_type,
                pipeline_status=pipeline_status,
            )
            sess.add(incident)
            sess.flush()

            if agent_executions:
                for ae in agent_executions:
                    sess.add(AgentExecution(incident_id=incident.id, **ae))
            if cost_records:
                for cr in cost_records:
                    sess.add(CostRecord(incident_id=incident.id, **cr))
            if metrics:
                for m in metrics:
                    sess.add(Metric(incident_id=incident.id, **m))

            sess.flush()
            return incident.id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_incident(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an incident by correlation ID.

        Args:
            correlation_id: The pipeline correlation ID.

        Returns:
            A dictionary of incident fields, or ``None``.
        """
        with self._conn.session() as sess:
            stmt = select(Incident).where(Incident.correlation_id == correlation_id)
            row = sess.execute(stmt).scalar_one_or_none()
            if row is None:
                return None
            return self._incident_to_dict(row)

    def get_recent_incidents(
        self, limit: int = 20, days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return the most recent incidents.

        Args:
            limit: Maximum number to return.
            days: Only include incidents from the last *days* days.

        Returns:
            List of incident dictionaries ordered newest-first.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = (
                select(Incident)
                .where(Incident.created_at >= cutoff)
                .order_by(Incident.created_at.desc())
                .limit(limit)
            )
            rows = sess.execute(stmt).scalars().all()
            return [self._incident_to_dict(r) for r in rows]

    def get_incidents_by_root_cause(
        self, root_cause: str,
    ) -> List[Dict[str, Any]]:
        """Return incidents matching *root_cause*.

        Args:
            root_cause: Root cause string to filter by.

        Returns:
            List of matching incident dictionaries.
        """
        with self._conn.session() as sess:
            stmt = select(Incident).where(Incident.root_cause == root_cause)
            rows = sess.execute(stmt).scalars().all()
            return [self._incident_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def calculate_mttr(self, days: int = 30) -> float:
        """Calculate Mean Time To Resolve (minutes) over the last *days* days.

        Args:
            days: Lookback window.

        Returns:
            MTTR in minutes, or 0.0 if no data.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = select(func.avg(Incident.duration)).where(
                Incident.created_at >= cutoff,
                Incident.duration > 0,
            )
            avg_sec = sess.execute(stmt).scalar_one_or_none()
            if avg_sec is None:
                return 0.0
            return float(avg_sec) / 60.0

    def calculate_mttd(self, days: int = 30) -> float:
        """Calculate Mean Time To Detect (minutes).

        Uses the pipeline execution time as a proxy for detection time.

        Args:
            days: Lookback window.

        Returns:
            MTTD in minutes, or 0.0.
        """
        # In our model, detection time is the pipeline duration itself
        return self.calculate_mttr(days)

    def get_common_root_causes(
        self, limit: int = 10,
    ) -> List[Tuple[str, int]]:
        """Return the top *limit* root causes by frequency.

        Args:
            limit: Number of root causes to return.

        Returns:
            List of ``(root_cause, count)`` tuples.
        """
        with self._conn.session() as sess:
            stmt = (
                select(Incident.root_cause, func.count(Incident.id).label("cnt"))
                .where(Incident.root_cause != "")
                .group_by(Incident.root_cause)
                .order_by(func.count(Incident.id).desc())
                .limit(limit)
            )
            rows = sess.execute(stmt).all()
            return [(str(r[0]), int(r[1])) for r in rows]

    def get_cost_summary(self, days: int = 30) -> Dict[str, float]:
        """Aggregate costs over the last *days* days.

        Args:
            days: Lookback window.

        Returns:
            Dict with ``total_cost``, ``avg_cost``, ``max_cost``.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = select(
                func.sum(Incident.total_cost),
                func.avg(Incident.total_cost),
                func.max(Incident.total_cost),
            ).where(Incident.created_at >= cutoff)
            row = sess.execute(stmt).one()
            return {
                "total_cost": float(row[0] or 0.0),
                "avg_cost": float(row[1] or 0.0),
                "max_cost": float(row[2] or 0.0),
            }

    def get_all_durations(self, days: int = 30) -> List[float]:
        """Return all incident durations (seconds) for percentile calculations.

        Args:
            days: Lookback window.

        Returns:
            List of duration floats.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = (
                select(Incident.duration)
                .where(Incident.created_at >= cutoff, Incident.duration > 0)
            )
            rows = sess.execute(stmt).scalars().all()
            return [float(d) for d in rows]

    def get_incident_count(self, days: int = 30) -> int:
        """Return total incident count in the last *days* days.

        Args:
            days: Lookback window.

        Returns:
            Count of incidents.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = select(func.count(Incident.id)).where(
                Incident.created_at >= cutoff,
            )
            return int(sess.execute(stmt).scalar_one())

    def get_slo_compliance(
        self, slo_seconds: float, days: int = 30,
    ) -> float:
        """Calculate SLO compliance ratio.

        Args:
            slo_seconds: SLO target in seconds.
            days: Lookback window.

        Returns:
            Compliance ratio ∈ [0, 1].
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            total = sess.execute(
                select(func.count(Incident.id)).where(
                    Incident.created_at >= cutoff,
                )
            ).scalar_one()
            if total == 0:
                return 1.0
            within_slo = sess.execute(
                select(func.count(Incident.id)).where(
                    Incident.created_at >= cutoff,
                    Incident.duration <= slo_seconds,
                )
            ).scalar_one()
            return float(within_slo) / float(total)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_old_incidents(self, days: int = 90) -> int:
        """Delete incidents older than *days* days.

        Args:
            days: Retention period.

        Returns:
            Number of deleted rows.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._conn.session() as sess:
            stmt = delete(Incident).where(Incident.created_at < cutoff)
            result = sess.execute(stmt)
            return result.rowcount  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _incident_to_dict(row: Incident) -> Dict[str, Any]:
        return {
            "id": row.id,
            "correlation_id": row.correlation_id,
            "started_at": row.started_at,
            "detected_at": row.detected_at,
            "resolved_at": row.resolved_at,
            "duration": row.duration,
            "root_cause": row.root_cause,
            "confidence": row.confidence,
            "severity": row.severity,
            "status": row.status,
            "total_cost": row.total_cost,
            "total_tokens": row.total_tokens,
            "total_llm_calls": row.total_llm_calls,
            "affected_services_count": row.affected_services_count,
            "validation_accuracy": row.validation_accuracy,
            "scenario_name": row.scenario_name,
            "failure_type": row.failure_type,
            "pipeline_status": row.pipeline_status,
            "created_at": row.created_at,
        }
