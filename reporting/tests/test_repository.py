"""Tests for reporting.database.repository."""

from __future__ import annotations

import pytest

from reporting.config import ReportingConfig
from reporting.database.connection import DatabaseConnection
from reporting.database.repository import IncidentRepository


@pytest.fixture
def repo(tmp_path) -> IncidentRepository:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    cfg = ReportingConfig(database_url=db_url)
    conn = DatabaseConnection(cfg)
    conn.create_tables()
    return IncidentRepository(conn)


class TestInsertAndGet:
    def test_insert_and_get(self, repo: IncidentRepository) -> None:
        iid = repo.insert_incident(
            correlation_id="corr-1",
            duration=5.0,
            root_cause="Memory leak",
            confidence=0.9,
            severity="P1_HIGH",
            total_cost=0.05,
            total_tokens=5000,
        )
        assert isinstance(iid, int)
        inc = repo.get_incident("corr-1")
        assert inc is not None
        assert inc["root_cause"] == "Memory leak"
        assert inc["confidence"] == 0.9

    def test_get_nonexistent(self, repo: IncidentRepository) -> None:
        assert repo.get_incident("nope") is None

    def test_insert_with_children(self, repo: IncidentRepository) -> None:
        iid = repo.insert_incident(
            correlation_id="corr-children",
            duration=3.0,
            root_cause="CPU spike",
            agent_executions=[
                {"agent_name": "log_agent", "duration": 1.0, "status": "success"},
                {"agent_name": "rca_agent", "duration": 2.0, "status": "success"},
            ],
            cost_records=[
                {"agent_name": "log_agent", "tokens_input": 100, "tokens_output": 50, "cost": 0.001},
            ],
            metrics=[
                {"metric_name": "tokens_total", "metric_value": 150},
            ],
        )
        assert isinstance(iid, int)


class TestQueries:
    def test_recent_incidents(self, repo: IncidentRepository) -> None:
        for i in range(5):
            repo.insert_incident(
                correlation_id=f"recent-{i}",
                duration=float(i),
                root_cause="test",
            )
        results = repo.get_recent_incidents(limit=3, days=30)
        assert len(results) == 3

    def test_incidents_by_root_cause(self, repo: IncidentRepository) -> None:
        repo.insert_incident(correlation_id="rc-1", root_cause="DB pool")
        repo.insert_incident(correlation_id="rc-2", root_cause="DB pool")
        repo.insert_incident(correlation_id="rc-3", root_cause="OOM")
        results = repo.get_incidents_by_root_cause("DB pool")
        assert len(results) == 2

    def test_incident_count(self, repo: IncidentRepository) -> None:
        assert repo.get_incident_count(days=30) == 0
        repo.insert_incident(correlation_id="cnt-1")
        assert repo.get_incident_count(days=30) == 1


class TestAggregations:
    def test_calculate_mttr(self, repo: IncidentRepository) -> None:
        for i in range(3):
            repo.insert_incident(
                correlation_id=f"mttr-{i}",
                duration=600.0,  # 10 minutes each
            )
        mttr = repo.calculate_mttr(days=30)
        assert abs(mttr - 10.0) < 0.01

    def test_mttr_empty(self, repo: IncidentRepository) -> None:
        assert repo.calculate_mttr(days=30) == 0.0

    def test_common_root_causes(self, repo: IncidentRepository) -> None:
        for _ in range(3):
            repo.insert_incident(correlation_id=f"rc-a-{_}", root_cause="DB pool")
        for _ in range(2):
            repo.insert_incident(correlation_id=f"rc-b-{_}", root_cause="OOM")
        causes = repo.get_common_root_causes(limit=5)
        assert causes[0] == ("DB pool", 3)
        assert causes[1] == ("OOM", 2)

    def test_cost_summary(self, repo: IncidentRepository) -> None:
        repo.insert_incident(correlation_id="cost-1", total_cost=0.01)
        repo.insert_incident(correlation_id="cost-2", total_cost=0.03)
        summary = repo.get_cost_summary(days=30)
        assert abs(summary["total_cost"] - 0.04) < 0.001
        assert summary["max_cost"] == 0.03

    def test_slo_compliance(self, repo: IncidentRepository) -> None:
        repo.insert_incident(correlation_id="slo-1", duration=5.0)
        repo.insert_incident(correlation_id="slo-2", duration=15.0)
        repo.insert_incident(correlation_id="slo-3", duration=25.0)
        compliance = repo.get_slo_compliance(slo_seconds=20.0, days=30)
        assert abs(compliance - 2.0 / 3.0) < 0.01

    def test_all_durations(self, repo: IncidentRepository) -> None:
        repo.insert_incident(correlation_id="dur-1", duration=5.0)
        repo.insert_incident(correlation_id="dur-2", duration=10.0)
        durations = repo.get_all_durations(days=30)
        assert len(durations) == 2


class TestDelete:
    def test_delete_old_incidents(self, repo: IncidentRepository) -> None:
        repo.insert_incident(correlation_id="old-1", duration=1.0)
        # With retention_days=0, everything is "old"
        deleted = repo.delete_old_incidents(days=0)
        assert deleted >= 1
