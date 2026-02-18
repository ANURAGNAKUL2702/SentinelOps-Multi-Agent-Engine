"""Tests for database ORM models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from reporting.database.models import Base, Incident, AgentExecution, CostRecord, Metric


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as sess:
        yield sess


class TestIncidentModel:
    def test_create_incident(self, session: Session) -> None:
        inc = Incident(
            correlation_id="test-corr-1",
            duration=5.0,
            root_cause="DB pool exhaustion",
            confidence=0.9,
            severity="P1_HIGH",
            status="resolved",
        )
        session.add(inc)
        session.commit()
        assert inc.id is not None
        assert inc.correlation_id == "test-corr-1"

    def test_incident_defaults(self, session: Session) -> None:
        inc = Incident(correlation_id="defaults-test")
        session.add(inc)
        session.commit()
        assert inc.total_cost == 0.0
        assert inc.total_tokens == 0

    def test_incident_relationships(self, session: Session) -> None:
        inc = Incident(correlation_id="rel-test")
        session.add(inc)
        session.flush()

        ae = AgentExecution(incident_id=inc.id, agent_name="log_agent", duration=1.5)
        cr = CostRecord(incident_id=inc.id, agent_name="log_agent", cost=0.01)
        m = Metric(incident_id=inc.id, metric_name="tokens", metric_value=500)
        session.add_all([ae, cr, m])
        session.commit()

        assert len(inc.agent_executions) == 1
        assert len(inc.cost_records) == 1
        assert len(inc.metrics) == 1

    def test_cascade_delete(self, session: Session) -> None:
        inc = Incident(correlation_id="cascade-test")
        session.add(inc)
        session.flush()
        session.add(AgentExecution(incident_id=inc.id, agent_name="x", duration=0.5))
        session.commit()

        session.delete(inc)
        session.commit()
        assert session.query(AgentExecution).count() == 0


class TestAgentExecutionModel:
    def test_create(self, session: Session) -> None:
        inc = Incident(correlation_id="ae-test")
        session.add(inc)
        session.flush()
        ae = AgentExecution(
            incident_id=inc.id,
            agent_name="rca_agent",
            duration=2.3,
            status="success",
            cost=0.02,
            tokens=1500,
        )
        session.add(ae)
        session.commit()
        assert ae.id is not None
        assert ae.agent_name == "rca_agent"


class TestCostRecordModel:
    def test_create(self, session: Session) -> None:
        inc = Incident(correlation_id="cr-test")
        session.add(inc)
        session.flush()
        cr = CostRecord(
            incident_id=inc.id,
            agent_name="hyp_agent",
            tokens_input=500,
            tokens_output=200,
            cost=0.005,
        )
        session.add(cr)
        session.commit()
        assert cr.cost == 0.005


class TestMetricModel:
    def test_create(self, session: Session) -> None:
        inc = Incident(correlation_id="m-test")
        session.add(inc)
        session.flush()
        m = Metric(
            incident_id=inc.id,
            metric_name="p95_latency",
            metric_value=2.5,
        )
        session.add(m)
        session.commit()
        assert m.metric_value == 2.5
