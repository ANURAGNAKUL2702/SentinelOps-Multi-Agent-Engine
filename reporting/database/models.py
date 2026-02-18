"""SQLAlchemy ORM models for the incident time-series database."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class Incident(Base):
    """An incident record persisted from a pipeline run."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    correlation_id = Column(String(64), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True, index=True)
    detected_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    duration = Column(Float, default=0.0)
    root_cause = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    severity = Column(String(32), default="P2_MEDIUM")
    status = Column(String(32), default="resolved")
    total_cost = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_llm_calls = Column(Integer, default=0)
    affected_services_count = Column(Integer, default=0)
    validation_accuracy = Column(Float, default=0.0)
    scenario_name = Column(String(256), default="")
    failure_type = Column(String(128), default="")
    pipeline_status = Column(String(32), default="success")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False,
    )

    # Relationships
    agent_executions = relationship(
        "AgentExecution", back_populates="incident", cascade="all, delete-orphan",
    )
    cost_records = relationship(
        "CostRecord", back_populates="incident", cascade="all, delete-orphan",
    )
    metrics = relationship(
        "Metric", back_populates="incident", cascade="all, delete-orphan",
    )

    __table_args__: tuple = ()


class AgentExecution(Base):
    """Execution record for a single agent within an incident."""

    __tablename__ = "agent_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    agent_name = Column(String(64), nullable=False, index=True)
    duration = Column(Float, default=0.0)
    status = Column(String(32), default="success")
    cost = Column(Float, default=0.0)
    tokens = Column(Integer, default=0)
    error_type = Column(String(64), nullable=True)
    retry_count = Column(Integer, default=0)
    used_fallback = Column(Boolean, default=False)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False,
    )

    incident = relationship("Incident", back_populates="agent_executions")

    __table_args__: tuple = ()


class CostRecord(Base):
    """Per-agent cost breakdown for an incident."""

    __tablename__ = "cost_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    agent_name = Column(String(64), nullable=False)
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False,
    )

    incident = relationship("Incident", back_populates="cost_records")


class Metric(Base):
    """Arbitrary named metric associated with an incident."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False)
    metric_value = Column(Float, default=0.0)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False,
    )

    incident = relationship("Incident", back_populates="metrics")
