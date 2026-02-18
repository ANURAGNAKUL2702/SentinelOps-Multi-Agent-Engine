"""Pydantic v2 schemas for the Reporting & Analysis Layer.

Every model uses ``model_config = ConfigDict(frozen=True)`` for immutability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportFormat(str, Enum):
    """Supported output formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


class SeverityLevel(str, Enum):
    """Incident severity used in reports."""
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"


class IncidentStatus(str, Enum):
    """Incident lifecycle status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


class TrendDirection(str, Enum):
    """Direction of a metric trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class RecommendationCategory(str, Enum):
    """Category for actionable recommendations."""
    MONITORING = "monitoring"
    ARCHITECTURE = "architecture"
    PROCESS = "process"
    COST = "cost"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


# ---------------------------------------------------------------------------
# Report sub-models
# ---------------------------------------------------------------------------

class ExecutiveSummary(BaseModel):
    """One-paragraph TL;DR with key numbers."""
    model_config = ConfigDict(frozen=True)

    summary: str = Field(default="", min_length=0)
    severity: SeverityLevel = SeverityLevel.P2_MEDIUM
    status: IncidentStatus = IncidentStatus.RESOLVED
    affected_services: int = Field(default=0, ge=0)
    estimated_users_impacted: int = Field(default=0, ge=0)
    total_downtime_minutes: float = Field(default=0.0, ge=0.0)
    estimated_revenue_impact: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class IncidentDetails(BaseModel):
    """Core incident metadata."""
    model_config = ConfigDict(frozen=True)

    incident_id: str = ""
    started_at: Optional[datetime] = None
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    scenario_name: str = ""
    failure_type: str = ""
    affected_services: List[str] = Field(default_factory=list)
    primary_symptoms: List[str] = Field(default_factory=list)


class RootCauseAnalysis(BaseModel):
    """Root cause section of the report."""
    model_config = ConfigDict(frozen=True)

    root_cause: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_trail: List[Dict[str, Any]] = Field(default_factory=list)
    causal_chain: List[str] = Field(default_factory=list)
    alternative_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    validation_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucinations_detected: int = Field(default=0, ge=0)


class ActionItem(BaseModel):
    """A single remediation action item."""
    model_config = ConfigDict(frozen=True)

    priority: str = "P2"
    description: str = ""
    owner: str = "SRE Team"
    estimated_minutes: float = Field(default=15.0, ge=0.0)


class RemediationPlan(BaseModel):
    """Runbook and action-items section."""
    model_config = ConfigDict(frozen=True)

    runbook_title: str = ""
    runbook_steps: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    rollback_plan: str = ""
    estimated_resolution_minutes: float = Field(default=0.0, ge=0.0)


class TimelineEvent(BaseModel):
    """Single event on the incident timeline."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    event: str = ""
    service: str = ""
    severity: str = "info"


class IncidentTimeline(BaseModel):
    """Timeline section of the report."""
    model_config = ConfigDict(frozen=True)

    events: List[TimelineEvent] = Field(default_factory=list)
    critical_moments: List[str] = Field(default_factory=list)
    detection_delay_seconds: float = Field(default=0.0, ge=0.0)


class CostReport(BaseModel):
    """LLM cost breakdown."""
    model_config = ConfigDict(frozen=True)

    total_cost: float = Field(default=0.0, ge=0.0)
    cost_by_agent: Dict[str, float] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)
    tokens_by_agent: Dict[str, int] = Field(default_factory=dict)
    total_llm_calls: int = Field(default=0, ge=0)
    llm_calls_by_agent: Dict[str, int] = Field(default_factory=dict)
    cost_per_minute_downtime: float = Field(default=0.0, ge=0.0)
    cost_efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)


class PerformanceMetrics(BaseModel):
    """Pipeline performance section."""
    model_config = ConfigDict(frozen=True)

    total_pipeline_time: float = Field(default=0.0, ge=0.0)
    agent_latencies: Dict[str, float] = Field(default_factory=dict)
    parallel_speedup: float = Field(default=1.0, ge=0.0)
    timeout_violations: int = Field(default=0, ge=0)
    circuit_breaker_trips: int = Field(default=0, ge=0)
    retry_attempts: int = Field(default=0, ge=0)
    fallback_usage: Dict[str, bool] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """Single actionable recommendation."""
    model_config = ConfigDict(frozen=True)

    category: str = "process"
    description: str = ""
    priority: str = "P2"


class ReportMetadata(BaseModel):
    """Metadata about the generated report itself."""
    model_config = ConfigDict(frozen=True)

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    report_version: str = "1.0.0"
    generator_versions: Dict[str, str] = Field(default_factory=dict)


class HistoricalAnalytics(BaseModel):
    """Aggregated historical analytics."""
    model_config = ConfigDict(frozen=True)

    total_incidents: int = Field(default=0, ge=0)
    incidents_by_type: Dict[str, int] = Field(default_factory=dict)
    average_mttr_minutes: float = Field(default=0.0, ge=0.0)
    average_mttd_minutes: float = Field(default=0.0, ge=0.0)
    accuracy_trend: List[float] = Field(default_factory=list)
    cost_trend: List[float] = Field(default_factory=list)
    common_root_causes: List[Tuple[str, int]] = Field(default_factory=list)
    busiest_hours: List[int] = Field(default_factory=list)
    slo_compliance: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------

class IncidentReport(BaseModel):
    """Complete incident report encompassing all sections."""
    model_config = ConfigDict(frozen=True)

    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    executive_summary: ExecutiveSummary = Field(default_factory=ExecutiveSummary)
    incident_details: IncidentDetails = Field(default_factory=IncidentDetails)
    root_cause_analysis: RootCauseAnalysis = Field(default_factory=RootCauseAnalysis)
    remediation_plan: RemediationPlan = Field(default_factory=RemediationPlan)
    timeline: IncidentTimeline = Field(default_factory=IncidentTimeline)
    cost_report: CostReport = Field(default_factory=CostReport)
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    recommendations: List[Recommendation] = Field(default_factory=list)
    historical_analytics: Optional[HistoricalAnalytics] = None
    visualizations: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dashboard models
# ---------------------------------------------------------------------------

class KPICard(BaseModel):
    """Single KPI for executive dashboard."""
    model_config = ConfigDict(frozen=True)

    label: str = ""
    value: str = ""
    unit: str = ""
    trend: TrendDirection = TrendDirection.STABLE
    change_pct: float = 0.0


class DashboardData(BaseModel):
    """Data backing an executive dashboard."""
    model_config = ConfigDict(frozen=True)

    kpis: List[KPICard] = Field(default_factory=list)
    recent_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    charts: Dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    analytics: Optional[HistoricalAnalytics] = None


# ---------------------------------------------------------------------------
# Insight model
# ---------------------------------------------------------------------------

class Insight(BaseModel):
    """AI-generated or rule-based insight."""
    model_config = ConfigDict(frozen=True)

    category: str = ""
    title: str = ""
    description: str = ""
    priority: str = "P2"
    data: Dict[str, Any] = Field(default_factory=dict)
