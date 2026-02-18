"""API-specific Pydantic v2 request / response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """POST /api/v1/reports/generate request body."""
    model_config = ConfigDict(frozen=True)

    pipeline_result: Dict[str, Any] = Field(
        ..., description="Serialised PipelineResult dictionary."
    )
    formats: List[str] = Field(
        default=["html"],
        description="Output formats (html, markdown, json, pdf).",
    )
    save_to_database: bool = Field(
        default=True,
        description="Persist incident to the database.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class GenerateReportResponse(BaseModel):
    """POST /api/v1/reports/generate response."""
    model_config = ConfigDict(frozen=True)

    report_id: str
    correlation_id: str
    formats: List[str]
    files: Dict[str, str] = Field(default_factory=dict)


class ReportMetadataResponse(BaseModel):
    """GET /api/v1/reports/{report_id} response."""
    model_config = ConfigDict(frozen=True)

    report_id: str
    correlation_id: str
    generated_at: str
    formats: List[str]
    files: Dict[str, str] = Field(default_factory=dict)


class IncidentListResponse(BaseModel):
    """GET /api/v1/incidents response."""
    model_config = ConfigDict(frozen=True)

    incidents: List[Dict[str, Any]]
    total: int
    filters: Dict[str, Any] = Field(default_factory=dict)


class IncidentDetailResponse(BaseModel):
    """GET /api/v1/incidents/{correlation_id} response."""
    model_config = ConfigDict(frozen=True)

    incident: Dict[str, Any]


class DashboardResponse(BaseModel):
    """GET /api/v1/analytics/dashboard response."""
    model_config = ConfigDict(frozen=True)

    total_incidents: int = 0
    mttr: float = 0.0
    mttd: float = 0.0
    slo_compliance: float = 1.0
    total_cost: float = 0.0
    common_root_causes: List[Dict[str, Any]] = Field(default_factory=list)


class TrendsResponse(BaseModel):
    """GET /api/v1/analytics/trends response."""
    model_config = ConfigDict(frozen=True)

    metric: str
    direction: str = "stable"
    slope: float = 0.0
    r_squared: float = 0.0
    data_points: int = 0
    values: List[float] = Field(default_factory=list)
    timestamps: List[str] = Field(default_factory=list)


class CostAnalysisResponse(BaseModel):
    """GET /api/v1/analytics/costs response."""
    model_config = ConfigDict(frozen=True)

    total_cost: float = 0.0
    avg_cost: float = 0.0
    max_cost: float = 0.0
    cost_by_agent: Dict[str, float] = Field(default_factory=dict)
    outlier_count: int = 0


class InsightsResponse(BaseModel):
    """GET /api/v1/analytics/insights response."""
    model_config = ConfigDict(frozen=True)

    insights: List[Dict[str, str]] = Field(default_factory=list)
    based_on_incidents: int = 0


class HealthResponse(BaseModel):
    """GET /api/v1/health response."""
    model_config = ConfigDict(frozen=True)

    status: str = "healthy"
    database: str = "connected"
    version: str = "1.0.0"
    uptime: float = 0.0
