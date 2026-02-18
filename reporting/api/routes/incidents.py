"""Incident history query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_repository
from ..models import IncidentDetailResponse, IncidentListResponse
from ...database.repository import IncidentRepository

router = APIRouter(tags=["incidents"])


@router.get(
    "/",
    response_model=IncidentListResponse,
    summary="List incidents",
)
def list_incidents(
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    days: int = Query(30, ge=1, le=365, description="Lookback window"),
    severity: str | None = Query(None, description="Filter by severity"),
    root_cause: str | None = Query(None, description="Filter by root cause substring"),
    repo: IncidentRepository = Depends(get_repository),
) -> IncidentListResponse:
    """Query incident history with optional filters."""
    incidents = repo.get_recent_incidents(limit=limit, days=days)

    # Apply optional in-memory filters
    if severity:
        incidents = [
            i for i in incidents
            if i.get("severity", "").upper() == severity.upper()
        ]
    if root_cause:
        lc = root_cause.lower()
        incidents = [
            i for i in incidents
            if lc in str(i.get("root_cause", "")).lower()
        ]

    # Serialise datetimes
    for inc in incidents:
        for key in ("started_at", "detected_at", "resolved_at", "created_at"):
            val = inc.get(key)
            if val is not None and hasattr(val, "isoformat"):
                inc[key] = val.isoformat()

    return IncidentListResponse(
        incidents=incidents,
        total=len(incidents),
        filters={
            "limit": limit,
            "days": days,
            "severity": severity,
            "root_cause": root_cause,
        },
    )


@router.get(
    "/{correlation_id}",
    response_model=IncidentDetailResponse,
    summary="Get incident by correlation ID",
)
def get_incident(
    correlation_id: str,
    repo: IncidentRepository = Depends(get_repository),
) -> IncidentDetailResponse:
    """Get detailed incident by correlation ID."""
    incident = repo.get_incident(correlation_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Serialise datetimes
    for key in ("started_at", "detected_at", "resolved_at", "created_at"):
        val = incident.get(key)
        if val is not None and hasattr(val, "isoformat"):
            incident[key] = val.isoformat()

    return IncidentDetailResponse(incident=incident)
