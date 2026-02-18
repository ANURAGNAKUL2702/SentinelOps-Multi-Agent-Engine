"""Report generation and retrieval endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..dependencies import get_report_builder, get_repository
from ..models import GenerateReportRequest, GenerateReportResponse, ReportMetadataResponse
from ...report_builder import ReportBuilder
from ...database.repository import IncidentRepository

router = APIRouter(tags=["reports"])

VALID_FORMATS = {"html", "markdown", "json", "pdf"}


@router.post(
    "/generate",
    response_model=GenerateReportResponse,
    summary="Generate an incident report",
    status_code=200,
)
def generate_report(
    request: GenerateReportRequest,
    builder: ReportBuilder = Depends(get_report_builder),
    repo: IncidentRepository = Depends(get_repository),
) -> GenerateReportResponse:
    """Generate an incident report from a pipeline result."""
    # Validate requested formats
    for fmt in request.formats:
        if fmt.lower() not in VALID_FORMATS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid format '{fmt}'. Valid: {sorted(VALID_FORMATS)}",
            )

    report = builder.build_report(
        request.pipeline_result,
        formats=request.formats,
    )

    files: dict[str, str] = {}
    for fmt in request.formats:
        try:
            path = builder.save(report, fmt=fmt.lower())
            files[fmt.lower()] = path
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Optionally persist to database
    if request.save_to_database:
        try:
            pr = request.pipeline_result
            telemetry = pr.get("telemetry") or {}
            repo.insert_incident(
                correlation_id=pr.get("correlation_id", report.metadata.correlation_id),
                duration=float(pr.get("execution_time", 0.0)),
                root_cause=report.root_cause_analysis.root_cause,
                confidence=report.root_cause_analysis.confidence,
                severity=report.executive_summary.severity.value,
                total_cost=float(telemetry.get("total_llm_cost", 0.0)),
                total_tokens=int(telemetry.get("total_tokens", 0)),
                total_llm_calls=int(telemetry.get("total_llm_calls", 0)),
                pipeline_status=str(pr.get("status", "success")),
                validation_accuracy=report.root_cause_analysis.validation_accuracy,
            )
        except Exception:
            pass  # DB persistence is best-effort

    return GenerateReportResponse(
        report_id=report.metadata.report_id,
        correlation_id=report.metadata.correlation_id,
        formats=request.formats,
        files=files,
    )


@router.get(
    "/{report_id}",
    response_model=ReportMetadataResponse,
    summary="Get report metadata",
)
def get_report(
    report_id: str,
    builder: ReportBuilder = Depends(get_report_builder),
) -> ReportMetadataResponse:
    """Retrieve metadata for a previously generated report."""
    meta = builder.get_report_metadata(report_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportMetadataResponse(**meta)


@router.get(
    "/{report_id}/download/{fmt}",
    summary="Download report file",
)
def download_report(
    report_id: str,
    fmt: str,
    builder: ReportBuilder = Depends(get_report_builder),
) -> FileResponse:
    """Download a generated report file."""
    if fmt.lower() not in VALID_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid format '{fmt}'. Valid: {sorted(VALID_FORMATS)}",
        )

    file_path = builder.get_report_file(report_id, fmt.lower())
    if file_path is None or not Path(file_path).is_file():
        raise HTTPException(status_code=404, detail="Report file not found")

    media = {
        "html": "text/html",
        "markdown": "text/markdown",
        "json": "application/json",
        "pdf": "application/pdf",
    }.get(fmt.lower(), "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media,
        filename=Path(file_path).name,
    )
