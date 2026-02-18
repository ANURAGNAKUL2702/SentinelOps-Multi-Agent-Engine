"""Analytics endpoints — dashboard, trends, costs, insights."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_repository, get_config
from ..models import (
    CostAnalysisResponse,
    DashboardResponse,
    InsightsResponse,
    TrendsResponse,
)
from ...database.repository import IncidentRepository
from ...analysis.cost_analyzer import CostAnalyzer
from ...analysis.insight_generator import InsightGenerator
from ...analysis.mttr_calculator import MTTRCalculator
from ...analysis.trend_analyzer import TrendAnalyzer

router = APIRouter(tags=["analytics"])


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Dashboard KPIs",
)
def dashboard(
    days: int = Query(30, ge=1, le=365),
    repo: IncidentRepository = Depends(get_repository),
) -> DashboardResponse:
    """Aggregate dashboard KPIs: incidents, MTTR, MTTD, SLO, costs."""
    cfg = get_config()
    total = repo.get_incident_count(days=days)
    mttr = repo.calculate_mttr(days=days)
    mttd = repo.calculate_mttd(days=days)
    slo = repo.get_slo_compliance(slo_seconds=cfg.slo_resolution_time, days=days)
    cost_summary = repo.get_cost_summary(days=days)
    root_causes = repo.get_common_root_causes(limit=5)

    return DashboardResponse(
        total_incidents=total,
        mttr=round(mttr, 4),
        mttd=round(mttd, 4),
        slo_compliance=round(slo, 4),
        total_cost=round(cost_summary["total_cost"], 6),
        common_root_causes=[
            {"root_cause": rc, "count": cnt} for rc, cnt in root_causes
        ],
    )


@router.get(
    "/trends",
    response_model=TrendsResponse,
    summary="Metric trends",
)
def trends(
    days: int = Query(90, ge=1, le=365),
    metric: str = Query("duration", description="Metric name"),
    repo: IncidentRepository = Depends(get_repository),
) -> TrendsResponse:
    """Analyse time-series trend for a specified metric."""
    incidents = repo.get_recent_incidents(limit=10000, days=days)
    analyzer = TrendAnalyzer()
    results = analyzer.analyze_incidents(incidents)

    result = results.get(metric)
    if result is None:
        # Try as raw values
        vals = [float(i.get(metric, 0.0)) for i in incidents]
        result = analyzer.analyze_metric(vals, metric_name=metric)

    return TrendsResponse(
        metric=result.metric_name,
        direction=result.direction,
        slope=result.slope,
        r_squared=result.r_squared,
        data_points=result.data_points,
        values=result.values,
        timestamps=result.timestamps,
    )


@router.get(
    "/costs",
    response_model=CostAnalysisResponse,
    summary="Cost analysis",
)
def costs(
    days: int = Query(30, ge=1, le=365),
    repo: IncidentRepository = Depends(get_repository),
) -> CostAnalysisResponse:
    """Cost breakdown and analysis."""
    cost_summary = repo.get_cost_summary(days=days)
    incidents = repo.get_recent_incidents(limit=10000, days=days)

    # Build pseudo cost records from incidents for the analyzer
    cost_records = [
        {"agent_name": "pipeline", "cost": float(i.get("total_cost", 0.0)),
         "tokens_input": int(i.get("total_tokens", 0)), "tokens_output": 0}
        for i in incidents
    ]

    analyzer = CostAnalyzer()
    summary = analyzer.analyze(cost_records)

    return CostAnalysisResponse(
        total_cost=round(cost_summary["total_cost"], 6),
        avg_cost=round(cost_summary["avg_cost"], 6),
        max_cost=round(cost_summary["max_cost"], 6),
        cost_by_agent=summary.cost_by_agent,
        outlier_count=len(summary.outliers),
    )


@router.get(
    "/insights",
    response_model=InsightsResponse,
    summary="AI-powered insights",
)
def insights(
    days: int = Query(30, ge=1, le=365),
    repo: IncidentRepository = Depends(get_repository),
) -> InsightsResponse:
    """Generate rule-based insights from historical incident data."""
    cfg = get_config()
    incidents = repo.get_recent_incidents(limit=10000, days=days)
    total = len(incidents)
    durations = repo.get_all_durations(days=days)
    cost_summary = repo.get_cost_summary(days=days)
    root_causes = repo.get_common_root_causes(limit=5)

    mttr_calc = MTTRCalculator(target_minutes=cfg.target_mttr)
    mttr_summary = mttr_calc.calculate(durations)

    slo = repo.get_slo_compliance(slo_seconds=cfg.slo_resolution_time, days=days)

    generator = InsightGenerator()
    insight_list = generator.generate(
        avg_duration=mttr_summary.mean_minutes * 60,
        target_duration=cfg.target_pipeline_time,
        total_cost=cost_summary["total_cost"],
        slo_compliance=slo,
        mttr_minutes=mttr_summary.mean_minutes,
        mttr_target=cfg.target_mttr,
        common_root_causes=[(rc, cnt) for rc, cnt in root_causes],
    )

    return InsightsResponse(
        insights=[
            {
                "category": ins.category,
                "severity": ins.severity,
                "title": ins.title,
                "description": ins.description,
                "recommendation": ins.recommendation,
            }
            for ins in insight_list
        ],
        based_on_incidents=total,
    )
