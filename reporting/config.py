"""Reporting configuration — frozen dataclass with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ReportingConfig:
    """Immutable configuration for the Reporting & Analysis Layer.

    Attributes:
        default_format: Default output format (``"html"``, ``"markdown"``, ``"json"``, ``"pdf"``).
        include_visualizations: Whether to embed charts in reports.
        include_cost_breakdown: Whether to include cost breakdown section.
        database_url: SQLAlchemy connection string.
        enable_database: Whether to persist incidents to DB.
        retention_days: How long to keep incidents in the DB.
        chart_width: Chart width in pixels.
        chart_height: Chart height in pixels.
        chart_dpi: Chart resolution in DPI.
        theme: ``"light"`` or ``"dark"``.
        groq_cost_per_1k_input_tokens: LLM input token cost.
        groq_cost_per_1k_output_tokens: LLM output token cost.
        target_pipeline_time: Target pipeline execution seconds.
        target_mttr: Target MTTR in minutes.
        slo_resolution_time: SLO resolution target in minutes.
        enable_prometheus_export: Export Prometheus metrics.
        enable_slack_notifications: Post to Slack.
        slack_webhook_url: Slack incoming webhook URL.
        enable_pagerduty: Create PagerDuty incidents.
        pagerduty_api_key: PagerDuty API key.
        enable_ai_insights: Generate AI-powered insights.
        templates_dir: Path to Jinja2 templates.
        output_dir: Path to write generated reports.
    """

    # Output ------------------------------------------------------------------
    default_format: str = "html"
    include_visualizations: bool = True
    include_cost_breakdown: bool = True

    # Database ----------------------------------------------------------------
    database_url: str = "sqlite:///incidents.db"
    enable_database: bool = True
    retention_days: int = 90

    # Charts ------------------------------------------------------------------
    chart_width: int = 800
    chart_height: int = 400
    chart_dpi: int = 100
    theme: str = "light"

    # Cost rates --------------------------------------------------------------
    groq_cost_per_1k_input_tokens: float = 0.0006
    groq_cost_per_1k_output_tokens: float = 0.0006

    # Targets / SLOs ----------------------------------------------------------
    target_pipeline_time: float = 10.0
    target_mttr: float = 15.0
    slo_resolution_time: float = 30.0

    # Integrations ------------------------------------------------------------
    enable_prometheus_export: bool = True
    enable_slack_notifications: bool = False
    slack_webhook_url: Optional[str] = None
    enable_pagerduty: bool = False
    pagerduty_api_key: Optional[str] = None

    # AI insights -------------------------------------------------------------
    enable_ai_insights: bool = True

    # Paths -------------------------------------------------------------------
    templates_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "templates",
        ),
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "reports",
        ),
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.retention_days < 1:
            raise ValueError("retention_days must be >= 1")
        if self.chart_dpi < 50:
            raise ValueError("chart_dpi must be >= 50")
        if self.slo_resolution_time <= 0:
            raise ValueError("slo_resolution_time must be > 0")
