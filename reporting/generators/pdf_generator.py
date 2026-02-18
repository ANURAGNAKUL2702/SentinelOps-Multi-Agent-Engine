"""Generate professional multi-page PDF reports using reportlab."""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

from ..config import ReportingConfig
from ..schema import IncidentReport
from ..telemetry import get_logger

_logger = get_logger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _HAS_REPORTLAB = True
except ImportError:  # pragma: no cover
    _HAS_REPORTLAB = False


class PDFGenerator:
    """Create multi-page PDF incident reports.

    Falls back to an error message if ``reportlab`` is not installed.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    @property
    def available(self) -> bool:
        """Whether reportlab is installed and PDF generation is possible."""
        return _HAS_REPORTLAB

    # ------------------------------------------------------------------

    def generate(self, report: IncidentReport) -> bytes:
        """Render *report* as PDF bytes.

        Args:
            report: The incident report model.

        Returns:
            Raw PDF bytes.

        Raises:
            ImportError: If ``reportlab`` is unavailable.
        """
        if not _HAS_REPORTLAB:
            raise ImportError("reportlab is required for PDF generation")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            spaceAfter=8,
            spaceBefore=16,
            textColor=colors.HexColor("#4361ee"),
        )
        body_style = styles["BodyText"]

        elements: List[Any] = []

        # Title
        scenario = report.incident_details.scenario_name or "Incident Report"
        elements.append(Paragraph(scenario, title_style))
        elements.append(
            Paragraph(
                f"<b>ID:</b> {report.metadata.correlation_id} &bull; "
                f"<b>Severity:</b> {report.executive_summary.severity.value} &bull; "
                f"<b>Status:</b> {report.executive_summary.status.value}",
                body_style,
            )
        )
        elements.append(Spacer(1, 12))

        # Executive summary
        elements.append(Paragraph("Executive Summary", heading_style))
        elements.append(
            Paragraph(
                report.executive_summary.summary or "No summary available.",
                body_style,
            )
        )
        kpi_data = [
            ["Metric", "Value"],
            ["Affected Services", str(report.executive_summary.affected_services)],
            [
                "Confidence",
                f"{report.executive_summary.confidence * 100:.0f}%",
            ],
            [
                "Downtime",
                f"{report.executive_summary.total_downtime_minutes:.1f} min",
            ],
        ]
        elements.append(self._make_table(kpi_data))
        elements.append(Spacer(1, 8))

        # Root cause
        elements.append(Paragraph("Root Cause Analysis", heading_style))
        elements.append(
            Paragraph(
                f"<b>Root Cause:</b> {report.root_cause_analysis.root_cause or 'Unknown'}",
                body_style,
            )
        )
        elements.append(
            Paragraph(
                f"Confidence: {report.root_cause_analysis.confidence * 100:.0f}% "
                f"| Validation accuracy: {report.root_cause_analysis.validation_accuracy * 100:.0f}%",
                body_style,
            )
        )
        if report.root_cause_analysis.causal_chain:
            for idx, step in enumerate(report.root_cause_analysis.causal_chain, 1):
                elements.append(Paragraph(f"{idx}. {step}", body_style))
        elements.append(Spacer(1, 8))

        # Remediation
        elements.append(Paragraph("Remediation Plan", heading_style))
        if report.remediation_plan.runbook_steps:
            for idx, step in enumerate(report.remediation_plan.runbook_steps, 1):
                elements.append(Paragraph(f"{idx}. {step}", body_style))
        if report.remediation_plan.action_items:
            rows: List[List[str]] = [["Priority", "Description", "Owner", "Time"]]
            for ai in report.remediation_plan.action_items:
                rows.append(
                    [ai.priority, ai.description, ai.owner, f"{ai.estimated_minutes}m"]
                )
            elements.append(self._make_table(rows))
        elements.append(Spacer(1, 8))

        # Cost
        elements.append(Paragraph("Cost Report", heading_style))
        cost_rows: List[List[str]] = [["Metric", "Value"]]
        cost_rows.append(["Total Cost", f"${report.cost_report.total_cost:.4f}"])
        cost_rows.append(["Total Tokens", str(report.cost_report.total_tokens)])
        cost_rows.append(["LLM Calls", str(report.cost_report.total_llm_calls)])
        elements.append(self._make_table(cost_rows))
        elements.append(Spacer(1, 8))

        # Performance
        elements.append(Paragraph("Performance", heading_style))
        perf_rows: List[List[str]] = [["Metric", "Value"]]
        perf_rows.append(
            ["Pipeline Time", f"{report.performance_metrics.total_pipeline_time:.2f}s"]
        )
        perf_rows.append(
            ["Parallel Speedup", f"{report.performance_metrics.parallel_speedup:.1f}×"]
        )
        elements.append(self._make_table(perf_rows))

        # Recommendations
        if report.recommendations:
            elements.append(Paragraph("Recommendations", heading_style))
            for rec in report.recommendations:
                elements.append(
                    Paragraph(f"<b>[{rec.category}]</b> {rec.description}", body_style)
                )

        doc.build(elements)
        return buf.getvalue()

    # ------------------------------------------------------------------

    def save(self, report: IncidentReport, path: str) -> str:
        """Generate and save the PDF report to *path*.

        Args:
            report: The incident report model.
            path: Destination file path.

        Returns:
            Absolute path where the file was saved.
        """
        data = self.generate(report)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        _logger.info("PDF report saved", extra={"path": path})
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_table(data: List[List[str]]) -> Table:
        """Build a styled reportlab Table.

        Args:
            data: Row data including a header row.

        Returns:
            Styled :class:`Table`.
        """
        t = Table(data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4361ee")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ("TOPPADDING", (0, 1), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ]
            )
        )
        return t
