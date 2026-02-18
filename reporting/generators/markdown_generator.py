"""Generate incident reports in Markdown using Jinja2 templates."""

from __future__ import annotations

import os
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import ReportingConfig
from ..schema import IncidentReport
from ..telemetry import get_logger

_logger = get_logger(__name__)


class MarkdownGenerator:
    """Render :class:`IncidentReport` as Markdown via Jinja2.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()
        self._env = Environment(
            loader=FileSystemLoader(self.config.templates_dir),
            autoescape=select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, report: IncidentReport) -> str:
        """Render *report* as a Markdown string.

        Args:
            report: The incident report model.

        Returns:
            Rendered Markdown text.

        Raises:
            FileNotFoundError: If the template file is missing.
        """
        template_path = "incident_report.md.j2"
        if not os.path.isfile(
            os.path.join(self.config.templates_dir, template_path)
        ):
            raise FileNotFoundError(
                f"Template not found: {os.path.join(self.config.templates_dir, template_path)}"
            )
        template = self._env.get_template(template_path)
        return template.render(report=report)

    def save(self, report: IncidentReport, path: str) -> str:
        """Render and save the Markdown report to *path*.

        Args:
            report: The incident report model.
            path: Destination file path.

        Returns:
            The absolute path where the file was saved.
        """
        content = self.generate(report)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        _logger.info("Markdown report saved", extra={"path": path})
        return os.path.abspath(path)
