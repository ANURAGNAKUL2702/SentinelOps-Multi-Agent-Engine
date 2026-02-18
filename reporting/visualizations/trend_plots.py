"""Trend plots — line charts for historical trends (incidents, costs, accuracy)."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..config import ReportingConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


class TrendPlots:
    """Generate line plots for time-series trend data.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    def generate_line_chart(
        self,
        data: List[Tuple[datetime, float]],
        title: str = "Trend",
        ylabel: str = "Value",
        *,
        fmt: str = "png",
    ) -> str:
        """Create a line chart from ``(datetime, value)`` pairs.

        Args:
            data: Time-series data points.
            title: Chart title.
            ylabel: Y-axis label.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required")

        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        if not data:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=11)
            ax.axis("off")
        else:
            dates = [d[0] for d in data]
            values = [d[1] for d in data]
            ax.plot(dates, values, marker="o", color="#4361ee", linewidth=2, markersize=4)
            ax.fill_between(dates, values, alpha=0.1, color="#4361ee")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

        fig.tight_layout()
        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    def generate_multi_line(
        self,
        series: Dict[str, List[Tuple[datetime, float]]],
        title: str = "Trends",
        ylabel: str = "Value",
        *,
        fmt: str = "png",
    ) -> str:
        """Create a multi-line chart from several named time-series.

        Args:
            series: Mapping ``{name: [(datetime, value), ...]}``.
            title: Chart title.
            ylabel: Y-axis label.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required")

        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        if not series:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=11)
            ax.axis("off")
        else:
            for name, points in series.items():
                if not points:
                    continue
                dates = [p[0] for p in points]
                values = [p[1] for p in points]
                ax.plot(dates, values, marker="o", linewidth=2, markersize=3, label=name)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()

        fig.tight_layout()
        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    @staticmethod
    def _fig_to_base64(fig: object, fmt: str) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")  # type: ignore[union-attr]
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
