"""Timeline chart — visualise incident events on a time axis."""

from __future__ import annotations

import base64
import io
from typing import List, Optional

from ..config import ReportingConfig
from ..schema import TimelineEvent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# Colour map for severity levels
_SEVERITY_COLORS = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "medium": "#ffc107",
    "low": "#28a745",
    "info": "#17a2b8",
}


class TimelineChart:
    """Generate timeline charts for incident events.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    def generate(
        self,
        events: List[TimelineEvent],
        *,
        fmt: str = "png",
    ) -> str:
        """Create a timeline chart and return it as a base64-encoded string.

        Args:
            events: Chronologically ordered timeline events.
            fmt: Image format (``"png"`` or ``"svg"``).

        Returns:
            Base64-encoded image string.

        Raises:
            ImportError: If matplotlib is unavailable.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for timeline charts")

        if not events:
            return self._empty_chart(fmt)

        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        timestamps = [ev.timestamp for ev in events]
        labels = [ev.event[:40] for ev in events]
        colours = [
            _SEVERITY_COLORS.get(ev.severity, "#17a2b8") for ev in events
        ]

        y_positions = list(range(len(events)))

        ax.barh(
            y_positions,
            [0.5] * len(events),
            left=mdates.date2num(timestamps),
            color=colours,
            height=0.6,
            edgecolor="white",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_xlabel("Time")
        ax.set_title("Incident Timeline", fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()

        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    # ------------------------------------------------------------------

    def _empty_chart(self, fmt: str) -> str:
        """Produce a placeholder chart when no events are available."""
        fig, ax = plt.subplots(figsize=(4, 2), dpi=72)
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    @staticmethod
    def _fig_to_base64(fig: object, fmt: str) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")  # type: ignore[union-attr]
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
