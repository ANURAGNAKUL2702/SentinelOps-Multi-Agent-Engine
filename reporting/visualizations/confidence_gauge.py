"""Confidence gauge chart — speedometer-style gauge for confidence scores."""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

from ..config import ReportingConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


class ConfidenceGauge:
    """Generate gauge charts for confidence scores.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    def generate(
        self,
        confidence: float,
        label: str = "Confidence",
        *,
        fmt: str = "png",
    ) -> str:
        """Create a gauge chart for *confidence* ∈ [0, 1].

        Args:
            confidence: Value between 0 and 1.
            label: Label text below the gauge.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required")

        confidence = max(0.0, min(1.0, confidence))

        fig, ax = plt.subplots(figsize=(3, 2), dpi=self.config.chart_dpi)

        # Draw arc background
        theta_start = 180
        theta_end = 0
        colours = ["#dc3545", "#fd7e14", "#ffc107", "#28a745"]
        for i, clr in enumerate(colours):
            a1 = theta_start - i * (180 / len(colours))
            a2 = a1 - (180 / len(colours))
            wedge = mpatches.Wedge(
                (0.5, 0),
                0.45,
                a2,
                a1,
                facecolor=clr,
                edgecolor="white",
                linewidth=1,
                transform=ax.transAxes,
            )
            ax.add_patch(wedge)

        # Needle
        angle_deg = 180 - confidence * 180
        angle_rad = math.radians(angle_deg)
        nx = 0.5 + 0.35 * math.cos(angle_rad)
        ny = 0.35 * math.sin(angle_rad)
        ax.annotate(
            "",
            xy=(nx, ny),
            xytext=(0.5, 0),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2d3436", lw=2),
        )

        ax.text(
            0.5,
            -0.15,
            f"{confidence * 100:.0f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.3,
            label,
            ha="center",
            fontsize=9,
            color="#636e72",
            transform=ax.transAxes,
        )
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, 0.6)
        ax.axis("off")
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
