"""Tests for reporting.visualizations.timeline_chart."""

from __future__ import annotations

import base64
from datetime import datetime

import pytest

from reporting.schema import TimelineEvent
from reporting.visualizations.timeline_chart import TimelineChart


@pytest.fixture
def chart() -> TimelineChart:
    return TimelineChart()


class TestTimelineChart:
    def test_generate_with_events(self, chart: TimelineChart) -> None:
        events = [
            TimelineEvent(timestamp=datetime(2025, 1, 1, 10, 0), event="Alert fired", severity="critical", service="api"),
            TimelineEvent(timestamp=datetime(2025, 1, 1, 10, 5), event="Investigated", severity="medium", service="db"),
        ]
        result = chart.generate(events)
        assert isinstance(result, str)
        # Should be base64-encoded PNG
        assert len(result) > 100
        base64.b64decode(result)  # Should not raise

    def test_generate_empty_events(self, chart: TimelineChart) -> None:
        result = chart.generate([])
        assert isinstance(result, str)

    def test_generate_single_event(self, chart: TimelineChart) -> None:
        events = [TimelineEvent(timestamp=datetime(2025, 1, 1, 12, 0), event="Detected", severity="high")]
        result = chart.generate(events)
        assert isinstance(result, str)
