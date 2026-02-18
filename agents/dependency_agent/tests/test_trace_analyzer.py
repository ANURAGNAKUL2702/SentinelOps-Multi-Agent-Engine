"""Tests for TraceAnalyzer — Algorithm 4 (critical path analysis)."""

import pytest

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.trace_analyzer import TraceAnalyzer
from agents.dependency_agent.schema import (
    DependencyAnalysisInput,
    DistributedTrace,
    ServiceGraph,
    TraceSpan,
)


@pytest.fixture
def analyzer() -> TraceAnalyzer:
    return TraceAnalyzer(DependencyAgentConfig())


def _make_input_with_trace(
    spans: list[TraceSpan],
) -> DependencyAnalysisInput:
    """Helper to build input with a single trace."""
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(),
        traces=[
            DistributedTrace(
                trace_id="test-trace",
                root_service=spans[0].service_name if spans else "",
                total_duration_ms=sum(s.duration_ms for s in spans),
                spans=spans,
            )
        ],
    )


class TestTraceAnalyzerEmpty:
    """Test with no traces."""

    def test_no_traces_returns_empty(
        self, analyzer: TraceAnalyzer
    ) -> None:
        input_data = DependencyAnalysisInput(
            service_graph=ServiceGraph()
        )
        result = analyzer.analyze(input_data)
        assert result.critical_path is None
        assert result.slow_spans == []

    def test_empty_spans_returns_no_critical_path(
        self, analyzer: TraceAnalyzer
    ) -> None:
        input_data = DependencyAnalysisInput(
            service_graph=ServiceGraph(),
            traces=[DistributedTrace(spans=[])],
        )
        result = analyzer.analyze(input_data)
        assert result.critical_path is None


class TestTraceAnalyzerCriticalPath:
    """Test Case 5: Critical path analysis."""

    def test_linear_critical_path(
        self, analyzer: TraceAnalyzer
    ) -> None:
        """A → B → C chain, critical path should be all three."""
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                operation="op1", duration_ms=100.0,
            ),
            TraceSpan(
                span_id="s2", service_name="B",
                operation="op2", duration_ms=200.0,
                parent_span_id="s1",
            ),
            TraceSpan(
                span_id="s3", service_name="C",
                operation="op3", duration_ms=50.0,
                parent_span_id="s2",
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)

        assert result.critical_path is not None
        assert result.critical_path.path == ["A", "B", "C"]
        assert result.critical_path.total_duration_ms == 350.0
        assert result.critical_path.bottleneck_service == "B"

    def test_branching_takes_longer_path(
        self, analyzer: TraceAnalyzer
    ) -> None:
        """A → B (200ms) and A → C (50ms), should pick A → B."""
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                operation="root", duration_ms=10.0,
            ),
            TraceSpan(
                span_id="s2", service_name="B",
                operation="slow", duration_ms=200.0,
                parent_span_id="s1",
            ),
            TraceSpan(
                span_id="s3", service_name="C",
                operation="fast", duration_ms=50.0,
                parent_span_id="s1",
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)

        assert result.critical_path is not None
        assert "B" in result.critical_path.path
        assert "C" not in result.critical_path.path

    def test_bottleneck_percentage(
        self, analyzer: TraceAnalyzer
    ) -> None:
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=10.0,
            ),
            TraceSpan(
                span_id="s2", service_name="B",
                duration_ms=90.0, parent_span_id="s1",
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)

        assert result.critical_path is not None
        assert result.critical_path.bottleneck_percentage > 0


class TestTraceAnalyzerSlowSpans:
    """Test slow span detection."""

    def test_slow_spans_detected(
        self, analyzer: TraceAnalyzer
    ) -> None:
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=500.0,
            ),
            TraceSpan(
                span_id="s2", service_name="B",
                duration_ms=1500.0, parent_span_id="s1",
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)

        assert len(result.slow_spans) == 1
        assert result.slow_spans[0].service_name == "B"

    def test_no_slow_spans_under_threshold(
        self, analyzer: TraceAnalyzer
    ) -> None:
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=100.0,
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)
        assert result.slow_spans == []


class TestTraceAnalyzerLatencyContributions:
    """Test latency contribution calculation."""

    def test_contributions_sum_correctly(
        self, analyzer: TraceAnalyzer
    ) -> None:
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=100.0,
            ),
            TraceSpan(
                span_id="s2", service_name="A",
                duration_ms=50.0, parent_span_id="s1",
            ),
            TraceSpan(
                span_id="s3", service_name="B",
                duration_ms=200.0, parent_span_id="s1",
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)

        assert result.service_latency_contributions["A"] == 150.0
        assert result.service_latency_contributions["B"] == 200.0

    def test_analysis_latency_recorded(
        self, analyzer: TraceAnalyzer
    ) -> None:
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=10.0,
            ),
        ]
        input_data = _make_input_with_trace(spans)
        result = analyzer.analyze(input_data)
        assert result.analysis_latency_ms >= 0.0
