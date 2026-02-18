"""Tests for BottleneckDetector — Algorithm 6 (fan-in, fan-out, sequential)."""

import pytest

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.bottleneck_detector import (
    BottleneckDetector,
)
from agents.dependency_agent.core.graph_builder import GraphBuilder
from agents.dependency_agent.schema import (
    BottleneckType,
    CriticalPathResult,
    DependencyAnalysisInput,
    ServiceEdge,
    ServiceGraph,
    ServiceNode,
    TraceSpan,
)


@pytest.fixture
def config() -> DependencyAgentConfig:
    return DependencyAgentConfig()


@pytest.fixture
def detector(config: DependencyAgentConfig) -> BottleneckDetector:
    return BottleneckDetector(config)


@pytest.fixture
def builder(config: DependencyAgentConfig) -> GraphBuilder:
    return GraphBuilder(config)


def _make_input(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> DependencyAnalysisInput:
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(
            nodes=[ServiceNode(service_name=n) for n in nodes],
            edges=[
                ServiceEdge(source=s, target=t)
                for s, t in edges
            ],
        ),
    )


class TestFanInBottleneck:
    """Test Case 4: Fan-in bottleneck detection."""

    def test_fan_in_detected(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        """DB receives calls from 4 services (threshold=3)."""
        input_data = _make_input(
            ["A", "B", "C", "D", "DB"],
            [
                ("A", "DB"), ("B", "DB"),
                ("C", "DB"), ("D", "DB"),
            ],
        )
        graph_result = builder.build(input_data)
        result = detector.detect(graph_result.graph)

        fan_in = [
            b for b in result.bottlenecks
            if b.bottleneck_type == BottleneckType.FAN_IN
        ]
        assert len(fan_in) >= 1
        assert fan_in[0].service_name == "DB"

    def test_below_threshold_not_detected(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        """DB receives from 2 services (below threshold=3)."""
        input_data = _make_input(
            ["A", "B", "DB"],
            [("A", "DB"), ("B", "DB")],
        )
        graph_result = builder.build(input_data)
        result = detector.detect(graph_result.graph)

        fan_in = [
            b for b in result.bottlenecks
            if b.bottleneck_type == BottleneckType.FAN_IN
        ]
        assert len(fan_in) == 0


class TestFanOutBottleneck:
    """Test fan-out bottleneck detection."""

    def test_fan_out_detected(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        """Gateway calls 6 services (threshold=5)."""
        input_data = _make_input(
            ["GW", "A", "B", "C", "D", "E", "F"],
            [
                ("GW", "A"), ("GW", "B"), ("GW", "C"),
                ("GW", "D"), ("GW", "E"), ("GW", "F"),
            ],
        )
        graph_result = builder.build(input_data)
        result = detector.detect(graph_result.graph)

        fan_out = [
            b for b in result.bottlenecks
            if b.bottleneck_type == BottleneckType.FAN_OUT
        ]
        assert len(fan_out) >= 1
        assert fan_out[0].service_name == "GW"


class TestSequentialBottleneck:
    """Test sequential bottleneck on critical path."""

    def test_sequential_bottleneck_detected(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        """Service contributing >50% of critical path."""
        input_data = _make_input(
            ["A", "B"],
            [("A", "B")],
        )
        graph_result = builder.build(input_data)
        cp = CriticalPathResult(
            path=["A", "B"],
            total_duration_ms=100.0,
        )
        spans = [
            TraceSpan(
                span_id="s1", service_name="A",
                duration_ms=20.0,
            ),
            TraceSpan(
                span_id="s2", service_name="B",
                duration_ms=80.0, parent_span_id="s1",
            ),
        ]
        result = detector.detect(
            graph_result.graph, critical_path=cp, spans=spans
        )

        sequential = [
            b for b in result.bottlenecks
            if b.bottleneck_type == BottleneckType.SEQUENTIAL
        ]
        assert len(sequential) >= 1
        assert sequential[0].service_name == "B"
        assert sequential[0].bottleneck_percentage > 50.0


class TestBottleneckDetectorLatency:
    """Test detection latency."""

    def test_detection_latency_recorded(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        input_data = _make_input(["A"], [])
        graph_result = builder.build(input_data)
        result = detector.detect(graph_result.graph)
        assert result.detection_latency_ms >= 0.0

    def test_empty_graph_no_bottlenecks(
        self,
        detector: BottleneckDetector,
        builder: GraphBuilder,
    ) -> None:
        input_data = _make_input([], [])
        graph_result = builder.build(input_data)
        result = detector.detect(graph_result.graph)
        assert result.bottlenecks == []
