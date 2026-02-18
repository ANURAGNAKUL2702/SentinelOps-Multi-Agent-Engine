"""Tests for GraphBuilder — Algorithms 1 (directed graph) and 2 (cycle detection)."""

import pytest

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.graph_builder import GraphBuilder
from agents.dependency_agent.schema import (
    DependencyAnalysisInput,
    ServiceEdge,
    ServiceGraph,
    ServiceNode,
)


@pytest.fixture
def builder() -> GraphBuilder:
    return GraphBuilder(DependencyAgentConfig())


def _make_input(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> DependencyAnalysisInput:
    """Helper to build input from simple node/edge specs."""
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(
            nodes=[ServiceNode(service_name=n) for n in nodes],
            edges=[
                ServiceEdge(source=s, target=t)
                for s, t in edges
            ],
        ),
    )


class TestGraphBuilderEmptyGraph:
    """Test Case 1: Empty graph."""

    def test_empty_graph_returns_zero_services(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input([], [])
        result = builder.build(input_data)
        assert result.total_services == 0
        assert result.total_dependencies == 0
        assert result.has_cycles is False
        assert result.max_depth == 0

    def test_empty_graph_has_empty_adjacency(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input([], [])
        result = builder.build(input_data)
        assert result.graph.adjacency_list == {}
        assert result.graph.reverse_adjacency == {}


class TestGraphBuilderLinearChain:
    """Test Case 2: Linear A → B → C."""

    def test_linear_chain_adjacency(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
        )
        result = builder.build(input_data)
        assert result.total_services == 3
        assert result.total_dependencies == 2
        assert result.has_cycles is False
        assert "B" in result.graph.adjacency_list["A"]
        assert "C" in result.graph.adjacency_list["B"]

    def test_linear_chain_reverse_adjacency(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
        )
        result = builder.build(input_data)
        assert "A" in result.graph.reverse_adjacency["B"]
        assert "B" in result.graph.reverse_adjacency["C"]

    def test_linear_chain_degrees(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
        )
        result = builder.build(input_data)
        assert result.graph.out_degree["A"] == 1
        assert result.graph.in_degree["B"] == 1
        assert result.graph.in_degree["C"] == 1

    def test_linear_chain_max_depth(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
        )
        result = builder.build(input_data)
        assert result.max_depth >= 2


class TestGraphBuilderCycleDetection:
    """Test Case 3: Cycle detection."""

    def test_simple_cycle_detected(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C"), ("C", "A")],
        )
        result = builder.build(input_data)
        assert result.has_cycles is True
        assert len(result.cycle_paths) > 0

    def test_no_cycle_in_dag(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B", "C", "D"],
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        result = builder.build(input_data)
        assert result.has_cycles is False

    def test_self_loop_cycle(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(
            ["A", "B"],
            [("A", "B"), ("B", "B")],
        )
        result = builder.build(input_data)
        assert result.has_cycles is True


class TestGraphBuilderNodeMap:
    """Test node_map population."""

    def test_node_map_populated(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(["X", "Y"], [("X", "Y")])
        result = builder.build(input_data)
        assert "X" in result.graph.node_map
        assert "Y" in result.graph.node_map

    def test_edge_map_populated(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(["X", "Y"], [("X", "Y")])
        result = builder.build(input_data)
        assert "X" in result.graph.edge_map
        assert len(result.graph.edge_map["X"]) == 1

    def test_build_latency_recorded(
        self, builder: GraphBuilder
    ) -> None:
        input_data = _make_input(["A"], [])
        result = builder.build(input_data)
        assert result.build_latency_ms >= 0.0
