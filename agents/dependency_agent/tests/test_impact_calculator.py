"""Tests for ImpactCalculator — Algorithms 3 (blast radius) and 5 (criticality)."""

import pytest

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.graph_builder import GraphBuilder
from agents.dependency_agent.core.impact_calculator import (
    ImpactCalculator,
)
from agents.dependency_agent.schema import (
    CriticalPathResult,
    DependencyAnalysisInput,
    ServiceEdge,
    ServiceGraph,
    ServiceNode,
)


@pytest.fixture
def config() -> DependencyAgentConfig:
    return DependencyAgentConfig()


@pytest.fixture
def calculator(config: DependencyAgentConfig) -> ImpactCalculator:
    return ImpactCalculator(config)


@pytest.fixture
def builder(config: DependencyAgentConfig) -> GraphBuilder:
    return GraphBuilder(config)


def _make_input(
    nodes: list[dict],
    edges: list[tuple[str, str]],
) -> DependencyAnalysisInput:
    """Helper to build input."""
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(
            nodes=[ServiceNode(**n) for n in nodes],
            edges=[
                ServiceEdge(source=s, target=t)
                for s, t in edges
            ],
        ),
    )


class TestBlastRadius:
    """Test Algorithm 3: BFS blast radius."""

    def test_direct_blast_radius(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """A → B, A → C: A's blast radius = B, C."""
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B", "C"]],
            [("A", "B"), ("A", "C")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        assert "B" in result["A"].blast_radius.directly_affected_services
        assert "C" in result["A"].blast_radius.directly_affected_services
        assert result["A"].blast_radius.total_affected_count == 2

    def test_transitive_blast_radius(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """A → B → C: A's total blast = B (direct) + C (indirect)."""
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B", "C"]],
            [("A", "B"), ("B", "C")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        assert result["A"].blast_radius.total_affected_count == 2
        assert "B" in result["A"].blast_radius.directly_affected_services
        assert "C" in result["A"].blast_radius.indirectly_affected_services

    def test_leaf_node_no_blast(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """C is a leaf: blast radius should be 0."""
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B", "C"]],
            [("A", "B"), ("B", "C")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        assert result["C"].blast_radius.total_affected_count == 0


class TestUpstreamDownstream:
    """Test upstream/downstream dependency tracking."""

    def test_upstream_deps(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """B depends on A: B's upstream includes A."""
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B"]],
            [("A", "B")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        assert "A" in result["B"].upstream_dependencies.services_depending_on_failed

    def test_downstream_deps(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """A calls B: A's downstream includes B."""
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B"]],
            [("A", "B")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        assert "B" in result["A"].downstream_dependencies.services_failed_depends_on


class TestCriticalityScore:
    """Test Algorithm 5: Criticality score."""

    def test_criticality_score_range(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B", "C"]],
            [("A", "B"), ("B", "C")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )

        for svc, impact in result.items():
            assert 0.0 <= impact.criticality_score <= 1.0

    def test_critical_path_boosts_score(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        input_data = _make_input(
            [{"service_name": n} for n in ["A", "B", "C"]],
            [("A", "B"), ("B", "C")],
        )
        graph_result = builder.build(input_data)

        # Without critical path
        result_no_cp = calculator.calculate(
            input_data, graph_result.graph
        )
        # With critical path including B
        cp = CriticalPathResult(
            path=["A", "B", "C"],
            total_duration_ms=100.0,
        )
        result_cp = calculator.calculate(
            input_data, graph_result.graph, critical_path=cp
        )

        assert result_cp["B"].criticality_score >= result_no_cp["B"].criticality_score


class TestSPOFDetection:
    """Test SPOF detection."""

    def test_single_instance_with_many_dependents_is_spof(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        """DB with 1 instance and >=3 dependents should be SPOF."""
        input_data = _make_input(
            [
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
                {"service_name": "DB", "instance_count": 1},
            ],
            [("A", "DB"), ("B", "DB"), ("C", "DB")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )
        assert result["DB"].is_single_point_of_failure is True

    def test_multi_instance_not_spof(
        self,
        calculator: ImpactCalculator,
        builder: GraphBuilder,
    ) -> None:
        input_data = _make_input(
            [
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
                {"service_name": "DB", "instance_count": 3},
            ],
            [("A", "DB"), ("B", "DB"), ("C", "DB")],
        )
        graph_result = builder.build(input_data)
        result = calculator.calculate(
            input_data, graph_result.graph
        )
        assert result["DB"].is_single_point_of_failure is False
