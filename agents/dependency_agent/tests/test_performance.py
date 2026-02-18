"""Performance benchmark — 1000 services must complete <200ms deterministic."""

import time

import pytest

from agents.dependency_agent.agent import DependencyAgent
from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.core.graph_builder import GraphBuilder
from agents.dependency_agent.schema import (
    DependencyAnalysisInput,
    ServiceEdge,
    ServiceGraph,
    ServiceNode,
)


def _generate_large_graph(
    n_services: int = 1000,
    edges_per_node: int = 3,
) -> DependencyAnalysisInput:
    """Generate a graph with *n_services* nodes and edges_per_node outgoing edges each."""
    nodes = [
        ServiceNode(service_name=f"svc-{i}")
        for i in range(n_services)
    ]
    edges = []
    for i in range(n_services):
        for j in range(1, edges_per_node + 1):
            target = (i + j) % n_services
            if target != i:
                edges.append(
                    ServiceEdge(
                        source=f"svc-{i}",
                        target=f"svc-{target}",
                    )
                )
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(
            nodes=nodes, edges=edges
        ),
    )


@pytest.fixture
def config() -> DependencyAgentConfig:
    return DependencyAgentConfig()


@pytest.fixture
def agent(config: DependencyAgentConfig) -> DependencyAgent:
    return DependencyAgent(config)


class TestGraphBuildPerformance:
    """Graph builder must handle 1000 services quickly."""

    def test_build_1000_services_under_100ms(
        self, config: DependencyAgentConfig
    ) -> None:
        builder = GraphBuilder(config)
        input_data = _generate_large_graph(1000)

        start = time.perf_counter()
        result = builder.build(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.total_services == 1000
        assert elapsed_ms < 100, (
            f"Graph build took {elapsed_ms:.1f}ms (limit 100ms)"
        )

    def test_build_500_services_returns_adjacency(
        self, config: DependencyAgentConfig
    ) -> None:
        builder = GraphBuilder(config)
        input_data = _generate_large_graph(500, 2)

        result = builder.build(input_data)
        assert len(result.graph.adjacency_list) > 0
        assert result.total_services == 500


class TestPipelinePerformance:
    """Full pipeline (deterministic path) <200ms for 1000 services."""

    def test_pipeline_1000_services_under_200ms(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _generate_large_graph(1000)

        start = time.perf_counter()
        output = agent.analyze(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert output.agent == "dependency_agent"
        assert elapsed_ms < 200, (
            f"Pipeline took {elapsed_ms:.1f}ms (limit 200ms)"
        )

    def test_pipeline_100_services_under_50ms(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _generate_large_graph(100, 2)

        start = time.perf_counter()
        output = agent.analyze(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert output.dependency_analysis.total_services == 100
        assert elapsed_ms < 50, (
            f"Pipeline took {elapsed_ms:.1f}ms (limit 50ms)"
        )

    def test_pipeline_scales_linearly(
        self, agent: DependencyAgent
    ) -> None:
        """100 → 500 services should scale roughly linearly."""
        input_100 = _generate_large_graph(100, 2)
        input_500 = _generate_large_graph(500, 2)

        start = time.perf_counter()
        agent.analyze(input_100)
        time_100 = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        agent.analyze(input_500)
        time_500 = (time.perf_counter() - start) * 1000

        # 500/100 = 5x, but we allow up to 10x for overhead
        ratio = time_500 / max(time_100, 0.001)
        assert ratio < 15, (
            f"Scaling ratio {ratio:.1f}x exceeds 15x "
            f"(100: {time_100:.1f}ms, 500: {time_500:.1f}ms)"
        )
