"""Tests for reporting.visualizations.dependency_graph_viz."""

from __future__ import annotations

import base64

import pytest

from reporting.visualizations.dependency_graph_viz import DependencyGraphViz


@pytest.fixture
def viz() -> DependencyGraphViz:
    return DependencyGraphViz()


class TestDependencyGraphViz:
    def test_generate_with_graph(self, viz: DependencyGraphViz) -> None:
        nodes = ["api-gateway", "auth-service", "db", "cache"]
        edges = [("api-gateway", "auth-service"), ("api-gateway", "db"), ("auth-service", "cache")]
        affected = ["db"]
        result = viz.generate(nodes, edges, affected)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_generate_empty_graph(self, viz: DependencyGraphViz) -> None:
        result = viz.generate([], [], [])
        assert isinstance(result, str)

    def test_generate_single_node(self, viz: DependencyGraphViz) -> None:
        result = viz.generate(["service-a"], [], [])
        assert isinstance(result, str)
