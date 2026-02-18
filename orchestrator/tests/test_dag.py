"""Tests for orchestrator.dag — DAG construction, topological sort, cycle detection."""

from __future__ import annotations

import pytest

from orchestrator.dag import ExecutionDAG
from orchestrator.schema import CycleDetectedError


class TestDAGConstruction:
    def test_add_node(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("agent_a")
        assert "agent_a" in dag.nodes

    def test_add_duplicate_node_raises(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("agent_a")
        with pytest.raises(ValueError, match="Duplicate"):
            dag.add_node("agent_a")

    def test_add_edge(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        assert ("a", "b") in dag.edges

    def test_add_edge_missing_source_raises(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("b")
        with pytest.raises(ValueError, match="Unknown source"):
            dag.add_edge("a", "b")

    def test_add_edge_missing_target_raises(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        with pytest.raises(ValueError, match="Unknown target"):
            dag.add_edge("a", "b")


class TestTopologicalSort:
    def test_empty_dag(self) -> None:
        dag = ExecutionDAG()
        assert dag.topological_sort() == []

    def test_single_node(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        stages = dag.topological_sort()
        assert stages == [["a"]]

    def test_linear_dag(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        stages = dag.topological_sort()
        assert len(stages) == 3
        assert stages[0] == ["a"]
        assert stages[1] == ["b"]
        assert stages[2] == ["c"]

    def test_parallel_stage(self) -> None:
        """Three nodes with no dependencies → single parallel stage."""
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        stages = dag.topological_sort()
        assert len(stages) == 1
        assert set(stages[0]) == {"a", "b", "c"}

    def test_pipeline_dag(self) -> None:
        """Mimics the real pipeline: 3 parallel → hypothesis → rc → val → ic."""
        dag = ExecutionDAG()
        for name in ["log", "metrics", "dep", "hyp", "rc", "val", "ic"]:
            dag.add_node(name)
        dag.add_edge("log", "hyp")
        dag.add_edge("metrics", "hyp")
        dag.add_edge("dep", "hyp")
        dag.add_edge("hyp", "rc")
        dag.add_edge("rc", "val")
        dag.add_edge("val", "ic")
        stages = dag.topological_sort()
        assert len(stages) == 5
        assert set(stages[0]) == {"log", "metrics", "dep"}
        assert stages[1] == ["hyp"]
        assert stages[2] == ["rc"]
        assert stages[3] == ["val"]
        assert stages[4] == ["ic"]

    def test_complex_dag_10_nodes(self) -> None:
        dag = ExecutionDAG()
        for i in range(10):
            dag.add_node(f"n{i}")
        dag.add_edge("n0", "n3")
        dag.add_edge("n1", "n3")
        dag.add_edge("n2", "n3")
        dag.add_edge("n3", "n4")
        dag.add_edge("n4", "n5")
        dag.add_edge("n5", "n6")
        dag.add_edge("n6", "n7")
        dag.add_edge("n7", "n8")
        dag.add_edge("n8", "n9")
        stages = dag.topological_sort()
        assert len(stages) == 8
        assert set(stages[0]) == {"n0", "n1", "n2"}


class TestCycleDetection:
    def test_no_cycle(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        assert dag.detect_cycles() is None

    def test_simple_cycle(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        cycle = dag.detect_cycles()
        assert cycle is not None
        assert len(cycle) >= 3

    def test_self_loop(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_edge("a", "a")
        cycle = dag.detect_cycles()
        assert cycle is not None
        assert "a" in cycle

    def test_cycle_blocks_topological_sort(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(CycleDetectedError):
            dag.topological_sort()


class TestGetDependencies:
    def test_get_dependencies(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")
        deps = dag.get_dependencies("c")
        assert set(deps) == {"a", "b"}

    def test_no_dependencies(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        assert dag.get_dependencies("a") == []


class TestValidate:
    def test_validate_ok(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.validate()  # should not raise

    def test_validate_cycle_raises(self) -> None:
        dag = ExecutionDAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(CycleDetectedError):
            dag.validate()
