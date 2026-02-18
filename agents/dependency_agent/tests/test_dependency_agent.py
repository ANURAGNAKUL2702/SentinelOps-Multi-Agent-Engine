"""Integration tests for DependencyAgent — 6 test cases from spec.

Test Case 1: Empty graph
Test Case 2: Linear chain A→B→C
Test Case 3: Cycle detection A→B→C→A
Test Case 4: Fan-in bottleneck (covered in bottleneck tests too)
Test Case 5: Critical path with traces
Test Case 6: Full integration test
"""

import pytest

from agents.dependency_agent.agent import DependencyAgent
from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    CallType,
    CascadePattern,
    CurrentFailure,
    DependencyAnalysisInput,
    DistributedTrace,
    FailureType,
    HealthStatus,
    ServiceEdge,
    ServiceGraph,
    ServiceNode,
    TraceSpan,
)


@pytest.fixture
def config() -> DependencyAgentConfig:
    return DependencyAgentConfig()


@pytest.fixture
def agent(config: DependencyAgentConfig) -> DependencyAgent:
    return DependencyAgent(config)


# ── helpers ─────────────────────────────────────────────────────


def _make_input(
    nodes: list[dict] | None = None,
    edges: list[tuple[str, str]] | None = None,
    traces: list[DistributedTrace] | None = None,
    current_failure: CurrentFailure | None = None,
) -> DependencyAnalysisInput:
    node_list = [
        ServiceNode(**n) if isinstance(n, dict) else n
        for n in (nodes or [])
    ]
    edge_list = [
        ServiceEdge(source=s, target=t)
        for s, t in (edges or [])
    ]
    return DependencyAnalysisInput(
        service_graph=ServiceGraph(
            nodes=node_list, edges=edge_list
        ),
        traces=traces or [],
        current_failure=current_failure,
    )


# ── Test Case 1: Empty graph ───────────────────────────────────


class TestEmptyGraph:
    """TC1: Empty graph should return zero-state output."""

    def test_empty_graph_returns_output(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input()
        output = agent.analyze(input_data)

        assert output.agent == "dependency_agent"
        assert output.confidence_score >= 0.0
        assert output.confidence_score <= 1.0
        assert output.classification_source == "deterministic"
        assert output.pipeline_latency_ms >= 0.0

    def test_empty_graph_dependency_analysis(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input()
        output = agent.analyze(input_data)

        da = output.dependency_analysis
        assert da.total_services == 0
        assert da.total_dependencies == 0
        assert da.graph_has_cycles is False

    def test_empty_graph_no_bottlenecks(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input()
        output = agent.analyze(input_data)
        assert output.bottlenecks == []

    def test_empty_graph_no_spofs(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input()
        output = agent.analyze(input_data)
        assert output.single_points_of_failure == []


# ── Test Case 2: Linear chain A→B→C ────────────────────────────


class TestLinearChain:
    """TC2: A→B→C linear chain."""

    def test_linear_chain_services(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        output = agent.analyze(input_data)

        assert output.dependency_analysis.total_services == 3
        assert output.dependency_analysis.total_dependencies == 2
        assert output.dependency_analysis.graph_has_cycles is False
        assert output.dependency_analysis.max_dependency_depth == 3

    def test_linear_chain_no_cascading(
        self, agent: DependencyAgent
    ) -> None:
        """No current failure → no cascading."""
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        output = agent.analyze(input_data)
        assert output.cascading_failure_risk.is_cascading is False

    def test_linear_chain_confidence(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        output = agent.analyze(input_data)
        assert output.confidence_score >= 0.2  # at least base
        assert output.confidence_score <= 1.0


# ── Test Case 3: Cycle detection A→B→C→A ───────────────────────


class TestCycleDetection:
    """TC3: Graph with cycle A→B→C→A."""

    def test_cycle_detected(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        output = agent.analyze(input_data)
        assert output.dependency_analysis.graph_has_cycles is True

    def test_cycle_lowers_confidence(
        self, agent: DependencyAgent
    ) -> None:
        """Cycles should remove the +0.10 no-cycles bonus."""
        input_data_no_cycle = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
            ],
            edges=[("A", "B")],
        )
        input_data_cycle = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
            ],
            edges=[("A", "B"), ("B", "A")],
        )

        out_no_cycle = agent.analyze(input_data_no_cycle)
        out_cycle = agent.analyze(input_data_cycle)

        assert out_no_cycle.confidence_score > out_cycle.confidence_score


# ── Test Case 4: Fan-in bottleneck ──────────────────────────────


class TestFanInBottleneck:
    """TC4: DB receives calls from 4+ services."""

    def test_fan_in_bottleneck_detected(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
                {"service_name": "D"},
                {"service_name": "DB"},
            ],
            edges=[
                ("A", "DB"), ("B", "DB"),
                ("C", "DB"), ("D", "DB"),
            ],
        )
        output = agent.analyze(input_data)

        bn_names = [b.service_name for b in output.bottlenecks]
        assert "DB" in bn_names

    def test_fan_in_bottleneck_has_severity(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": n}
                for n in ["A", "B", "C", "D", "DB"]
            ],
            edges=[
                ("A", "DB"), ("B", "DB"),
                ("C", "DB"), ("D", "DB"),
            ],
        )
        output = agent.analyze(input_data)

        db_bottlenecks = [
            b for b in output.bottlenecks
            if b.service_name == "DB"
        ]
        assert len(db_bottlenecks) >= 1
        assert db_bottlenecks[0].severity is not None


# ── Test Case 5: Critical path with traces ──────────────────────


class TestCriticalPathTraces:
    """TC5: Distributed traces determine critical path."""

    def test_critical_path_extracted(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("A", "C")],
            traces=[
                DistributedTrace(
                    trace_id="t1",
                    spans=[
                        TraceSpan(
                            span_id="s1",
                            service_name="A",
                            duration_ms=10.0,
                        ),
                        TraceSpan(
                            span_id="s2",
                            service_name="B",
                            duration_ms=50.0,
                            parent_span_id="s1",
                        ),
                        TraceSpan(
                            span_id="s3",
                            service_name="C",
                            duration_ms=20.0,
                            parent_span_id="s1",
                        ),
                    ],
                )
            ],
        )
        output = agent.analyze(input_data)

        assert output.critical_path is not None
        assert output.critical_path.total_duration_ms > 0
        assert "A" in output.critical_path.path

    def test_bottleneck_service_on_critical_path(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
            ],
            edges=[("A", "B")],
            traces=[
                DistributedTrace(
                    trace_id="t1",
                    spans=[
                        TraceSpan(
                            span_id="s1",
                            service_name="A",
                            duration_ms=10.0,
                        ),
                        TraceSpan(
                            span_id="s2",
                            service_name="B",
                            duration_ms=90.0,
                            parent_span_id="s1",
                        ),
                    ],
                )
            ],
        )
        output = agent.analyze(input_data)
        cp = output.critical_path
        assert cp is not None
        assert cp.bottleneck_service == "B"
        assert cp.bottleneck_percentage > 80.0


# ── Test Case 6: Full integration test ──────────────────────────


class TestFullIntegration:
    """TC6: Full pipeline with failure, traces, cascading."""

    def test_full_pipeline(
        self, agent: DependencyAgent
    ) -> None:
        input_data = DependencyAnalysisInput(
            service_graph=ServiceGraph(
                nodes=[
                    ServiceNode(
                        service_name="gateway",
                        instance_count=2,
                    ),
                    ServiceNode(
                        service_name="auth",
                        instance_count=1,
                        health_status=HealthStatus.UNHEALTHY,
                    ),
                    ServiceNode(
                        service_name="orders",
                        instance_count=3,
                    ),
                    ServiceNode(
                        service_name="payments",
                        instance_count=2,
                        health_status=HealthStatus.DEGRADED,
                    ),
                    ServiceNode(
                        service_name="db",
                        instance_count=1,
                    ),
                ],
                edges=[
                    ServiceEdge(
                        source="gateway", target="auth"
                    ),
                    ServiceEdge(
                        source="gateway", target="orders"
                    ),
                    ServiceEdge(
                        source="orders", target="payments"
                    ),
                    ServiceEdge(
                        source="orders", target="db"
                    ),
                    ServiceEdge(
                        source="auth", target="db"
                    ),
                    ServiceEdge(
                        source="payments", target="db"
                    ),
                ],
            ),
            traces=[
                DistributedTrace(
                    trace_id="t-full",
                    spans=[
                        TraceSpan(
                            span_id="s1",
                            service_name="gateway",
                            duration_ms=5.0,
                        ),
                        TraceSpan(
                            span_id="s2",
                            service_name="auth",
                            duration_ms=200.0,
                            parent_span_id="s1",
                        ),
                        TraceSpan(
                            span_id="s3",
                            service_name="orders",
                            duration_ms=30.0,
                            parent_span_id="s1",
                        ),
                        TraceSpan(
                            span_id="s4",
                            service_name="payments",
                            duration_ms=50.0,
                            parent_span_id="s3",
                        ),
                        TraceSpan(
                            span_id="s5",
                            service_name="db",
                            duration_ms=15.0,
                            parent_span_id="s3",
                        ),
                    ],
                )
            ],
            current_failure=CurrentFailure(
                service_name="auth",
                failure_type=FailureType.TIMEOUT,
            ),
        )

        output = agent.analyze(input_data)

        # ── structural checks ───────────────────────────────
        assert output.agent == "dependency_agent"
        assert output.dependency_analysis.total_services == 5
        assert output.dependency_analysis.total_dependencies == 6
        assert output.dependency_analysis.graph_has_cycles is False

        # ── failed service ──────────────────────────────────
        assert output.failed_service is not None
        assert output.failed_service.service_name == "auth"

        # ── critical path ───────────────────────────────────
        assert output.critical_path is not None
        assert output.critical_path.total_duration_ms > 0

        # ── cascading failure ───────────────────────────────
        # auth is unhealthy, downstream from gateway
        # payments is degraded (downstream of orders)
        # db depends on auth — so cascade should be detected
        cfr = output.cascading_failure_risk
        assert cfr is not None
        # auth has downstream db that may be affected
        # and gateway → auth makes gateway upstream

        # ── confidence ──────────────────────────────────────
        assert output.confidence_score >= 0.35
        assert output.confidence_score <= 1.0
        assert output.confidence_reasoning != ""

        # ── metadata ────────────────────────────────────────
        assert output.metadata is not None
        assert output.metadata.total_time_ms >= 0.0
        assert output.metadata.used_llm is False
        assert output.metadata.used_fallback is True

        # ── validation ──────────────────────────────────────
        assert output.validation is not None
        assert output.validation.checks_executed >= 20

    def test_full_pipeline_latency(
        self, agent: DependencyAgent
    ) -> None:
        """Deterministic pipeline <200ms."""
        input_data = _make_input(
            nodes=[
                {"service_name": "A"},
                {"service_name": "B"},
                {"service_name": "C"},
            ],
            edges=[("A", "B"), ("B", "C")],
        )
        output = agent.analyze(input_data)
        assert output.pipeline_latency_ms < 200

    def test_health_check(
        self, agent: DependencyAgent
    ) -> None:
        health = agent.health_check()
        assert health["status"] == "healthy"
        assert health["agent"] == "dependency_agent"
        assert "components" in health

    def test_telemetry_counters(
        self, agent: DependencyAgent
    ) -> None:
        input_data = _make_input(
            nodes=[{"service_name": "X"}],
            edges=[],
        )
        agent.analyze(input_data)
        snap = agent.telemetry.snapshot()
        assert snap["counters"]["analyses_total"] >= 1
        assert snap["counters"]["analyses_succeeded"] >= 1
