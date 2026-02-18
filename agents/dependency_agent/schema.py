"""
File: schema.py
Purpose: Type-safe Pydantic v2 schemas for the dependency agent pipeline.
Dependencies: pydantic >=2.0
Performance: Schema validation <1ms per object

Defines input/output contracts for every layer — graph construction,
trace analysis, impact calculation, bottleneck detection, and validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class ServiceType(str, Enum):
    """Type of service in the dependency graph."""
    GATEWAY = "gateway"
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    EXTERNAL = "external"


class HealthStatus(str, Enum):
    """Health status of a service node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CallType(str, Enum):
    """Type of communication between services."""
    HTTP = "http"
    GRPC = "grpc"
    TCP = "tcp"
    AMQP = "amqp"
    KAFKA = "kafka"


class BottleneckType(str, Enum):
    """Type of bottleneck detected."""
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    SEQUENTIAL = "sequential"


class Severity(str, Enum):
    """Severity classification for bottlenecks and issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CascadePattern(str, Enum):
    """Pattern of cascading failure propagation."""
    DOWNSTREAM_PROPAGATION = "downstream_propagation"
    UPSTREAM_PROPAGATION = "upstream_propagation"
    BIDIRECTIONAL = "bidirectional"
    ISOLATED = "isolated"


class FailureType(str, Enum):
    """Type of service failure."""
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    TIMEOUT = "timeout"
    CONNECTION_REFUSED = "connection_refused"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ValidationSeverity(str, Enum):
    """Severity of a validation failure."""
    CRITICAL = "critical"
    WARNING = "warning"


# ═══════════════════════════════════════════════════════════════
#  INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ServiceNode(BaseModel):
    """A service node in the dependency graph.

    Example::

        ServiceNode(
            service_name="payment-service",
            service_type=ServiceType.SERVICE,
            instance_count=8,
            health_status=HealthStatus.DEGRADED,
        )
    """
    service_name: str = Field(..., min_length=1)
    service_type: ServiceType = ServiceType.SERVICE
    instance_count: int = Field(default=1, ge=1)
    health_status: HealthStatus = HealthStatus.HEALTHY


class ServiceEdge(BaseModel):
    """A directed dependency edge between two services.

    Example::

        ServiceEdge(
            source="api-gateway",
            target="payment-service",
            call_type=CallType.HTTP,
            request_rate_per_sec=500.0,
            error_rate_percent=8.5,
        )
    """
    source: str = Field(..., min_length=1, alias="from")
    target: str = Field(..., min_length=1, alias="to")
    call_type: CallType = CallType.HTTP
    request_rate_per_sec: float = Field(default=0.0, ge=0.0)
    error_rate_percent: float = Field(default=0.0, ge=0.0, le=100.0)

    model_config = {"populate_by_name": True}


class ServiceGraph(BaseModel):
    """Service dependency graph containing nodes and edges.

    Example::

        ServiceGraph(
            nodes=[ServiceNode(service_name="A")],
            edges=[ServiceEdge(source="A", target="B")],
        )
    """
    nodes: List[ServiceNode] = Field(default_factory=list)
    edges: List[ServiceEdge] = Field(default_factory=list)


class TraceSpan(BaseModel):
    """A single span in a distributed trace.

    Example::

        TraceSpan(
            span_id="span-1",
            service_name="payment-service",
            operation="processPayment",
            duration_ms=1200.0,
            parent_span_id="span-0",
            error=True,
            error_message="Database timeout",
        )
    """
    span_id: str = Field(..., min_length=1)
    service_name: str = Field(..., min_length=1)
    operation: str = ""
    duration_ms: float = Field(default=0.0, ge=0.0)
    parent_span_id: Optional[str] = None
    error: bool = False
    error_message: Optional[str] = None


class DistributedTrace(BaseModel):
    """A distributed trace composed of spans.

    Example::

        DistributedTrace(
            trace_id="abc-123",
            root_service="api-gateway",
            total_duration_ms=1250.0,
            spans=[TraceSpan(...)],
        )
    """
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    root_service: str = ""
    total_duration_ms: float = Field(default=0.0, ge=0.0)
    spans: List[TraceSpan] = Field(default_factory=list)


class CurrentFailure(BaseModel):
    """Description of the currently failing service.

    Example::

        CurrentFailure(
            service_name="payment-service",
            failure_type=FailureType.HIGH_ERROR_RATE,
            timestamp="2026-02-13T10:15:00Z",
        )
    """
    service_name: str = Field(..., min_length=1)
    failure_type: FailureType = FailureType.HIGH_ERROR_RATE
    timestamp: str = ""


class DependencyAnalysisInput(BaseModel):
    """Raw input to the dependency agent pipeline.

    Example::

        DependencyAnalysisInput(
            service_graph=ServiceGraph(...),
            traces=[DistributedTrace(...)],
            current_failure=CurrentFailure(service_name="payment-service"),
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
        )
    """
    service_graph: ServiceGraph = Field(
        default_factory=ServiceGraph,
    )
    traces: List[DistributedTrace] = Field(default_factory=list)
    current_failure: Optional[CurrentFailure] = None
    time_window: str = ""
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )


# ═══════════════════════════════════════════════════════════════
#  GRAPH BUILD SCHEMAS
# ═══════════════════════════════════════════════════════════════


class GraphData(BaseModel):
    """Internal representation of the directed dependency graph.

    Example::

        GraphData(
            adjacency_list={"A": ["B"], "B": ["C"]},
            reverse_adjacency={"B": ["A"], "C": ["B"]},
            in_degree={"A": 0, "B": 1, "C": 1},
            out_degree={"A": 1, "B": 1, "C": 0},
        )
    """
    adjacency_list: Dict[str, List[str]] = Field(default_factory=dict)
    reverse_adjacency: Dict[str, List[str]] = Field(default_factory=dict)
    in_degree: Dict[str, int] = Field(default_factory=dict)
    out_degree: Dict[str, int] = Field(default_factory=dict)
    node_map: Dict[str, ServiceNode] = Field(default_factory=dict)
    edge_map: Dict[str, List[ServiceEdge]] = Field(default_factory=dict)


class GraphBuildResult(BaseModel):
    """Output of graph construction phase."""
    graph: GraphData
    total_services: int = Field(ge=0, default=0)
    total_dependencies: int = Field(ge=0, default=0)
    has_cycles: bool = False
    cycle_paths: List[List[str]] = Field(default_factory=list)
    max_depth: int = Field(ge=0, default=0)
    build_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  TRACE ANALYSIS SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CriticalPathResult(BaseModel):
    """Critical path extracted from distributed traces.

    Example::

        CriticalPathResult(
            path=["api-gateway", "payment-service", "database-primary"],
            total_duration_ms=1250.0,
            bottleneck_service="payment-service",
            bottleneck_duration_ms=1200.0,
            bottleneck_percentage=96.0,
        )
    """
    path: List[str] = Field(default_factory=list)
    total_duration_ms: float = Field(ge=0.0, default=0.0)
    bottleneck_service: str = ""
    bottleneck_duration_ms: float = Field(ge=0.0, default=0.0)
    bottleneck_percentage: float = Field(ge=0.0, le=100.0, default=0.0)


class SlowSpan(BaseModel):
    """A span exceeding the slow threshold."""
    span_id: str
    service_name: str
    operation: str
    duration_ms: float
    is_error: bool = False


class TraceAnalysisResult(BaseModel):
    """Output of trace analysis phase."""
    critical_path: Optional[CriticalPathResult] = None
    slow_spans: List[SlowSpan] = Field(default_factory=list)
    service_latency_contributions: Dict[str, float] = Field(
        default_factory=dict
    )
    analysis_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  IMPACT ANALYSIS SCHEMAS
# ═══════════════════════════════════════════════════════════════


class BlastRadius(BaseModel):
    """Blast radius of a failed service."""
    directly_affected_services: List[str] = Field(default_factory=list)
    indirectly_affected_services: List[str] = Field(default_factory=list)
    total_affected_count: int = Field(ge=0, default=0)
    affected_request_rate_per_sec: float = Field(ge=0.0, default=0.0)


class UpstreamDependencies(BaseModel):
    """Services that depend on the failed service."""
    services_depending_on_failed: List[str] = Field(default_factory=list)
    count: int = Field(ge=0, default=0)


class DownstreamDependencies(BaseModel):
    """Services the failed service depends on."""
    services_failed_depends_on: List[str] = Field(default_factory=list)
    count: int = Field(ge=0, default=0)


class ImpactAnalysisResult(BaseModel):
    """Output of impact analysis phase."""
    blast_radius: BlastRadius = Field(default_factory=BlastRadius)
    upstream_dependencies: UpstreamDependencies = Field(
        default_factory=UpstreamDependencies
    )
    downstream_dependencies: DownstreamDependencies = Field(
        default_factory=DownstreamDependencies
    )
    criticality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    is_single_point_of_failure: bool = False
    impact_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  BOTTLENECK SCHEMAS
# ═══════════════════════════════════════════════════════════════


class Bottleneck(BaseModel):
    """A detected bottleneck in the service graph.

    Example::

        Bottleneck(
            service_name="database-primary",
            bottleneck_type=BottleneckType.FAN_IN,
            severity=Severity.HIGH,
            fan_in_count=5,
            reasoning="Database receives calls from 5 services",
        )
    """
    service_name: str
    bottleneck_type: BottleneckType
    severity: Severity = Severity.MEDIUM
    fan_in_count: int = Field(ge=0, default=0)
    fan_out_count: int = Field(ge=0, default=0)
    contributing_duration_ms: float = Field(ge=0.0, default=0.0)
    bottleneck_percentage: float = Field(ge=0.0, le=100.0, default=0.0)
    reasoning: str = ""


class BottleneckDetectionResult(BaseModel):
    """Output of the bottleneck detection phase."""
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    detection_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  CASCADING FAILURE SCHEMAS
# ═══════════════════════════════════════════════════════════════


class CascadingFailureRisk(BaseModel):
    """Analysis of cascading failure risk.

    Example::

        CascadingFailureRisk(
            is_cascading=True,
            cascade_pattern=CascadePattern.DOWNSTREAM_PROPAGATION,
            cascade_depth=2,
            reasoning="Failure propagates from database to payment to gateway",
        )
    """
    is_cascading: bool = False
    cascade_pattern: CascadePattern = CascadePattern.ISOLATED
    cascade_depth: int = Field(ge=0, default=0)
    affected_services: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ═══════════════════════════════════════════════════════════════
#  SPOF SCHEMAS
# ═══════════════════════════════════════════════════════════════


class SinglePointOfFailure(BaseModel):
    """A single point of failure in the architecture.

    Example::

        SinglePointOfFailure(
            service_name="database-primary",
            reason="Single instance, critical to 2 services",
            mitigation="Add replica or implement caching",
        )
    """
    service_name: str
    reason: str = ""
    mitigation: str = ""


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class FailedServiceInfo(BaseModel):
    """Summary of the failed service for output."""
    service_name: str
    failure_type: str = ""
    health_status: str = "unknown"


class DependencyAnalysisSummary(BaseModel):
    """Graph-level summary for the output."""
    total_services: int = Field(ge=0, default=0)
    total_dependencies: int = Field(ge=0, default=0)
    graph_has_cycles: bool = False
    max_dependency_depth: int = Field(ge=0, default=0)


class ClassificationResult(BaseModel):
    """Output of the classification layer (LLM or fallback).

    Combines all analysis results into a unified classification.
    """
    failed_service: Optional[FailedServiceInfo] = None
    dependency_analysis: DependencyAnalysisSummary = Field(
        default_factory=DependencyAnalysisSummary,
    )
    impact_analysis: Optional[ImpactAnalysisResult] = None
    critical_path: Optional[CriticalPathResult] = None
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    cascading_failure_risk: CascadingFailureRisk = Field(
        default_factory=CascadingFailureRisk,
    )
    single_points_of_failure: List[SinglePointOfFailure] = Field(
        default_factory=list,
    )
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.2)
    confidence_reasoning: str = ""
    classification_source: str = Field(
        default="deterministic",
        description="'deterministic' | 'llm' | 'fallback' | 'cached'",
    )
    classification_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  VALIDATION SCHEMAS
# ═══════════════════════════════════════════════════════════════


class ValidatorError(BaseModel):
    """A single validation failure."""
    check_number: int
    check_name: str
    error_description: str
    expected: str
    actual: str
    severity: ValidationSeverity = ValidationSeverity.WARNING


class ValidationResult(BaseModel):
    """Output of the validation layer."""
    validation_passed: bool
    checks_executed: int = Field(ge=0)
    errors: List[ValidatorError] = Field(default_factory=list)
    warnings: List[ValidatorError] = Field(default_factory=list)
    validation_latency_ms: float = Field(ge=0.0, default=0.0)


# ═══════════════════════════════════════════════════════════════
#  METADATA SCHEMA
# ═══════════════════════════════════════════════════════════════


class PipelineMetadata(BaseModel):
    """Pipeline execution metadata."""
    graph_build_time_ms: float = Field(ge=0.0, default=0.0)
    trace_analysis_time_ms: float = Field(ge=0.0, default=0.0)
    impact_calculation_time_ms: float = Field(ge=0.0, default=0.0)
    classification_time_ms: float = Field(ge=0.0, default=0.0)
    validation_time_ms: float = Field(ge=0.0, default=0.0)
    total_time_ms: float = Field(ge=0.0, default=0.0)
    used_llm: bool = False
    used_fallback: bool = False
    cache_hit: bool = False
    correlation_id: str = ""


# ═══════════════════════════════════════════════════════════════
#  FINAL OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════


class DependencyAgentOutput(BaseModel):
    """Final output of the dependency agent.

    Example::

        DependencyAgentOutput(
            agent="dependency_agent",
            time_window="...",
            failed_service=FailedServiceInfo(...),
            dependency_analysis=DependencyAnalysisSummary(...),
            impact_analysis=ImpactAnalysisResult(...),
            critical_path=CriticalPathResult(...),
            confidence_score=0.88,
        )
    """
    agent: str = Field(default="dependency_agent", frozen=True)
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    time_window: str = ""
    failed_service: Optional[FailedServiceInfo] = None
    dependency_analysis: DependencyAnalysisSummary = Field(
        default_factory=DependencyAnalysisSummary,
    )
    impact_analysis: Optional[ImpactAnalysisResult] = None
    critical_path: Optional[CriticalPathResult] = None
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    cascading_failure_risk: CascadingFailureRisk = Field(
        default_factory=CascadingFailureRisk,
    )
    single_points_of_failure: List[SinglePointOfFailure] = Field(
        default_factory=list,
    )
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.2)
    confidence_reasoning: str = ""
    correlation_id: str = ""
    classification_source: str = "deterministic"
    pipeline_latency_ms: float = Field(ge=0.0, default=0.0)
    metadata: Optional[PipelineMetadata] = None
    validation: Optional[ValidationResult] = None

    @field_validator("agent")
    @classmethod
    def agent_must_be_dependency_agent(cls, v: str) -> str:
        if v != "dependency_agent":
            raise ValueError(
                f"agent must be 'dependency_agent', got '{v}'"
            )
        return v
