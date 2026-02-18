"""
File: validator.py
Purpose: 25-check validation for DependencyAgentOutput.
Dependencies: Schema models only
Performance: <5ms, O(n) where n=fields

Validation categories:
  Structural   (4 checks): Required fields exist
  Type         (6 checks): Correct data types
  Enum         (3 checks): Valid enum values
  Range        (4 checks): Numeric bounds
  Business     (5 checks): Semantic correctness
  Graph        (3 checks): Graph integrity
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    BottleneckType,
    CascadePattern,
    DependencyAgentOutput,
    DependencyAnalysisInput,
    Severity,
    ValidationResult,
    ValidationSeverity,
    ValidatorError,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.validator")


class OutputValidator:
    """Validates DependencyAgentOutput against 25 checks.

    Validation failures are informational — they do NOT block
    output generation. Errors are classified as CRITICAL or
    WARNING severity.

    Args:
        config: Agent configuration.

    Example::

        validator = OutputValidator(DependencyAgentConfig())
        result = validator.validate(output, input_data)
        print(result.validation_passed)
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def validate(
        self,
        output: DependencyAgentOutput,
        input_data: DependencyAnalysisInput,
        correlation_id: str = "",
    ) -> ValidationResult:
        """Run all 25 validation checks.

        Args:
            output: The assembled output to validate.
            input_data: Original input for cross-reference.
            correlation_id: Request correlation ID.

        Returns:
            ValidationResult with errors and warnings.
        """
        start = time.perf_counter()
        errors: List[ValidatorError] = []
        warnings: List[ValidatorError] = []
        checks = 0

        # ── Structural Checks (1–4) ────────────────────────────

        # 1. agent field is "dependency_agent"
        checks += 1
        if output.agent != "dependency_agent":
            errors.append(ValidatorError(
                check_number=1,
                check_name="agent_name",
                error_description="agent must be 'dependency_agent'",
                expected="dependency_agent",
                actual=output.agent,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 2. analysis_timestamp is not empty
        checks += 1
        if not output.analysis_timestamp:
            errors.append(ValidatorError(
                check_number=2,
                check_name="analysis_timestamp_present",
                error_description="analysis_timestamp is empty",
                expected="ISO 8601 timestamp",
                actual="",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 3. dependency_analysis is present
        checks += 1
        if output.dependency_analysis is None:
            errors.append(ValidatorError(
                check_number=3,
                check_name="dependency_analysis_present",
                error_description="dependency_analysis is None",
                expected="DependencyAnalysisSummary",
                actual="None",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 4. classification_source is valid
        checks += 1
        valid_sources = {
            "deterministic", "llm", "fallback", "cached"
        }
        if output.classification_source not in valid_sources:
            errors.append(ValidatorError(
                check_number=4,
                check_name="classification_source_valid",
                error_description=(
                    f"Invalid classification_source: "
                    f"'{output.classification_source}'"
                ),
                expected=str(valid_sources),
                actual=output.classification_source,
                severity=ValidationSeverity.CRITICAL,
            ))

        # ── Type Checks (5–10) ─────────────────────────────────

        # 5. confidence_score is float
        checks += 1
        if not isinstance(output.confidence_score, (int, float)):
            errors.append(ValidatorError(
                check_number=5,
                check_name="confidence_score_type",
                error_description="confidence_score must be numeric",
                expected="float",
                actual=type(output.confidence_score).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 6. pipeline_latency_ms is float
        checks += 1
        if not isinstance(output.pipeline_latency_ms, (int, float)):
            errors.append(ValidatorError(
                check_number=6,
                check_name="pipeline_latency_type",
                error_description="pipeline_latency_ms must be numeric",
                expected="float",
                actual=type(output.pipeline_latency_ms).__name__,
                severity=ValidationSeverity.WARNING,
            ))

        # 7. total_services is int
        checks += 1
        if output.dependency_analysis is not None:
            ts = output.dependency_analysis.total_services
            if not isinstance(ts, int):
                errors.append(ValidatorError(
                    check_number=7,
                    check_name="total_services_type",
                    error_description="total_services must be int",
                    expected="int",
                    actual=type(ts).__name__,
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 8. total_dependencies is int
        checks += 1
        if output.dependency_analysis is not None:
            td = output.dependency_analysis.total_dependencies
            if not isinstance(td, int):
                errors.append(ValidatorError(
                    check_number=8,
                    check_name="total_dependencies_type",
                    error_description="total_dependencies must be int",
                    expected="int",
                    actual=type(td).__name__,
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 9. bottlenecks is a list
        checks += 1
        if not isinstance(output.bottlenecks, list):
            errors.append(ValidatorError(
                check_number=9,
                check_name="bottlenecks_type",
                error_description="bottlenecks must be a list",
                expected="list",
                actual=type(output.bottlenecks).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 10. single_points_of_failure is a list
        checks += 1
        if not isinstance(output.single_points_of_failure, list):
            errors.append(ValidatorError(
                check_number=10,
                check_name="spof_type",
                error_description=(
                    "single_points_of_failure must be a list"
                ),
                expected="list",
                actual=type(
                    output.single_points_of_failure
                ).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # ── Enum Checks (11–13) ────────────────────────────────

        # 11. bottleneck_type values are valid
        checks += 1
        valid_bn_types = {bt.value for bt in BottleneckType}
        for bn in output.bottlenecks:
            if isinstance(bn.bottleneck_type, str):
                if bn.bottleneck_type not in valid_bn_types:
                    warnings.append(ValidatorError(
                        check_number=11,
                        check_name="bottleneck_type_enum",
                        error_description=(
                            f"Invalid bottleneck_type: "
                            f"'{bn.bottleneck_type}'"
                        ),
                        expected=str(valid_bn_types),
                        actual=str(bn.bottleneck_type),
                        severity=ValidationSeverity.WARNING,
                    ))

        # 12. cascade_pattern is valid enum
        checks += 1
        if output.cascading_failure_risk is not None:
            cp = output.cascading_failure_risk.cascade_pattern
            valid_cp = {p.value for p in CascadePattern}
            if isinstance(cp, str) and cp not in valid_cp:
                warnings.append(ValidatorError(
                    check_number=12,
                    check_name="cascade_pattern_enum",
                    error_description=(
                        f"Invalid cascade_pattern: '{cp}'"
                    ),
                    expected=str(valid_cp),
                    actual=str(cp),
                    severity=ValidationSeverity.WARNING,
                ))

        # 13. severity values in bottlenecks are valid
        checks += 1
        valid_sev = {s.value for s in Severity}
        for bn in output.bottlenecks:
            if isinstance(bn.severity, str):
                if bn.severity not in valid_sev:
                    warnings.append(ValidatorError(
                        check_number=13,
                        check_name="severity_enum",
                        error_description=(
                            f"Invalid severity: '{bn.severity}'"
                        ),
                        expected=str(valid_sev),
                        actual=str(bn.severity),
                        severity=ValidationSeverity.WARNING,
                    ))

        # ── Range Checks (14–17) ───────────────────────────────

        # 14. confidence_score in [0.0, 1.0]
        checks += 1
        if isinstance(output.confidence_score, (int, float)):
            if output.confidence_score < 0.0 or output.confidence_score > 1.0:
                errors.append(ValidatorError(
                    check_number=14,
                    check_name="confidence_score_range",
                    error_description=(
                        f"confidence_score out of range: "
                        f"{output.confidence_score}"
                    ),
                    expected="0.0 to 1.0",
                    actual=str(output.confidence_score),
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 15. pipeline_latency_ms >= 0
        checks += 1
        if isinstance(output.pipeline_latency_ms, (int, float)):
            if output.pipeline_latency_ms < 0:
                errors.append(ValidatorError(
                    check_number=15,
                    check_name="pipeline_latency_range",
                    error_description="pipeline_latency_ms is negative",
                    expected=">= 0.0",
                    actual=str(output.pipeline_latency_ms),
                    severity=ValidationSeverity.WARNING,
                ))

        # 16. total_services >= 0
        checks += 1
        if output.dependency_analysis is not None:
            if output.dependency_analysis.total_services < 0:
                errors.append(ValidatorError(
                    check_number=16,
                    check_name="total_services_range",
                    error_description="total_services is negative",
                    expected=">= 0",
                    actual=str(
                        output.dependency_analysis.total_services
                    ),
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 17. total_dependencies >= 0
        checks += 1
        if output.dependency_analysis is not None:
            if output.dependency_analysis.total_dependencies < 0:
                errors.append(ValidatorError(
                    check_number=17,
                    check_name="total_dependencies_range",
                    error_description="total_dependencies is negative",
                    expected=">= 0",
                    actual=str(
                        output.dependency_analysis.total_dependencies
                    ),
                    severity=ValidationSeverity.CRITICAL,
                ))

        # ── Business Rule Checks (18–22) ───────────────────────

        # 18. total_services matches input node count
        checks += 1
        if output.dependency_analysis is not None:
            expected_svc = len(input_data.service_graph.nodes)
            actual_svc = output.dependency_analysis.total_services
            if actual_svc != expected_svc:
                warnings.append(ValidatorError(
                    check_number=18,
                    check_name="services_count_match",
                    error_description=(
                        "total_services doesn't match input"
                    ),
                    expected=str(expected_svc),
                    actual=str(actual_svc),
                    severity=ValidationSeverity.WARNING,
                ))

        # 19. total_dependencies matches input edge count
        checks += 1
        if output.dependency_analysis is not None:
            expected_dep = len(input_data.service_graph.edges)
            actual_dep = output.dependency_analysis.total_dependencies
            if actual_dep != expected_dep:
                warnings.append(ValidatorError(
                    check_number=19,
                    check_name="dependencies_count_match",
                    error_description=(
                        "total_dependencies doesn't match input"
                    ),
                    expected=str(expected_dep),
                    actual=str(actual_dep),
                    severity=ValidationSeverity.WARNING,
                ))

        # 20. failed_service references a real service if present
        checks += 1
        if output.failed_service is not None:
            node_names = {
                n.service_name
                for n in input_data.service_graph.nodes
            }
            if output.failed_service.service_name not in node_names:
                warnings.append(ValidatorError(
                    check_number=20,
                    check_name="failed_service_exists",
                    error_description=(
                        f"Failed service "
                        f"'{output.failed_service.service_name}' "
                        f"not in input nodes"
                    ),
                    expected="service in input nodes",
                    actual=output.failed_service.service_name,
                    severity=ValidationSeverity.WARNING,
                ))

        # 21. SPOF services reference real services
        checks += 1
        node_names = {
            n.service_name for n in input_data.service_graph.nodes
        }
        for spof in output.single_points_of_failure:
            if spof.service_name not in node_names:
                warnings.append(ValidatorError(
                    check_number=21,
                    check_name="spof_service_exists",
                    error_description=(
                        f"SPOF service '{spof.service_name}' "
                        f"not in input nodes"
                    ),
                    expected="service in input nodes",
                    actual=spof.service_name,
                    severity=ValidationSeverity.WARNING,
                ))

        # 22. bottleneck services reference real services
        checks += 1
        for bn in output.bottlenecks:
            if bn.service_name not in node_names:
                warnings.append(ValidatorError(
                    check_number=22,
                    check_name="bottleneck_service_exists",
                    error_description=(
                        f"Bottleneck service '{bn.service_name}' "
                        f"not in input nodes"
                    ),
                    expected="service in input nodes",
                    actual=bn.service_name,
                    severity=ValidationSeverity.WARNING,
                ))

        # ── Graph Integrity Checks (23–25) ─────────────────────

        # 23. critical_path services are in input nodes
        checks += 1
        if output.critical_path is not None:
            for svc in output.critical_path.path:
                if svc not in node_names:
                    warnings.append(ValidatorError(
                        check_number=23,
                        check_name="critical_path_service_exists",
                        error_description=(
                            f"Critical path service '{svc}' "
                            f"not in input nodes"
                        ),
                        expected="service in input nodes",
                        actual=svc,
                        severity=ValidationSeverity.WARNING,
                    ))

        # 24. critical_path bottleneck_percentage <= 100
        checks += 1
        if output.critical_path is not None:
            bp = output.critical_path.bottleneck_percentage
            if bp > 100.0:
                errors.append(ValidatorError(
                    check_number=24,
                    check_name="bottleneck_percentage_range",
                    error_description=(
                        f"bottleneck_percentage > 100: {bp}"
                    ),
                    expected="<= 100.0",
                    actual=str(bp),
                    severity=ValidationSeverity.WARNING,
                ))

        # 25. cascade affected_services are in input nodes
        checks += 1
        if output.cascading_failure_risk is not None:
            for svc in (
                output.cascading_failure_risk.affected_services
            ):
                if svc not in node_names:
                    warnings.append(ValidatorError(
                        check_number=25,
                        check_name="cascade_service_exists",
                        error_description=(
                            f"Cascade affected service '{svc}' "
                            f"not in input nodes"
                        ),
                        expected="service in input nodes",
                        actual=svc,
                        severity=ValidationSeverity.WARNING,
                    ))

        elapsed_ms = (time.perf_counter() - start) * 1000
        passed = len(errors) == 0

        logger.debug(
            f"Validation complete: {checks} checks, "
            f"{len(errors)} errors, {len(warnings)} warnings, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "validation",
                "context": {
                    "checks_executed": checks,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "passed": passed,
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return ValidationResult(
            validation_passed=passed,
            checks_executed=checks,
            errors=errors,
            warnings=warnings,
            validation_latency_ms=round(elapsed_ms, 2),
        )
