"""
File: validator.py
Purpose: 23 validation checks on MetricsAgentOutput.
Dependencies: Schema models only
Performance: <1ms, O(n) complexity

Checks: structural (3), type (6), enum (3), range (5),
        business rules (4), consistency (2).
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AnomalyType,
    MetricsAgentOutput,
    MetricsAnalysisInput,
    Severity,
    TrendType,
    ValidatorError,
    ValidationResult,
    ValidationSeverity,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.validator")


class OutputValidator:
    """Validates MetricsAgentOutput against 23 production checks.

    Categories:
        1-3:   Structural checks
        4-9:   Type checks
        10-12: Enum checks
        13-17: Range checks
        18-21: Business rule checks
        22-23: Consistency checks

    Args:
        config: Agent configuration.

    Example::

        validator = OutputValidator(MetricsAgentConfig())
        result = validator.validate(output, input_data)
        assert result.validation_passed
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def validate(
        self,
        output: MetricsAgentOutput,
        input_data: MetricsAnalysisInput,
        correlation_id: str = "",
    ) -> ValidationResult:
        """Run all 23 validation checks on the output.

        Args:
            output: MetricsAgentOutput to validate.
            input_data: Original input for cross-reference.
            correlation_id: Request correlation ID.

        Returns:
            ValidationResult with errors and warnings.
        """
        start = time.perf_counter()
        errors: List[ValidatorError] = []
        warnings: List[ValidatorError] = []

        checks = [
            self._check_1_valid_structure,
            self._check_2_required_fields,
            self._check_3_agent_name,
            self._check_4_metric_name_type,
            self._check_5_current_value_type,
            self._check_6_zscore_type,
            self._check_7_confidence_type,
            self._check_8_correlation_coeff_type,
            self._check_9_boolean_types,
            self._check_10_anomaly_type_enum,
            self._check_11_severity_enum,
            self._check_12_trend_enum,
            self._check_13_confidence_range,
            self._check_14_correlation_range,
            self._check_15_percentage_range,
            self._check_16_non_negative_metrics,
            self._check_17_zscore_sanity,
            self._check_18_empty_anomalies_confidence,
            self._check_19_critical_zscore,
            self._check_20_high_severity_check,
            self._check_21_spike_growth_rate,
            self._check_22_metric_names_in_input,
            self._check_23_correlation_metrics_anomalous,
        ]

        for check_fn in checks:
            result = check_fn(output, input_data)
            if result is not None:
                if result.severity == ValidationSeverity.CRITICAL:
                    errors.append(result)
                else:
                    warnings.append(result)

        elapsed_ms = (time.perf_counter() - start) * 1000
        passed = len(errors) == 0

        logger.info(
            f"Validation completed: {'PASS' if passed else 'FAIL'}, "
            f"{len(checks)} checks, {len(errors)} errors, "
            f"{len(warnings)} warnings, {elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "validation",
                "context": {
                    "passed": passed,
                    "checks": len(checks),
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "latency_ms": round(elapsed_ms, 2),
                },
            },
        )

        return ValidationResult(
            validation_passed=passed,
            checks_executed=len(checks),
            errors=errors,
            warnings=warnings,
            validation_latency_ms=round(elapsed_ms, 2),
        )

    # ── Structural Checks (1-3) ─────────────────────────────────

    def _check_1_valid_structure(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 1: Output can be serialized to valid JSON."""
        try:
            output.model_dump_json()
            return None
        except Exception as e:
            return ValidatorError(
                check_number=1,
                check_name="valid_json_structure",
                error_description=f"Output is not valid JSON: {e}",
                expected="Valid JSON",
                actual=str(e),
                severity=ValidationSeverity.CRITICAL,
            )

    def _check_2_required_fields(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 2: All required top-level fields present."""
        required = ["agent", "anomalous_metrics", "confidence_score", "metadata"]
        dump = output.model_dump()
        missing = [f for f in required if f not in dump or dump[f] is None]
        if missing:
            return ValidatorError(
                check_number=2,
                check_name="required_fields_present",
                error_description=f"Missing required fields: {missing}",
                expected=str(required),
                actual=f"Missing: {missing}",
                severity=ValidationSeverity.CRITICAL,
            )
        return None

    def _check_3_agent_name(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 3: agent == 'metrics_agent'."""
        if output.agent != "metrics_agent":
            return ValidatorError(
                check_number=3,
                check_name="agent_name_correct",
                error_description=f"agent must be 'metrics_agent'",
                expected="metrics_agent",
                actual=output.agent,
                severity=ValidationSeverity.CRITICAL,
            )
        return None

    # ── Type Checks (4-9) ───────────────────────────────────────

    def _check_4_metric_name_type(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 4: metric_name is string."""
        for am in output.anomalous_metrics:
            if not isinstance(am.metric_name, str):
                return ValidatorError(
                    check_number=4,
                    check_name="metric_name_is_string",
                    error_description="metric_name must be string",
                    expected="str",
                    actual=type(am.metric_name).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_5_current_value_type(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 5: current_value is float or int."""
        for am in output.anomalous_metrics:
            if not isinstance(am.current_value, (int, float)):
                return ValidatorError(
                    check_number=5,
                    check_name="current_value_is_numeric",
                    error_description="current_value must be numeric",
                    expected="float or int",
                    actual=type(am.current_value).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_6_zscore_type(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 6: zscore is float."""
        for am in output.anomalous_metrics:
            if not isinstance(am.zscore, (int, float)):
                return ValidatorError(
                    check_number=6,
                    check_name="zscore_is_float",
                    error_description="zscore must be float",
                    expected="float",
                    actual=type(am.zscore).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_7_confidence_type(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 7: confidence_score is float."""
        if not isinstance(output.confidence_score, (int, float)):
            return ValidatorError(
                check_number=7,
                check_name="confidence_is_float",
                error_description="confidence_score must be float",
                expected="float",
                actual=type(output.confidence_score).__name__,
                severity=ValidationSeverity.CRITICAL,
            )
        return None

    def _check_8_correlation_coeff_type(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 8: correlation_coefficient is float."""
        for corr in output.correlations:
            if not isinstance(corr.correlation_coefficient, (int, float)):
                return ValidatorError(
                    check_number=8,
                    check_name="correlation_coeff_is_float",
                    error_description="correlation_coefficient must be float",
                    expected="float",
                    actual=type(corr.correlation_coefficient).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_9_boolean_types(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 9: Boolean fields are bool."""
        for am in output.anomalous_metrics:
            if not isinstance(am.is_anomalous, bool):
                return ValidatorError(
                    check_number=9,
                    check_name="boolean_types",
                    error_description="is_anomalous must be bool",
                    expected="bool",
                    actual=type(am.is_anomalous).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
            if not isinstance(am.threshold_breached, bool):
                return ValidatorError(
                    check_number=9,
                    check_name="boolean_types",
                    error_description="threshold_breached must be bool",
                    expected="bool",
                    actual=type(am.threshold_breached).__name__,
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    # ── Enum Checks (10-12) ─────────────────────────────────────

    def _check_10_anomaly_type_enum(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 10: anomaly_type in valid enum values."""
        valid = {e.value for e in AnomalyType}
        for am in output.anomalous_metrics:
            val = am.anomaly_type.value if isinstance(am.anomaly_type, AnomalyType) else am.anomaly_type
            if val not in valid:
                return ValidatorError(
                    check_number=10,
                    check_name="anomaly_type_valid",
                    error_description=f"Invalid anomaly_type: {val}",
                    expected=str(valid),
                    actual=str(val),
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_11_severity_enum(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 11: severity in valid enum values."""
        valid = {e.value for e in Severity}
        for am in output.anomalous_metrics:
            val = am.severity.value if isinstance(am.severity, Severity) else am.severity
            if val not in valid:
                return ValidatorError(
                    check_number=11,
                    check_name="severity_valid",
                    error_description=f"Invalid severity: {val}",
                    expected=str(valid),
                    actual=str(val),
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_12_trend_enum(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 12: trend in valid enum values."""
        valid = {e.value for e in TrendType}
        for am in output.anomalous_metrics:
            val = am.trend.value if isinstance(am.trend, TrendType) else am.trend
            if val not in valid:
                return ValidatorError(
                    check_number=12,
                    check_name="trend_valid",
                    error_description=f"Invalid trend: {val}",
                    expected=str(valid),
                    actual=str(val),
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    # ── Range Checks (13-17) ────────────────────────────────────

    def _check_13_confidence_range(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 13: confidence_score in [0.0, 1.0]."""
        c = output.confidence_score
        if c < 0.0 or c > 1.0:
            return ValidatorError(
                check_number=13,
                check_name="confidence_range",
                error_description=f"confidence_score out of range: {c}",
                expected="0.0 ≤ value ≤ 1.0",
                actual=str(c),
                severity=ValidationSeverity.CRITICAL,
            )
        return None

    def _check_14_correlation_range(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 14: correlation_coefficient in [-1.0, 1.0]."""
        for corr in output.correlations:
            r = corr.correlation_coefficient
            if r < -1.0 or r > 1.0:
                return ValidatorError(
                    check_number=14,
                    check_name="correlation_range",
                    error_description=f"correlation_coefficient out of range: {r}",
                    expected="-1.0 ≤ value ≤ 1.0",
                    actual=str(r),
                    severity=ValidationSeverity.CRITICAL,
                )
        return None

    def _check_15_percentage_range(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 15: Percentage metrics in [0, 100]."""
        pct_metrics = {"cpu_percent", "memory_percent", "error_rate_percent",
                       "disk_usage_percent"}
        for am in output.anomalous_metrics:
            if am.metric_name in pct_metrics:
                if am.current_value < 0 or am.current_value > 100:
                    return ValidatorError(
                        check_number=15,
                        check_name="percentage_range",
                        error_description=(
                            f"{am.metric_name} value {am.current_value} "
                            f"out of [0, 100] range"
                        ),
                        expected="0.0 ≤ value ≤ 100.0",
                        actual=str(am.current_value),
                        severity=ValidationSeverity.WARNING,
                    )
        return None

    def _check_16_non_negative_metrics(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 16: Latency/throughput metrics ≥ 0."""
        non_neg = {
            "api_latency_p50_ms", "api_latency_p95_ms",
            "api_latency_p99_ms", "request_rate_per_sec",
            "disk_io_read_mb_per_sec", "disk_io_write_mb_per_sec",
            "network_in_mbps", "network_out_mbps",
        }
        for am in output.anomalous_metrics:
            if am.metric_name in non_neg and am.current_value < 0:
                return ValidatorError(
                    check_number=16,
                    check_name="non_negative_metrics",
                    error_description=(
                        f"{am.metric_name} has negative value: "
                        f"{am.current_value}"
                    ),
                    expected="value ≥ 0",
                    actual=str(am.current_value),
                    severity=ValidationSeverity.WARNING,
                )
        return None

    def _check_17_zscore_sanity(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 17: zscore in [-10, 10] sanity range."""
        for am in output.anomalous_metrics:
            if am.zscore < -10.0 or am.zscore > 10.0:
                return ValidatorError(
                    check_number=17,
                    check_name="zscore_sanity",
                    error_description=(
                        f"zscore {am.zscore} outside sanity range [-10, 10]"
                    ),
                    expected="-10.0 ≤ value ≤ 10.0",
                    actual=str(am.zscore),
                    severity=ValidationSeverity.WARNING,
                )
        return None

    # ── Business Rule Checks (18-21) ────────────────────────────

    def _check_18_empty_anomalies_confidence(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 18: IF no anomalies → confidence < 0.3."""
        if not output.anomalous_metrics and output.confidence_score >= 0.3:
            return ValidatorError(
                check_number=18,
                check_name="empty_anomalies_low_confidence",
                error_description=(
                    f"confidence={output.confidence_score} but 0 anomalies"
                ),
                expected="confidence < 0.3 when no anomalies",
                actual=f"confidence={output.confidence_score} with 0 anomalies",
                severity=ValidationSeverity.WARNING,
            )
        return None

    def _check_19_critical_zscore(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 19: IF severity=critical → zscore > 5.0 OR deviation > 100."""
        for am in output.anomalous_metrics:
            if am.severity == Severity.CRITICAL:
                if abs(am.zscore) <= 5.0 and abs(am.deviation_percent) <= 100.0:
                    return ValidatorError(
                        check_number=19,
                        check_name="critical_requires_extreme",
                        error_description=(
                            f"severity=critical but zscore={am.zscore} "
                            f"and deviation={am.deviation_percent}%"
                        ),
                        expected="zscore > 5.0 OR deviation > 100%",
                        actual=f"zscore={am.zscore}, deviation={am.deviation_percent}%",
                        severity=ValidationSeverity.WARNING,
                    )
        return None

    def _check_20_high_severity_check(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 20: IF severity=high → zscore > 4.0 OR threshold_breached."""
        for am in output.anomalous_metrics:
            if am.severity == Severity.HIGH:
                if abs(am.zscore) <= 4.0 and not am.threshold_breached:
                    return ValidatorError(
                        check_number=20,
                        check_name="high_severity_check",
                        error_description=(
                            f"severity=high but zscore={am.zscore} "
                            f"and threshold_breached=False"
                        ),
                        expected="zscore > 4.0 OR threshold_breached=True",
                        actual=(
                            f"zscore={am.zscore}, "
                            f"threshold_breached={am.threshold_breached}"
                        ),
                        severity=ValidationSeverity.WARNING,
                    )
        return None

    def _check_21_spike_growth_rate(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 21: IF anomaly_type=spike → growth_rate > 50."""
        for am in output.anomalous_metrics:
            if am.anomaly_type == AnomalyType.SPIKE:
                if am.growth_rate <= 50.0:
                    return ValidatorError(
                        check_number=21,
                        check_name="spike_growth_rate",
                        error_description=(
                            f"anomaly_type=spike but growth_rate={am.growth_rate}"
                        ),
                        expected="growth_rate > 50.0",
                        actual=f"growth_rate={am.growth_rate}",
                        severity=ValidationSeverity.WARNING,
                    )
        return None

    # ── Consistency Checks (22-23) ──────────────────────────────

    def _check_22_metric_names_in_input(
        self, output: MetricsAgentOutput, input_data: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 22: All anomalous metric names exist in input."""
        input_metrics = set(input_data.metrics.keys())
        for am in output.anomalous_metrics:
            if am.metric_name not in input_metrics:
                return ValidatorError(
                    check_number=22,
                    check_name="metric_names_in_input",
                    error_description=(
                        f"metric '{am.metric_name}' not in input metrics"
                    ),
                    expected=f"One of {input_metrics}",
                    actual=am.metric_name,
                    severity=ValidationSeverity.WARNING,
                )
        return None

    def _check_23_correlation_metrics_anomalous(
        self, output: MetricsAgentOutput, _: MetricsAnalysisInput
    ) -> Optional[ValidatorError]:
        """Check 23: Correlated metrics are in anomalous_metrics list."""
        anomalous_names = {am.metric_name for am in output.anomalous_metrics}
        for corr in output.correlations:
            if corr.metric_1 not in anomalous_names:
                return ValidatorError(
                    check_number=23,
                    check_name="correlation_metrics_anomalous",
                    error_description=(
                        f"correlation metric '{corr.metric_1}' "
                        f"not in anomalous_metrics"
                    ),
                    expected=f"In anomalous set: {anomalous_names}",
                    actual=corr.metric_1,
                    severity=ValidationSeverity.WARNING,
                )
            if corr.metric_2 not in anomalous_names:
                return ValidatorError(
                    check_number=23,
                    check_name="correlation_metrics_anomalous",
                    error_description=(
                        f"correlation metric '{corr.metric_2}' "
                        f"not in anomalous_metrics"
                    ),
                    expected=f"In anomalous set: {anomalous_names}",
                    actual=corr.metric_2,
                    severity=ValidationSeverity.WARNING,
                )
        return None
