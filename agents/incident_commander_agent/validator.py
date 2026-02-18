"""
File: validator.py
Purpose: Validate IncidentResponse output with 28+ checks.
Dependencies: schema models only.
Performance: <5ms total validation time.

Each check is a method _check_NN returning a list of
CommandValidatorError. CRITICAL failures prevent deployment;
WARNING failures are advisory.
"""

from __future__ import annotations

import time
from typing import Callable, List

from agents.incident_commander_agent.schema import (
    ActionPriority,
    CommandValidationSeverity,
    CommandValidatorError,
    CommandValidatorResult,
    IncidentResponse,
    IncidentSeverity,
    RollbackStrategy,
)


class CommandValidator:
    """Validate IncidentResponse output.

    Runs 28+ ordered checks and returns a
    CommandValidatorResult.
    """

    def validate(self, response: IncidentResponse) -> CommandValidatorResult:
        """Run all checks and return result.

        Args:
            response: The IncidentResponse to validate.

        Returns:
            CommandValidatorResult with errors and warnings.
        """
        start = time.perf_counter()

        checks: List[Callable[[IncidentResponse], List[CommandValidatorError]]] = [
            self._check_01_agent_name,
            self._check_02_incident_id,
            self._check_03_root_cause_summary,
            self._check_04_severity_valid,
            self._check_05_blast_radius_services,
            self._check_06_blast_radius_count_matches,
            self._check_07_blast_radius_users_non_negative,
            self._check_08_blast_radius_availability_range,
            self._check_09_runbook_has_steps,
            self._check_10_runbook_steps_ordered,
            self._check_11_runbook_steps_have_commands,
            self._check_12_runbook_title,
            self._check_13_action_items_exist,
            self._check_14_action_items_unique_ids,
            self._check_15_action_items_valid_deps,
            self._check_16_action_items_no_self_dep,
            self._check_17_rollback_strategy_valid,
            self._check_18_rollback_has_checkpoints,
            self._check_19_communication_status_update,
            self._check_20_communication_channels,
            self._check_21_prevention_has_items,
            self._check_22_prevention_categories_valid,
            self._check_23_escalation_has_reason,
            self._check_24_escalation_confidence_range,
            self._check_25_metadata_latency_non_negative,
            self._check_26_pipeline_latency_reasonable,
            self._check_27_correlation_id_present,
            self._check_28_classification_source_valid,
        ]

        errors: List[CommandValidatorError] = []
        warnings: List[CommandValidatorError] = []

        for check in checks:
            results = check(response)
            for err in results:
                if err.severity == CommandValidationSeverity.CRITICAL:
                    errors.append(err)
                else:
                    warnings.append(err)

        elapsed = (time.perf_counter() - start) * 1000
        passed = len(errors) == 0

        return CommandValidatorResult(
            validation_passed=passed,
            total_checks=len(checks),
            errors=errors,
            warnings=warnings,
            validation_latency_ms=round(elapsed, 3),
        )

    # ─── Individual checks ─────────────────────────────────

    @staticmethod
    def _check_01_agent_name(r: IncidentResponse) -> List[CommandValidatorError]:
        if r.agent != "incident_commander_agent":
            return [CommandValidatorError(
                check_number=1, check_name="agent_name",
                error_description="Agent name mismatch",
                expected="incident_commander_agent", actual=r.agent,
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_02_incident_id(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.incident_id:
            return [CommandValidatorError(
                check_number=2, check_name="incident_id_present",
                error_description="incident_id is empty",
                expected="non-empty string", actual="",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_03_root_cause_summary(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.root_cause_summary:
            return [CommandValidatorError(
                check_number=3, check_name="root_cause_summary",
                error_description="root_cause_summary is empty",
                expected="non-empty", actual="",
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_04_severity_valid(r: IncidentResponse) -> List[CommandValidatorError]:
        try:
            IncidentSeverity(r.severity.value if isinstance(r.severity, IncidentSeverity) else r.severity)
        except ValueError:
            return [CommandValidatorError(
                check_number=4, check_name="severity_valid",
                error_description="Invalid severity value",
                expected="P0-P3", actual=str(r.severity),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_05_blast_radius_services(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.blast_radius.affected_services:
            return [CommandValidatorError(
                check_number=5, check_name="blast_radius_services",
                error_description="No affected services in blast radius",
                expected="at least 1 service", actual="0",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_06_blast_radius_count_matches(r: IncidentResponse) -> List[CommandValidatorError]:
        br = r.blast_radius
        if br.affected_service_count != len(br.affected_services):
            return [CommandValidatorError(
                check_number=6, check_name="blast_radius_count",
                error_description="affected_service_count != len(affected_services)",
                expected=str(len(br.affected_services)),
                actual=str(br.affected_service_count),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_07_blast_radius_users_non_negative(r: IncidentResponse) -> List[CommandValidatorError]:
        if r.blast_radius.estimated_users < 0:
            return [CommandValidatorError(
                check_number=7, check_name="blast_radius_users",
                error_description="Negative estimated_users",
                expected=">=0", actual=str(r.blast_radius.estimated_users),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_08_blast_radius_availability_range(r: IncidentResponse) -> List[CommandValidatorError]:
        val = r.blast_radius.availability_impact
        if not (0.0 <= val <= 1.0):
            return [CommandValidatorError(
                check_number=8, check_name="blast_radius_availability",
                error_description="availability_impact out of [0,1]",
                expected="0.0-1.0", actual=str(val),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_09_runbook_has_steps(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.runbook.steps:
            return [CommandValidatorError(
                check_number=9, check_name="runbook_has_steps",
                error_description="Runbook has no steps",
                expected="at least 1 step", actual="0",
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_10_runbook_steps_ordered(r: IncidentResponse) -> List[CommandValidatorError]:
        nums = [s.step_number for s in r.runbook.steps]
        expected = list(range(1, len(nums) + 1))
        if nums != expected:
            return [CommandValidatorError(
                check_number=10, check_name="runbook_steps_ordered",
                error_description="Step numbers not sequential from 1",
                expected=str(expected), actual=str(nums),
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_11_runbook_steps_have_commands(r: IncidentResponse) -> List[CommandValidatorError]:
        for step in r.runbook.steps:
            if not step.command:
                return [CommandValidatorError(
                    check_number=11, check_name="runbook_step_commands",
                    error_description=f"Step {step.step_number} has no command",
                    expected="non-empty command", actual="",
                    severity=CommandValidationSeverity.WARNING,
                )]
        return []

    @staticmethod
    def _check_12_runbook_title(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.runbook.title:
            return [CommandValidatorError(
                check_number=12, check_name="runbook_title",
                error_description="Runbook title is empty",
                expected="non-empty", actual="",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_13_action_items_exist(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.action_items:
            return [CommandValidatorError(
                check_number=13, check_name="action_items_exist",
                error_description="No action items generated",
                expected="at least 1", actual="0",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_14_action_items_unique_ids(r: IncidentResponse) -> List[CommandValidatorError]:
        ids = [a.action_id for a in r.action_items]
        if len(ids) != len(set(ids)):
            return [CommandValidatorError(
                check_number=14, check_name="action_items_unique_ids",
                error_description="Duplicate action_id values",
                expected="all unique", actual=str(ids),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_15_action_items_valid_deps(r: IncidentResponse) -> List[CommandValidatorError]:
        valid_ids = {a.action_id for a in r.action_items}
        for action in r.action_items:
            for dep in action.dependencies:
                if dep not in valid_ids:
                    return [CommandValidatorError(
                        check_number=15, check_name="action_items_deps",
                        error_description=f"Invalid dependency: {dep}",
                        expected="existing action_id", actual=dep,
                        severity=CommandValidationSeverity.CRITICAL,
                    )]
        return []

    @staticmethod
    def _check_16_action_items_no_self_dep(r: IncidentResponse) -> List[CommandValidatorError]:
        for action in r.action_items:
            if action.action_id in action.dependencies:
                return [CommandValidatorError(
                    check_number=16, check_name="action_items_self_dep",
                    error_description=f"Action {action.action_id} depends on itself",
                    expected="no self-dependency", actual=action.action_id,
                    severity=CommandValidationSeverity.CRITICAL,
                )]
        return []

    @staticmethod
    def _check_17_rollback_strategy_valid(r: IncidentResponse) -> List[CommandValidatorError]:
        try:
            RollbackStrategy(
                r.rollback_plan.strategy.value
                if isinstance(r.rollback_plan.strategy, RollbackStrategy)
                else r.rollback_plan.strategy
            )
        except ValueError:
            return [CommandValidatorError(
                check_number=17, check_name="rollback_strategy",
                error_description="Invalid rollback strategy",
                expected="valid RollbackStrategy", actual=str(r.rollback_plan.strategy),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_18_rollback_has_checkpoints(r: IncidentResponse) -> List[CommandValidatorError]:
        rp = r.rollback_plan
        if rp.strategy != RollbackStrategy.NO_ROLLBACK and not rp.checkpoints:
            return [CommandValidatorError(
                check_number=18, check_name="rollback_checkpoints",
                error_description="Rollback plan missing checkpoints",
                expected="at least 1 checkpoint", actual="0",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_19_communication_status_update(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.communication_plan.status_update:
            return [CommandValidatorError(
                check_number=19, check_name="comm_status_update",
                error_description="Communication status_update is empty",
                expected="non-empty", actual="",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_20_communication_channels(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.communication_plan.notification_channels:
            return [CommandValidatorError(
                check_number=20, check_name="comm_channels",
                error_description="No notification channels configured",
                expected="at least 1 channel", actual="0",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_21_prevention_has_items(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.prevention_recommendations:
            return [CommandValidatorError(
                check_number=21, check_name="prevention_exists",
                error_description="No prevention recommendations",
                expected="at least 1", actual="0",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_22_prevention_categories_valid(r: IncidentResponse) -> List[CommandValidatorError]:
        from agents.incident_commander_agent.schema import PreventionCategory
        valid = set(PreventionCategory)
        for p in r.prevention_recommendations:
            if p.category not in valid:
                return [CommandValidatorError(
                    check_number=22, check_name="prevention_category",
                    error_description=f"Invalid prevention category: {p.category}",
                    expected="valid PreventionCategory", actual=str(p.category),
                    severity=CommandValidationSeverity.CRITICAL,
                )]
        return []

    @staticmethod
    def _check_23_escalation_has_reason(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.escalation_decision.reason:
            return [CommandValidatorError(
                check_number=23, check_name="escalation_reason",
                error_description="Escalation decision has no reason",
                expected="non-empty", actual="",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_24_escalation_confidence_range(r: IncidentResponse) -> List[CommandValidatorError]:
        val = r.escalation_decision.auto_resolve_confidence
        if not (0.0 <= val <= 1.0):
            return [CommandValidatorError(
                check_number=24, check_name="escalation_confidence",
                error_description="auto_resolve_confidence out of [0,1]",
                expected="0.0-1.0", actual=str(val),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_25_metadata_latency_non_negative(r: IncidentResponse) -> List[CommandValidatorError]:
        if r.metadata and r.metadata.total_pipeline_ms < 0:
            return [CommandValidatorError(
                check_number=25, check_name="metadata_latency",
                error_description="Negative total_pipeline_ms",
                expected=">=0", actual=str(r.metadata.total_pipeline_ms),
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []

    @staticmethod
    def _check_26_pipeline_latency_reasonable(r: IncidentResponse) -> List[CommandValidatorError]:
        if r.pipeline_latency_ms > 10_000:
            return [CommandValidatorError(
                check_number=26, check_name="pipeline_latency",
                error_description="Pipeline latency exceeds 10s",
                expected="<10000ms", actual=str(r.pipeline_latency_ms),
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_27_correlation_id_present(r: IncidentResponse) -> List[CommandValidatorError]:
        if not r.correlation_id:
            return [CommandValidatorError(
                check_number=27, check_name="correlation_id",
                error_description="correlation_id is empty",
                expected="non-empty", actual="",
                severity=CommandValidationSeverity.WARNING,
            )]
        return []

    @staticmethod
    def _check_28_classification_source_valid(r: IncidentResponse) -> List[CommandValidatorError]:
        valid = {"llm", "fallback", "deterministic"}
        if r.classification_source not in valid:
            return [CommandValidatorError(
                check_number=28, check_name="classification_source",
                error_description=f"Invalid classification_source: {r.classification_source}",
                expected="llm|fallback|deterministic",
                actual=r.classification_source,
                severity=CommandValidationSeverity.CRITICAL,
            )]
        return []
