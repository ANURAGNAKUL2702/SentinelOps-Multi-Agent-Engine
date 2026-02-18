"""Tests for core/runbook_generator.py — Algorithm 1."""

from __future__ import annotations

from agents.incident_commander_agent.core.runbook_generator import (
    generate_runbook,
)
from agents.root_cause_agent.schema import CausalLink


class TestRunbookGenerator:
    def test_known_root_cause_db_pool(self):
        rb = generate_runbook(
            "database_connection_pool_exhaustion",
            ["payment-service"],
        )
        assert rb.title != ""
        assert len(rb.steps) > 0
        assert rb.steps[0].step_number == 1
        assert rb.estimated_total_minutes > 0

    def test_known_root_cause_memory_leak(self):
        rb = generate_runbook("memory_leak", ["api-service"])
        assert "memory" in rb.title.lower() or "memory" in rb.root_cause_category.lower()
        assert len(rb.steps) >= 4

    def test_unknown_root_cause_generic(self):
        rb = generate_runbook("something_completely_new", ["svc-x"])
        assert len(rb.steps) >= 3
        assert "investigation" in rb.title.lower() or "generic" in rb.title.lower() or rb.steps[0].step_number == 1

    def test_service_name_substitution(self):
        rb = generate_runbook(
            "database_connection_pool_exhaustion",
            ["my-database"],
        )
        steps_text = " ".join(s.command for s in rb.steps)
        # Should substitute placeholder with actual service
        assert "<service>" not in steps_text or "my-database" in steps_text

    def test_causal_chain_affects_services(self):
        chain = [
            CausalLink(cause="db-failure", effect="api-timeout", confidence=0.8),
        ]
        rb = generate_runbook("cascading_failure", ["api-service"], chain)
        assert len(rb.steps) >= 4

    def test_steps_are_sequential(self):
        rb = generate_runbook("network_partition", ["svc-a", "svc-b"])
        nums = [s.step_number for s in rb.steps]
        assert nums == list(range(1, len(nums) + 1))
