"""Tests for core/rollback_planner.py — Algorithm 2."""

from __future__ import annotations

from agents.incident_commander_agent.core.rollback_planner import (
    plan_rollback,
)
from agents.incident_commander_agent.schema import RollbackStrategy
from agents.root_cause_agent.schema import CausalLink


class TestRollbackPlanner:
    def test_deployment_strategy(self):
        plan = plan_rollback("bad deployment caused outage")
        assert plan.strategy == RollbackStrategy.DEPLOYMENT_ROLLBACK
        assert plan.is_safe is True
        assert len(plan.checkpoints) > 0

    def test_config_strategy(self):
        plan = plan_rollback("configuration change broke auth")
        assert plan.strategy == RollbackStrategy.CONFIG_ROLLBACK

    def test_feature_flag_strategy(self):
        plan = plan_rollback("feature flag toggle caused errors")
        assert plan.strategy == RollbackStrategy.FEATURE_FLAG_TOGGLE

    def test_database_strategy(self):
        plan = plan_rollback("database migration failed")
        assert plan.strategy == RollbackStrategy.DATABASE_RESTORE
        assert plan.requires_data_backup is True

    def test_irreversible_returns_no_rollback(self):
        plan = plan_rollback("data_corruption in production")
        assert plan.strategy == RollbackStrategy.NO_ROLLBACK
        assert plan.is_safe is False
        assert len(plan.risks) > 0

    def test_irreversible_via_causal_chain(self):
        chain = [
            CausalLink(cause="bug", effect="data_loss", confidence=0.9),
        ]
        plan = plan_rollback("unknown bug", causal_chain=chain)
        assert plan.strategy == RollbackStrategy.NO_ROLLBACK

    def test_rollback_has_steps(self):
        plan = plan_rollback("deployment failure")
        assert len(plan.rollback_steps) > 0
        assert plan.estimated_minutes > 0
