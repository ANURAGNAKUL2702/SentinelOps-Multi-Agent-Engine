"""
File: core/rollback_planner.py
Purpose: Plan safe rollback strategies with checkpoints.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 2: Analyze root cause and timeline to determine if rollback
is safe, choose strategy, and define verification checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Set

from agents.incident_commander_agent.schema import (
    CausalLink,
    RemediationStep,
    RollbackCheckpoint,
    RollbackPlan,
    RollbackStrategy,
    TimelineEvent,
)


# ═══════════════════════════════════════════════════════════════
#  STRATEGY SELECTION
# ═══════════════════════════════════════════════════════════════

_STRATEGY_MAP: Dict[str, RollbackStrategy] = {
    "deployment": RollbackStrategy.DEPLOYMENT_ROLLBACK,
    "deploy": RollbackStrategy.DEPLOYMENT_ROLLBACK,
    "release": RollbackStrategy.DEPLOYMENT_ROLLBACK,
    "config": RollbackStrategy.CONFIG_ROLLBACK,
    "configuration": RollbackStrategy.CONFIG_ROLLBACK,
    "setting": RollbackStrategy.CONFIG_ROLLBACK,
    "feature": RollbackStrategy.FEATURE_FLAG_TOGGLE,
    "flag": RollbackStrategy.FEATURE_FLAG_TOGGLE,
    "toggle": RollbackStrategy.FEATURE_FLAG_TOGGLE,
    "traffic": RollbackStrategy.TRAFFIC_REROUTE,
    "routing": RollbackStrategy.TRAFFIC_REROUTE,
    "loadbalancer": RollbackStrategy.TRAFFIC_REROUTE,
    "database": RollbackStrategy.DATABASE_RESTORE,
    "data": RollbackStrategy.DATABASE_RESTORE,
    "corruption": RollbackStrategy.DATABASE_RESTORE,
    "migration": RollbackStrategy.DATABASE_RESTORE,
}

_IRREVERSIBLE_KEYWORDS: Set[str] = {
    "data_corruption", "data_loss", "schema_migration",
    "irreversible", "deleted", "purged", "truncated",
}


def _select_strategy(root_cause: str) -> RollbackStrategy:
    """Select rollback strategy based on root cause keywords.

    Args:
        root_cause: Root cause string.

    Returns:
        Best matching RollbackStrategy.
    """
    normalized = root_cause.lower().replace("-", "_").replace(" ", "_")

    for keyword, strategy in _STRATEGY_MAP.items():
        if keyword in normalized:
            return strategy

    return RollbackStrategy.DEPLOYMENT_ROLLBACK


def _is_irreversible(root_cause: str, causal_chain: List[CausalLink]) -> bool:
    """Check if the failure is likely irreversible.

    Args:
        root_cause: Root cause string.
        causal_chain: Causal chain links.

    Returns:
        True if rollback is likely impossible.
    """
    normalized = root_cause.lower().replace("-", "_").replace(" ", "_")
    for keyword in _IRREVERSIBLE_KEYWORDS:
        if keyword in normalized:
            return True
    for link in causal_chain:
        combined = f"{link.cause} {link.effect}".lower()
        for keyword in _IRREVERSIBLE_KEYWORDS:
            if keyword in combined:
                return True
    return False


def _build_checkpoints(
    strategy: RollbackStrategy,
) -> List[RollbackCheckpoint]:
    """Build verification checkpoints for the rollback strategy.

    Args:
        strategy: The chosen rollback strategy.

    Returns:
        List of checkpoints.
    """
    common = [
        RollbackCheckpoint(
            checkpoint_name="pre_rollback_health",
            verification_command="kubectl get pods --field-selector=status.phase=Running",
            expected_result="All critical pods running",
        ),
    ]

    strategy_specific: Dict[RollbackStrategy, List[RollbackCheckpoint]] = {
        RollbackStrategy.DEPLOYMENT_ROLLBACK: [
            RollbackCheckpoint(
                checkpoint_name="deployment_history",
                verification_command="kubectl rollout history deployment/<service>",
                expected_result="Previous revision available",
            ),
            RollbackCheckpoint(
                checkpoint_name="post_rollback_health",
                verification_command="curl -s http://<service>/health",
                expected_result="HTTP 200 OK",
            ),
        ],
        RollbackStrategy.CONFIG_ROLLBACK: [
            RollbackCheckpoint(
                checkpoint_name="config_backup",
                verification_command="kubectl get configmap <config> -o yaml",
                expected_result="Previous config captured",
            ),
        ],
        RollbackStrategy.FEATURE_FLAG_TOGGLE: [
            RollbackCheckpoint(
                checkpoint_name="flag_state",
                verification_command="curl -s http://feature-service/flags/<flag>",
                expected_result="Flag state verified",
            ),
        ],
        RollbackStrategy.TRAFFIC_REROUTE: [
            RollbackCheckpoint(
                checkpoint_name="traffic_distribution",
                verification_command="kubectl get virtualservice -o yaml",
                expected_result="Traffic correctly distributed",
            ),
        ],
        RollbackStrategy.DATABASE_RESTORE: [
            RollbackCheckpoint(
                checkpoint_name="backup_available",
                verification_command="aws rds describe-db-snapshots --db-instance-identifier <db>",
                expected_result="Recent snapshot available",
            ),
        ],
    }

    return common + strategy_specific.get(strategy, [])


def _build_rollback_steps(
    strategy: RollbackStrategy,
) -> List[RemediationStep]:
    """Build rollback execution steps.

    Args:
        strategy: The chosen rollback strategy.

    Returns:
        Ordered list of rollback steps.
    """
    steps_map: Dict[RollbackStrategy, List[Dict]] = {
        RollbackStrategy.DEPLOYMENT_ROLLBACK: [
            {"desc": "Rollback deployment to previous revision",
             "cmd": "kubectl rollout undo deployment/<service>",
             "outcome": "Previous deployment version running",
             "minutes": 5.0},
            {"desc": "Verify new pods are healthy",
             "cmd": "kubectl rollout status deployment/<service>",
             "outcome": "Rollout complete, all pods ready",
             "minutes": 3.0},
        ],
        RollbackStrategy.CONFIG_ROLLBACK: [
            {"desc": "Restore previous configuration",
             "cmd": "kubectl apply -f config-backup.yaml",
             "outcome": "Previous config applied",
             "minutes": 2.0},
            {"desc": "Restart services to pick up config",
             "cmd": "kubectl rollout restart deployment/<service>",
             "outcome": "Services running with old config",
             "minutes": 5.0},
        ],
        RollbackStrategy.FEATURE_FLAG_TOGGLE: [
            {"desc": "Toggle feature flag off",
             "cmd": "curl -X PUT http://feature-service/flags/<flag> -d '{\"enabled\": false}'",
             "outcome": "Feature flag disabled",
             "minutes": 1.0},
        ],
        RollbackStrategy.TRAFFIC_REROUTE: [
            {"desc": "Reroute traffic to healthy instances",
             "cmd": "kubectl apply -f traffic-failover.yaml",
             "outcome": "Traffic routed away from failed service",
             "minutes": 3.0},
        ],
        RollbackStrategy.DATABASE_RESTORE: [
            {"desc": "Initiate point-in-time database restore",
             "cmd": "aws rds restore-db-instance-to-point-in-time --source-db-instance-identifier <db> --target-db-instance-identifier <db>-restored",
             "outcome": "Database restored from backup",
             "minutes": 30.0,
             "destructive": True},
        ],
    }

    steps_data = steps_map.get(strategy, [])
    return [
        RemediationStep(
            step_number=i + 1,
            description=s["desc"],
            command=s["cmd"],
            expected_outcome=s["outcome"],
            estimated_minutes=s.get("minutes", 5.0),
            is_destructive=s.get("destructive", False),
        )
        for i, s in enumerate(steps_data)
    ]


def _identify_risks(
    strategy: RollbackStrategy,
    root_cause: str,
) -> List[str]:
    """Identify risks associated with the rollback.

    Args:
        strategy: Rollback strategy.
        root_cause: Root cause string.

    Returns:
        List of risk descriptions.
    """
    risks: List[str] = []

    if strategy == RollbackStrategy.DATABASE_RESTORE:
        risks.append("Potential data loss for transactions after backup point")
        risks.append("Extended downtime during restore (~30 min)")

    if strategy == RollbackStrategy.DEPLOYMENT_ROLLBACK:
        risks.append("Rolling update may cause brief service interruption")

    if "cascade" in root_cause.lower() or "cascading" in root_cause.lower():
        risks.append("Rollback may trigger secondary cascading effects")

    if strategy == RollbackStrategy.TRAFFIC_REROUTE:
        risks.append("Backup region may have stale data or higher latency")

    return risks


def plan_rollback(
    root_cause: str,
    timeline: List[TimelineEvent] | None = None,
    causal_chain: List[CausalLink] | None = None,
) -> RollbackPlan:
    """Plan a safe rollback strategy.

    Args:
        root_cause: Root cause string.
        timeline: Incident timeline events.
        causal_chain: Causal chain from root cause agent.

    Returns:
        RollbackPlan with strategy, checkpoints, and steps.
    """
    chain = causal_chain or []

    if _is_irreversible(root_cause, chain):
        return RollbackPlan(
            strategy=RollbackStrategy.NO_ROLLBACK,
            is_safe=False,
            risks=[
                "Rollback not possible: failure involves irreversible changes",
                "Manual intervention required to assess data integrity",
            ],
            requires_data_backup=True,
        )

    strategy = _select_strategy(root_cause)
    checkpoints = _build_checkpoints(strategy)
    rollback_steps = _build_rollback_steps(strategy)
    risks = _identify_risks(strategy, root_cause)

    total_minutes = sum(s.estimated_minutes for s in rollback_steps)

    return RollbackPlan(
        strategy=strategy,
        is_safe=strategy != RollbackStrategy.DATABASE_RESTORE,
        checkpoints=checkpoints,
        rollback_steps=rollback_steps,
        risks=risks,
        estimated_minutes=total_minutes,
        requires_data_backup=strategy == RollbackStrategy.DATABASE_RESTORE,
    )
