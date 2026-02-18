"""
File: core/runbook_generator.py
Purpose: Generate step-by-step remediation runbooks from root cause.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 1: Map root cause to remediation templates with concrete
kubectl/SQL commands, expected outcomes, and validation checks.
"""

from __future__ import annotations

from typing import Dict, List

from agents.incident_commander_agent.schema import (
    CausalLink,
    RemediationStep,
    Runbook,
)


# ═══════════════════════════════════════════════════════════════
#  RUNBOOK TEMPLATES
# ═══════════════════════════════════════════════════════════════

_TEMPLATES: Dict[str, List[Dict[str, str | float | bool]]] = {
    "database_connection_pool_exhaustion": [
        {
            "description": "Check current pool size and active connections",
            "command": "SHOW PROCESSLIST",
            "expected_outcome": "See active connections and identify blockers",
            "validation_check": "SELECT COUNT(*) FROM information_schema.processlist",
            "estimated_minutes": 2.0,
            "is_destructive": False,
        },
        {
            "description": "Identify long-running queries",
            "command": "SELECT * FROM information_schema.processlist WHERE time > 30 ORDER BY time DESC",
            "expected_outcome": "List of queries running longer than 30 seconds",
            "validation_check": "Verify query list is populated",
            "estimated_minutes": 1.0,
            "is_destructive": False,
        },
        {
            "description": "Kill blocking queries",
            "command": "KILL QUERY <id>",
            "expected_outcome": "Long-running queries terminated",
            "validation_check": "SHOW PROCESSLIST shows reduced active queries",
            "estimated_minutes": 2.0,
            "is_destructive": True,
        },
        {
            "description": "Increase connection pool size",
            "command": "kubectl set env deployment/db MAX_POOL_SIZE=200",
            "expected_outcome": "Pool size increased to accommodate load",
            "validation_check": "kubectl get deployment/db -o jsonpath='{.spec.template.spec.containers[0].env}'",
            "estimated_minutes": 3.0,
            "is_destructive": False,
        },
        {
            "description": "Restart affected application services",
            "command": "kubectl rollout restart deployment/<service>",
            "expected_outcome": "Services reconnect with new pool settings",
            "validation_check": "kubectl get pods -l app=<service> --field-selector=status.phase=Running",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
        {
            "description": "Monitor connection pool metrics",
            "command": "kubectl top pods && curl -s http://<service>:9090/metrics | grep pool",
            "expected_outcome": "Pool utilisation below 80%",
            "validation_check": "Grafana dashboard shows healthy pool metrics",
            "estimated_minutes": 2.0,
            "is_destructive": False,
        },
    ],
    "memory_leak": [
        {
            "description": "Identify service with memory leak from metrics",
            "command": "kubectl top pods --sort-by=memory | head -20",
            "expected_outcome": "Identify pods with abnormal memory usage",
            "validation_check": "Pod memory usage values retrieved",
            "estimated_minutes": 1.0,
            "is_destructive": False,
        },
        {
            "description": "Capture heap dump for analysis",
            "command": "kubectl exec <pod> -- jmap -dump:live,format=b,file=/tmp/heap.hprof 1",
            "expected_outcome": "Heap dump file created for analysis",
            "validation_check": "kubectl exec <pod> -- ls -la /tmp/heap.hprof",
            "estimated_minutes": 3.0,
            "is_destructive": False,
        },
        {
            "description": "Restart service with increased memory limit",
            "command": "kubectl set resources deployment/<service> -c <container> --limits=memory=4Gi",
            "expected_outcome": "Service running with more memory headroom",
            "validation_check": "kubectl get pods -l app=<service> --field-selector=status.phase=Running",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
        {
            "description": "Analyze heap dump to find leak source",
            "command": "jhat /tmp/heap.hprof",
            "expected_outcome": "Leak source identified in object retention graph",
            "validation_check": "Leak class and allocation site documented",
            "estimated_minutes": 10.0,
            "is_destructive": False,
        },
        {
            "description": "Deploy fix or add memory monitoring alerts",
            "command": "kubectl apply -f memory-alert-rule.yaml",
            "expected_outcome": "Memory alerts configured for early detection",
            "validation_check": "Alert rule visible in Prometheus/Alertmanager",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
    ],
    "network_partition": [
        {
            "description": "Verify network connectivity between services",
            "command": "kubectl exec <pod> -- ping -c 3 <target-service>",
            "expected_outcome": "Connectivity confirmed or timeout identified",
            "validation_check": "Ping results show 0% packet loss",
            "estimated_minutes": 1.0,
            "is_destructive": False,
        },
        {
            "description": "Check network agent logs for partition events",
            "command": "kubectl logs -l app=network-agent --tail=100 | grep -i partition",
            "expected_outcome": "Partition events identified with timestamps",
            "validation_check": "Log entries show partition start/end times",
            "estimated_minutes": 2.0,
            "is_destructive": False,
        },
        {
            "description": "Restart network components",
            "command": "kubectl rollout restart daemonset/kube-proxy && kubectl rollout restart deployment/coredns -n kube-system",
            "expected_outcome": "Network components refreshed",
            "validation_check": "kubectl get pods -n kube-system | grep -E 'kube-proxy|coredns'",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
        {
            "description": "Failover to backup region if available",
            "command": "kubectl config use-context backup-region && kubectl apply -f failover-config.yaml",
            "expected_outcome": "Traffic routed to backup region",
            "validation_check": "curl -s https://<service>/health returns 200",
            "estimated_minutes": 10.0,
            "is_destructive": False,
        },
        {
            "description": "Re-establish connections and verify health",
            "command": "kubectl rollout restart deployment/<service>",
            "expected_outcome": "All services reconnected and healthy",
            "validation_check": "kubectl get pods --field-selector=status.phase=Running",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
    ],
    "cascading_failure": [
        {
            "description": "Isolate the failed service to stop cascade",
            "command": "kubectl scale deployment/<failed-service> --replicas=0",
            "expected_outcome": "Failed service isolated, cascade halted",
            "validation_check": "kubectl get deployment/<failed-service> shows 0 replicas",
            "estimated_minutes": 1.0,
            "is_destructive": True,
        },
        {
            "description": "Activate circuit breakers on dependent services",
            "command": "kubectl set env deployment/<service> CIRCUIT_BREAKER_ENABLED=true",
            "expected_outcome": "Circuit breakers prevent further propagation",
            "validation_check": "Service logs show circuit breaker OPEN state",
            "estimated_minutes": 2.0,
            "is_destructive": False,
        },
        {
            "description": "Restart dependent services in dependency order",
            "command": "kubectl rollout restart deployment/<service>",
            "expected_outcome": "Services recover in correct order",
            "validation_check": "kubectl get pods --field-selector=status.phase=Running",
            "estimated_minutes": 10.0,
            "is_destructive": False,
        },
        {
            "description": "Gradually restore isolated service",
            "command": "kubectl scale deployment/<failed-service> --replicas=1",
            "expected_outcome": "Service back online with single replica",
            "validation_check": "curl -s http://<failed-service>/health returns 200",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
        {
            "description": "Scale to full capacity and monitor",
            "command": "kubectl scale deployment/<failed-service> --replicas=3",
            "expected_outcome": "Full capacity restored, no cascade recurrence",
            "validation_check": "Grafana dashboard shows stable error rates",
            "estimated_minutes": 5.0,
            "is_destructive": False,
        },
    ],
}


def _normalize_root_cause(root_cause: str) -> str:
    """Normalize root cause string for template matching.

    Args:
        root_cause: Raw root cause string.

    Returns:
        Normalized lowercase string with underscores.
    """
    return root_cause.lower().strip().replace(" ", "_").replace("-", "_")


def _match_template(root_cause: str) -> str:
    """Match root cause to a template key.

    Args:
        root_cause: Normalized root cause string.

    Returns:
        Template key, or empty string if no match.
    """
    normalized = _normalize_root_cause(root_cause)

    # Exact match
    if normalized in _TEMPLATES:
        return normalized

    # Substring match
    for key in _TEMPLATES:
        if key in normalized or normalized in key:
            return key

    # Keyword match
    keyword_map = {
        "pool": "database_connection_pool_exhaustion",
        "connection": "database_connection_pool_exhaustion",
        "database": "database_connection_pool_exhaustion",
        "memory": "memory_leak",
        "leak": "memory_leak",
        "oom": "memory_leak",
        "network": "network_partition",
        "partition": "network_partition",
        "dns": "network_partition",
        "cascade": "cascading_failure",
        "cascading": "cascading_failure",
        "propagat": "cascading_failure",
    }
    for keyword, template_key in keyword_map.items():
        if keyword in normalized:
            return template_key

    return ""


def _build_generic_runbook(
    root_cause: str,
    affected_services: List[str],
) -> Runbook:
    """Build a generic investigation runbook for unknown root causes.

    Args:
        root_cause: The unknown root cause.
        affected_services: List of affected service names.

    Returns:
        Generic investigation Runbook.
    """
    svc_list = ", ".join(affected_services[:5]) or "affected services"
    steps = [
        RemediationStep(
            step_number=1,
            description=f"Investigate root cause: {root_cause}",
            command=f"kubectl logs -l app={{service}} --tail=200 | grep -i error",
            expected_outcome="Error patterns identified in logs",
            validation_check="Relevant log entries collected",
            estimated_minutes=5.0,
        ),
        RemediationStep(
            step_number=2,
            description="Check service health and metrics",
            command=f"kubectl get pods -o wide && kubectl top pods",
            expected_outcome="Service status and resource usage visible",
            validation_check="All pod statuses retrieved",
            estimated_minutes=2.0,
        ),
        RemediationStep(
            step_number=3,
            description=f"Restart affected services: {svc_list}",
            command="kubectl rollout restart deployment/<service>",
            expected_outcome="Services restarted successfully",
            validation_check="kubectl get pods --field-selector=status.phase=Running",
            estimated_minutes=5.0,
        ),
        RemediationStep(
            step_number=4,
            description="Monitor services for recovery",
            command="watch -n 5 'kubectl get pods && kubectl top pods'",
            expected_outcome="All services healthy and stable",
            validation_check="No error logs in the last 5 minutes",
            estimated_minutes=10.0,
        ),
    ]
    total = sum(s.estimated_minutes for s in steps)
    return Runbook(
        title=f"Investigation Runbook: {root_cause}",
        root_cause_category="unknown",
        steps=steps,
        estimated_total_minutes=total,
        requires_approval=True,
    )


def generate_runbook(
    root_cause: str,
    affected_services: List[str],
    causal_chain: List[CausalLink] | None = None,
) -> Runbook:
    """Generate a step-by-step remediation runbook.

    Maps root cause to remediation templates. Falls back to
    a generic investigation runbook for unknown root causes.

    Args:
        root_cause: Root cause string.
        affected_services: List of affected service names.
        causal_chain: Optional causal chain for context.

    Returns:
        Runbook with ordered remediation steps.
    """
    template_key = _match_template(root_cause)

    if not template_key:
        return _build_generic_runbook(root_cause, affected_services)

    template = _TEMPLATES[template_key]
    steps: List[RemediationStep] = []

    for i, step_data in enumerate(template, start=1):
        cmd = str(step_data.get("command", ""))
        desc = str(step_data.get("description", ""))

        # Substitute service names into commands
        if affected_services and "<service>" in cmd:
            cmd = cmd.replace("<service>", affected_services[0])
        if affected_services and "<failed-service>" in cmd:
            cmd = cmd.replace("<failed-service>", affected_services[0])

        steps.append(RemediationStep(
            step_number=i,
            description=desc,
            command=cmd,
            expected_outcome=str(step_data.get("expected_outcome", "")),
            validation_check=str(step_data.get("validation_check", "")),
            estimated_minutes=float(step_data.get("estimated_minutes", 2.0)),
            is_destructive=bool(step_data.get("is_destructive", False)),
        ))

    total = sum(s.estimated_minutes for s in steps)
    title = template_key.replace("_", " ").title()

    return Runbook(
        title=f"Runbook: {title}",
        root_cause_category=template_key,
        steps=steps,
        estimated_total_minutes=total,
        requires_approval=any(s.is_destructive for s in steps),
    )
