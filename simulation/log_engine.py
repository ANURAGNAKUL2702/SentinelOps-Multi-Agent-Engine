"""
log_engine.py — Structured log generator.

Produces realistic application log entries for every service, in both
normal and failure modes.  Log messages are tailored to match the
specific failure profile so AI agents can correlate logs with metrics.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from simulation.services import get_service_names


# ── message templates ───────────────────────────────────────────────

_NORMAL_INFO_MESSAGES: Dict[str, List[str]] = {
    "api-gateway": [
        "Request routed successfully to downstream service.",
        "Health check passed — all upstreams healthy.",
        "Rate limiter bucket refilled.",
        "TLS handshake completed for inbound connection.",
    ],
    "auth-service": [
        "Token validated successfully for user session.",
        "OAuth2 refresh token issued.",
        "SSO callback processed — session created.",
        "Password hash comparison passed.",
    ],
    "payment-service": [
        "Payment authorisation approved — txn_id={txn}.",
        "Settlement batch queued — 128 transactions.",
        "Refund processed successfully — txn_id={txn}.",
        "Idempotency key cache hit — duplicate request ignored.",
    ],
    "fraud-service": [
        "Transaction scored — risk=LOW.",
        "Rule engine evaluation completed in 12 ms.",
        "Model inference latency within SLA.",
        "Fraud check passed for merchant_id=M-10234.",
    ],
    "notification-service": [
        "Email dispatched to merchant — template=payment_received.",
        "SMS queued — provider=twilio.",
        "Push notification delivered.",
        "Webhook callback acknowledged — 200 OK.",
    ],
    "database": [
        "Connection pool utilisation at 42 %.",
        "Checkpoint completed — WAL flushed.",
        "Replication lag: 0.3 s.",
        "Vacuum completed on table 'transactions'.",
    ],
    "cache-service": [
        "Cache hit ratio: 94.2 %.",
        "Eviction cycle completed — 230 keys removed.",
        "Redis PING — PONG received in 1 ms.",
        "Session key refreshed.",
    ],
    "merchant-portal": [
        "Dashboard page rendered — 200 OK.",
        "Static asset bundle served from CDN.",
        "Merchant settings updated.",
        "CSV export started for merchant_id=M-5021.",
    ],
}

_NORMAL_WARNING_MESSAGES: Dict[str, List[str]] = {
    "_default": [
        "Response time approaching SLA threshold.",
        "Connection pool nearing capacity — 85 % utilised.",
        "Retry attempt 1/3 for downstream call.",
        "Garbage collection pause exceeded 200 ms.",
    ],
}

# ── failure-specific log templates ──────────────────────────────────

_FAILURE_LOG_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "memory_leak": {
        "WARNING": [
            "Heap usage at {mem}% — approaching threshold.",
            "GC overhead increasing — major collections every {gc_interval}s.",
            "Large object allocation detected — {alloc_mb} MB.",
            "Memory pool 'Old Gen' above 80 %.",
        ],
        "ERROR": [
            "OutOfMemoryError: Java heap space.",
            "OutOfMemoryError: GC overhead limit exceeded.",
            "Service crash — OOM killer invoked by kernel.",
            "Failed to allocate {alloc_mb} MB — heap exhausted.",
            "Container restarted — exit code 137 (OOMKilled).",
        ],
    },
    "cpu_spike": {
        "WARNING": [
            "CPU utilisation at {cpu}% — thread pool saturated.",
            "Request queue depth > 500 — processing delayed.",
            "High CPU usage detected — possible infinite loop.",
            "Worker threads blocked — context switch rate elevated.",
        ],
        "ERROR": [
            "Request timeout — CPU-bound task exceeded deadline.",
            "Watchdog: service health-check failed (high CPU).",
            "Thread starvation detected — 0 idle workers.",
            "Circuit breaker OPEN — downstream unresponsive.",
        ],
    },
    "database_timeout": {
        "WARNING": [
            "Database query took {latency}ms — exceeds threshold.",
            "Connection pool wait time elevated — {wait}ms.",
            "Replication lag: {lag}s — reads may be stale.",
            "Slow query log entry — SELECT on 'transactions' > 5 s.",
        ],
        "ERROR": [
            "Database connection timeout after 30 000 ms.",
            "java.sql.SQLTransientConnectionException: connection pool exhausted.",
            "Deadlock detected on table 'payments'.",
            "Database connection refused — max_connections reached.",
            "Transaction rolled back — lock wait timeout exceeded.",
        ],
    },
    "network_latency": {
        "WARNING": [
            "Upstream latency elevated — RTT {latency}ms.",
            "DNS resolution slow — {dns}ms.",
            "TCP retransmissions detected on outbound socket.",
            "TLS handshake timeout — retrying.",
        ],
        "ERROR": [
            "Connection reset by peer — downstream unreachable.",
            "Socket timeout: failed to read response within 10 s.",
            "java.net.SocketException: Connection timed out.",
            "Service unavailable — upstream returned HTTP 503.",
        ],
    },
}


# ── internal helpers ────────────────────────────────────────────────

def _pick(lst: List[str]) -> str:
    return random.choice(lst)


def _format_msg(template: str, step: int, total_steps: int) -> str:
    """Fill in placeholders with plausible values."""
    progress = step / max(total_steps - 1, 1)
    return template.format(
        txn=f"TXN-{random.randint(100000, 999999)}",
        mem=round(30 + 65 * progress),
        cpu=round(25 + 70 * progress),
        latency=round(80 + 4000 * progress),
        wait=round(50 + 2000 * progress),
        lag=round(0.3 + 10 * progress, 1),
        gc_interval=round(max(1, 15 - 13 * progress)),
        alloc_mb=random.randint(64, 512),
        dns=round(20 + 400 * progress),
    )


def _choose_level_normal() -> str:
    """Normal mode: ~85 % INFO, ~12 % WARNING, ~3 % ERROR."""
    r = random.random()
    if r < 0.85:
        return "INFO"
    elif r < 0.97:
        return "WARNING"
    return "ERROR"


def _choose_level_failure(progress: float) -> str:
    """Failure mode: ERROR frequency scales with progression."""
    # early in incident → mostly warnings; later → mostly errors
    error_prob = 0.10 + 0.55 * progress
    warn_prob = 0.30 + 0.10 * (1 - progress)
    r = random.random()
    if r < error_prob:
        return "ERROR"
    elif r < error_prob + warn_prob:
        return "WARNING"
    return "INFO"


# ── public API ──────────────────────────────────────────────────────

def generate_logs(
    services: Optional[List[str]] = None,
    duration_minutes: int = 30,
    interval_seconds: int = 30,
    failure_plan: Optional[Dict[str, str]] = None,
    start_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Generate structured log entries for each service over time.

    Parameters
    ----------
    services : list[str] | None
        Services to emit logs for.  Defaults to all known services.
    duration_minutes : int
        Observation window length.
    interval_seconds : int
        Average gap between log lines per service.
    failure_plan : dict[str, str] | None
        ``service_name → failure_profile`` mapping.
    start_time : datetime | None
        Timestamp of the first log line.  Defaults to *now* (UTC).

    Returns
    -------
    list[dict]
        Each dict has: timestamp, service, level, message.
    """
    services = services or get_service_names()
    failure_plan = failure_plan or {}
    start = start_time or datetime.now(timezone.utc)
    total_steps = max(duration_minutes * 60 // interval_seconds, 1)

    logs: List[Dict[str, Any]] = []

    for svc in services:
        profile = failure_plan.get(svc)
        templates = _FAILURE_LOG_TEMPLATES.get(profile) if profile else None

        for step in range(total_steps):
            # slight timestamp jitter so logs don't land on exact intervals
            jitter_sec = random.uniform(-interval_seconds * 0.3,
                                         interval_seconds * 0.3)
            ts = start + timedelta(seconds=step * interval_seconds + jitter_sec)
            progress = step / max(total_steps - 1, 1)

            if templates:
                level = _choose_level_failure(progress)
            else:
                level = _choose_level_normal()

            # pick a message
            if templates and level in templates:
                raw_msg = _pick(templates[level])
                message = _format_msg(raw_msg, step, total_steps)
            elif level == "INFO":
                svc_msgs = _NORMAL_INFO_MESSAGES.get(svc,
                            _NORMAL_INFO_MESSAGES.get("api-gateway"))
                message = _format_msg(_pick(svc_msgs), step, total_steps)
            elif level == "WARNING":
                message = _format_msg(
                    _pick(_NORMAL_WARNING_MESSAGES["_default"]),
                    step, total_steps,
                )
            else:
                # rare normal-mode error
                message = "Unexpected internal error — see stack trace."

            logs.append({
                "timestamp": ts.isoformat(),
                "service": svc,
                "level": level,
                "message": message,
            })

    # sort globally by timestamp for realism
    logs.sort(key=lambda l: l["timestamp"])
    return logs
