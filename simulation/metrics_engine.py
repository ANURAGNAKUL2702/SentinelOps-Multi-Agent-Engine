"""
metrics_engine.py — Time-series health-metric generator.

Produces realistic CPU, memory, latency, and error-rate readings for
every service across a configurable time window.

Supports two behavioural modes that can be mixed per-service:
  • **normal**  — healthy baseline with gentle random jitter.
  • **failure** — degraded pattern driven by a specific failure profile
                  (memory leak, CPU spike, latency surge, etc.).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from simulation.services import get_service_names


# ── failure profile types ───────────────────────────────────────────
FAILURE_PROFILES = {
    "memory_leak": {
        "description": "Gradual memory climb until OOM",
        # memory ramps linearly; CPU stays slightly elevated; latency creeps up
    },
    "cpu_spike": {
        "description": "Sudden sustained CPU saturation",
    },
    "database_timeout": {
        "description": "Backend DB becomes unresponsive — latency + errors explode",
    },
    "network_latency": {
        "description": "Upstream network degradation — latency climbs, sporadic errors",
    },
}


# ── internal helpers ────────────────────────────────────────────────

def _jitter(base: float, pct: float = 0.05) -> float:
    """Add ±pct% random noise to *base*."""
    delta = base * pct
    return round(base + random.uniform(-delta, delta), 2)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, value)), 2)


# ── normal-mode generators ─────────────────────────────────────────

def _normal_cpu(step: int, total_steps: int) -> float:
    """Healthy CPU: 15-35 % with light noise."""
    return _clamp(_jitter(random.uniform(15, 35), 0.08), 0, 100)


def _normal_memory(step: int, total_steps: int) -> float:
    """Healthy memory: 40-55 % stable."""
    return _clamp(_jitter(random.uniform(40, 55), 0.05), 0, 100)


def _normal_latency(step: int, total_steps: int) -> float:
    """Healthy latency: 20-80 ms."""
    return round(max(1, _jitter(random.uniform(20, 80), 0.10)), 2)


def _normal_error_rate(step: int, total_steps: int) -> float:
    """Healthy error rate: 0.0-0.5 %."""
    return _clamp(round(random.uniform(0.0, 0.5), 2), 0, 100)


# ── failure-mode generators ────────────────────────────────────────

def _memory_leak_metrics(step: int, total_steps: int) -> Dict[str, float]:
    """Memory climbs linearly from ~30 % to ~95 %; CPU & latency rise gently.

    DISTINCTIVE SIGNALS (unique to memory leaks):
      - heap_used_mb: monotonic growth 100 → 4000 MB
      - gc_overhead_percent: escalating GC overhead 2 → 80 %
      - gc_pause_ms: increasing pause times 10 → 5000 ms
    NON-DISTINCTIVE (kept normal to distinguish from other failures):
      - cpu_percent: moderate 25-55 % (NOT 100 % like cpu_spike)
      - db_query_duration_ms: normal ~10 ms (NOT spiking like db_timeout)
      - packet_loss_percent: 0 % (NOT elevated like network_latency)
    """
    progress = step / max(total_steps - 1, 1)
    memory = _clamp(_jitter(30 + 65 * progress, 0.03), 0, 100)
    cpu = _clamp(_jitter(25 + 30 * progress, 0.06), 0, 100)
    latency = round(max(1, _jitter(40 + 350 * progress, 0.08)), 2)
    error_rate = _clamp(round(0.1 + 8 * (progress ** 2), 2), 0, 100)
    # ── DISTINCTIVE: memory-leak-specific metrics ───────────────
    gc_overhead = _clamp(_jitter(2 + 78 * (progress ** 1.5), 0.05), 0, 100)
    heap_used_mb = round(_jitter(100 + 3900 * progress, 0.03), 0)
    gc_pause_ms = round(max(1, _jitter(10 + 4990 * (progress ** 2), 0.05)), 2)
    return {
        "cpu_percent": cpu, "memory_percent": memory,
        "latency_ms": latency, "error_rate": error_rate,
        "gc_overhead_percent": gc_overhead,
        "heap_used_mb": heap_used_mb,
        "gc_pause_ms": gc_pause_ms,
        "db_query_duration_ms": round(_jitter(10, 0.1), 2),
        "db_active_connections": round(_jitter(10, 0.15)),
        "db_pool_wait_ms": round(_jitter(2, 0.2), 2),
        "packet_loss_percent": 0.0,
        "tcp_retransmissions": 0,
        "thread_pool_active_pct": _clamp(_jitter(40 + 20 * progress, 0.05), 0, 100),
    }


def _cpu_spike_metrics(step: int, total_steps: int) -> Dict[str, float]:
    """CPU jumps to 85-99 % early and stays there.

    DISTINCTIVE SIGNALS (unique to CPU spikes):
      - cpu_percent: sudden jump 20-35 → 88-99 % at ~20 % progress
      - thread_pool_active_pct: maxed out 30 → 100 %
    NON-DISTINCTIVE (kept normal to distinguish from other failures):
      - heap_used_mb: STABLE ~200 MB (NOT growing like memory_leak)
      - gc_overhead_percent: STABLE ~3 % (NOT escalating like memory_leak)
      - db_query_duration_ms: STABLE ~10 ms (NOT spiking like db_timeout)
      - packet_loss_percent: 0 % (NOT elevated like network_latency)
    """
    progress = step / max(total_steps - 1, 1)
    # spike kicks in at ~20 % of the window
    if progress < 0.2:
        cpu = _clamp(_jitter(random.uniform(20, 35), 0.06), 0, 100)
        thread_pool = _clamp(_jitter(30 + 10 * (progress / 0.2), 0.05), 0, 100)
    else:
        cpu = _clamp(_jitter(random.uniform(88, 99), 0.02), 0, 100)
        thread_pool = _clamp(_jitter(95 + 5 * ((progress - 0.2) / 0.8), 0.02), 0, 100)
    memory = _clamp(_jitter(55 + 15 * progress, 0.04), 0, 100)
    latency = round(max(1, _jitter(50 + 250 * progress, 0.08)), 2)
    error_rate = _clamp(round(0.2 + 6 * progress, 2), 0, 100)
    return {
        "cpu_percent": cpu, "memory_percent": memory,
        "latency_ms": latency, "error_rate": error_rate,
        "gc_overhead_percent": _clamp(_jitter(3, 0.15), 0, 100),
        "heap_used_mb": round(_jitter(200, 0.05), 0),
        "gc_pause_ms": round(_jitter(12, 0.1), 2),
        "db_query_duration_ms": round(_jitter(10, 0.1), 2),
        "db_active_connections": round(_jitter(10, 0.15)),
        "db_pool_wait_ms": round(_jitter(2, 0.2), 2),
        "packet_loss_percent": 0.0,
        "tcp_retransmissions": 0,
        "thread_pool_active_pct": thread_pool,
    }


def _db_timeout_metrics(step: int, total_steps: int) -> Dict[str, float]:
    """Latency spikes massively; error rate climbs fast.

    DISTINCTIVE SIGNALS (unique to database timeouts):
      - db_query_duration_ms: explosion 10 → 10000 ms
      - db_active_connections: pool exhaustion 10 → 100
      - db_pool_wait_ms: long waits 2 → 5000 ms
    NON-DISTINCTIVE (kept normal to distinguish from other failures):
      - cpu_percent: moderate 30-50 % (app threads waiting, NOT computing)
      - heap_used_mb: STABLE ~200 MB (NOT growing like memory_leak)
      - packet_loss_percent: 0 % (NOT elevated like network_latency)
    """
    progress = step / max(total_steps - 1, 1)
    cpu = _clamp(_jitter(random.uniform(30, 50), 0.06), 0, 100)
    memory = _clamp(_jitter(random.uniform(45, 60), 0.04), 0, 100)
    latency = round(max(1, _jitter(80 + 4000 * (progress ** 1.5), 0.06)), 2)
    error_rate = _clamp(round(0.5 + 25 * progress, 2), 0, 100)
    # ── DISTINCTIVE: database-specific metrics ──────────────────
    db_query = round(max(1, _jitter(10 + 9990 * (progress ** 1.5), 0.04)), 2)
    db_conns = round(min(100, _jitter(10 + 90 * progress, 0.05)))
    db_wait = round(max(0, _jitter(2 + 4998 * (progress ** 1.8), 0.04)), 2)
    return {
        "cpu_percent": cpu, "memory_percent": memory,
        "latency_ms": latency, "error_rate": error_rate,
        "gc_overhead_percent": _clamp(_jitter(3, 0.15), 0, 100),
        "heap_used_mb": round(_jitter(200, 0.05), 0),
        "gc_pause_ms": round(_jitter(12, 0.1), 2),
        "db_query_duration_ms": db_query,
        "db_active_connections": db_conns,
        "db_pool_wait_ms": db_wait,
        "packet_loss_percent": 0.0,
        "tcp_retransmissions": 0,
        "thread_pool_active_pct": _clamp(_jitter(40 + 50 * progress, 0.05), 0, 100),
    }


def _network_latency_metrics(step: int, total_steps: int) -> Dict[str, float]:
    """Latency climbs steadily; intermittent errors.

    DISTINCTIVE SIGNALS (unique to network issues):
      - packet_loss_percent: increasing 0 → 20 %
      - tcp_retransmissions: spike 0 → 500 /interval
    NON-DISTINCTIVE (kept normal to distinguish from other failures):
      - cpu_percent: STABLE 20-40 % (NOT spiking like cpu_spike)
      - heap_used_mb: STABLE ~200 MB (NOT growing like memory_leak)
      - db_query_duration_ms: STABLE ~10 ms (NOT spiking like db_timeout)
    """
    progress = step / max(total_steps - 1, 1)
    cpu = _clamp(_jitter(random.uniform(20, 40), 0.06), 0, 100)
    memory = _clamp(_jitter(random.uniform(40, 55), 0.04), 0, 100)
    latency = round(max(1, _jitter(60 + 800 * progress, 0.10)), 2)
    error_rate = _clamp(round(0.3 + 5 * progress + random.uniform(0, 2), 2), 0, 100)
    # ── DISTINCTIVE: network-specific metrics ───────────────────
    pkt_loss = round(_jitter(0.0 + 20 * (progress ** 1.3), 0.08), 2)
    tcp_retrans = round(max(0, _jitter(0 + 500 * (progress ** 1.2), 0.10)))
    return {
        "cpu_percent": cpu, "memory_percent": memory,
        "latency_ms": latency, "error_rate": error_rate,
        "gc_overhead_percent": _clamp(_jitter(3, 0.15), 0, 100),
        "heap_used_mb": round(_jitter(200, 0.05), 0),
        "gc_pause_ms": round(_jitter(12, 0.1), 2),
        "db_query_duration_ms": round(_jitter(10, 0.1), 2),
        "db_active_connections": round(_jitter(10, 0.15)),
        "db_pool_wait_ms": round(_jitter(2, 0.2), 2),
        "packet_loss_percent": max(0.0, pkt_loss),
        "tcp_retransmissions": max(0, tcp_retrans),
        "thread_pool_active_pct": _clamp(_jitter(30 + 15 * progress, 0.05), 0, 100),
    }


_FAILURE_GENERATORS = {
    "memory_leak":      _memory_leak_metrics,
    "cpu_spike":        _cpu_spike_metrics,
    "database_timeout": _db_timeout_metrics,
    "network_latency":  _network_latency_metrics,
}


# ── public API ──────────────────────────────────────────────────────

def generate_metrics(
    services: Optional[List[str]] = None,
    duration_minutes: int = 30,
    interval_seconds: int = 60,
    failure_plan: Optional[Dict[str, str]] = None,
    start_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Generate a time-series of metrics for each service.

    Parameters
    ----------
    services : list[str] | None
        Service names to cover.  Defaults to all known services.
    duration_minutes : int
        Length of the simulated observation window.
    interval_seconds : int
        Gap between data points.
    failure_plan : dict[str, str] | None
        Mapping of ``service_name → failure_profile`` for services that
        should exhibit degraded behaviour.  Services not in this map
        get normal-mode metrics.
    start_time : datetime | None
        Timestamp of the first data point.  Defaults to *now* (UTC).

    Returns
    -------
    list[dict]
        One dict per (service, timestamp) pair.
    """
    services = services or get_service_names()
    failure_plan = failure_plan or {}
    start = start_time or datetime.now(timezone.utc)
    total_steps = max(duration_minutes * 60 // interval_seconds, 1)

    records: List[Dict[str, Any]] = []

    for svc in services:
        profile = failure_plan.get(svc)
        gen_fn = _FAILURE_GENERATORS.get(profile) if profile else None

        for step in range(total_steps):
            ts = start + timedelta(seconds=step * interval_seconds)

            if gen_fn:
                vals = gen_fn(step, total_steps)
            else:
                vals = {
                    "cpu_percent": _normal_cpu(step, total_steps),
                    "memory_percent": _normal_memory(step, total_steps),
                    "latency_ms": _normal_latency(step, total_steps),
                    "error_rate": _normal_error_rate(step, total_steps),
                    # Baseline values for scenario-specific metrics
                    "gc_overhead_percent": _clamp(_jitter(2.0, 0.2), 0, 100),
                    "heap_used_mb": round(_jitter(200, 0.08), 0),
                    "gc_pause_ms": round(_jitter(10, 0.15), 2),
                    "db_query_duration_ms": round(_jitter(10, 0.1), 2),
                    "db_active_connections": round(_jitter(10, 0.15)),
                    "db_pool_wait_ms": round(_jitter(2, 0.2), 2),
                    "packet_loss_percent": 0.0,
                    "tcp_retransmissions": 0,
                    "thread_pool_active_pct": _clamp(_jitter(30, 0.1), 0, 100),
                }

            records.append({
                "timestamp": ts.isoformat(),
                "service": svc,
                **vals,
            })

    return records
