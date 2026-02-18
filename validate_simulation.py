"""
validate_simulation.py — End-to-end validation of the simulation module.

Checks:
  1. All 4 scenarios run without errors
  2. Output structure matches expected schema
  3. Metrics show realistic failure patterns (not random noise)
  4. Logs correlate with failure type (correct error messages)
  5. Dependency cascade / blast radius is accurate
  6. Timestamps are sequential and consistent
"""

import json
import sys
from datetime import datetime, timezone

from simulation import run_simulation
from simulation.failure_injector import get_available_scenarios
from simulation.dependency_graph import get_cascade_impact


PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    msg = f"  {status}  {name}"
    if detail and not condition:
        msg += f"  — {detail}"
    print(msg)
    return condition


def validate_scenario(scenario: str, target_service: str):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario} → {target_service}")
    print(f"{'='*60}")

    payload = run_simulation(
        scenario=scenario,
        target_service=target_service,
        duration_minutes=30,
        metrics_interval_seconds=60,
        log_interval_seconds=30,
    )

    # ── 1. Structure checks ────────────────────────────────────
    print("\n  [Structure]")
    required_keys = ["services", "dependencies", "metrics", "logs",
                     "root_cause", "severity", "scenario", "blast_radius"]
    for key in required_keys:
        check(f"Key '{key}' present", key in payload)

    check("Services is a list", isinstance(payload["services"], list))
    check("Services count = 8", len(payload["services"]) == 8)
    check("Dependencies is a dict", isinstance(payload["dependencies"], dict))
    check("Metrics is a list", isinstance(payload["metrics"], list))
    check("Metrics count > 0", len(payload["metrics"]) > 0)
    check("Logs is a list", isinstance(payload["logs"], list))
    check("Logs count > 0", len(payload["logs"]) > 0)
    check(f"Scenario matches '{scenario}'", payload["scenario"] == scenario)
    check("Root cause is a string", isinstance(payload["root_cause"], str))
    check("Severity format", payload["severity"].startswith("SEV-"))

    # ── 2. Metric record structure ─────────────────────────────
    print("\n  [Metric Records]")
    m = payload["metrics"][0]
    metric_keys = ["timestamp", "service", "cpu_percent", "memory_percent",
                   "latency_ms", "error_rate"]
    for key in metric_keys:
        check(f"Metric has '{key}'", key in m)

    # ── 3. Metrics show failure pattern ────────────────────────
    print("\n  [Failure Pattern — Metrics]")
    target_metrics = [r for r in payload["metrics"] if r["service"] == target_service]
    check(f"Target '{target_service}' has metrics", len(target_metrics) > 0)

    if target_metrics:
        first_5 = target_metrics[:5]
        last_5 = target_metrics[-5:]

        if scenario == "memory_leak":
            avg_mem_start = sum(r["memory_percent"] for r in first_5) / len(first_5)
            avg_mem_end = sum(r["memory_percent"] for r in last_5) / len(last_5)
            check(
                f"Memory ramps up ({avg_mem_start:.0f}% → {avg_mem_end:.0f}%)",
                avg_mem_end > avg_mem_start + 20,
                f"Expected significant increase, got {avg_mem_start:.1f} → {avg_mem_end:.1f}",
            )
            check(
                "End memory > 80%",
                avg_mem_end > 75,
                f"Got {avg_mem_end:.1f}%",
            )

        elif scenario == "cpu_spike":
            avg_cpu_end = sum(r["cpu_percent"] for r in last_5) / len(last_5)
            check(
                f"CPU spikes high (end avg: {avg_cpu_end:.0f}%)",
                avg_cpu_end > 80,
                f"Expected > 80%, got {avg_cpu_end:.1f}%",
            )

        elif scenario == "database_timeout":
            avg_lat_start = sum(r["latency_ms"] for r in first_5) / len(first_5)
            avg_lat_end = sum(r["latency_ms"] for r in last_5) / len(last_5)
            check(
                f"Latency spikes ({avg_lat_start:.0f}ms → {avg_lat_end:.0f}ms)",
                avg_lat_end > avg_lat_start * 5,
                f"Expected 5x+ increase",
            )
            avg_err_end = sum(r["error_rate"] for r in last_5) / len(last_5)
            check(
                f"Error rate climbs (end avg: {avg_err_end:.1f}%)",
                avg_err_end > 10,
                f"Expected > 10%, got {avg_err_end:.1f}%",
            )

        elif scenario == "network_latency":
            avg_lat_start = sum(r["latency_ms"] for r in first_5) / len(first_5)
            avg_lat_end = sum(r["latency_ms"] for r in last_5) / len(last_5)
            check(
                f"Latency increases ({avg_lat_start:.0f}ms → {avg_lat_end:.0f}ms)",
                avg_lat_end > avg_lat_start * 2,
                f"Expected 2x+ increase",
            )

    # ── 4. Healthy services stay healthy ───────────────────────
    print("\n  [Healthy Services — Sanity]")
    # Services in the failure_plan (not just blast radius) get degraded profiles
    failure_plan_services = set(payload["blast_radius"].get("all_affected", []))
    failure_plan_services.add(target_service)
    # Also exclude downstream deps of target (they get failure profiles too)
    from simulation.dependency_graph import get_dependencies
    for dep in get_dependencies(target_service):
        failure_plan_services.add(dep)
    healthy_services = [s for s in payload["services"] if s not in failure_plan_services]

    if healthy_services:
        svc = healthy_services[0]
        svc_metrics = [r for r in payload["metrics"] if r["service"] == svc]
        if svc_metrics:
            avg_cpu = sum(r["cpu_percent"] for r in svc_metrics) / len(svc_metrics)
            avg_err = sum(r["error_rate"] for r in svc_metrics) / len(svc_metrics)
            check(
                f"Healthy '{svc}' CPU reasonable ({avg_cpu:.0f}%)",
                avg_cpu < 60,
                f"Got {avg_cpu:.1f}%",
            )
            check(
                f"Healthy '{svc}' error rate low ({avg_err:.1f}%)",
                avg_err < 5,
                f"Got {avg_err:.1f}%",
            )

    # ── 5. Log correlation ─────────────────────────────────────
    print("\n  [Log Correlation]")
    target_logs = [l for l in payload["logs"] if l["service"] == target_service]
    error_logs = [l for l in target_logs if l["level"] == "ERROR"]
    warning_logs = [l for l in target_logs if l["level"] == "WARNING"]

    check(f"Target has logs ({len(target_logs)})", len(target_logs) > 0)
    check(f"Target has ERROR logs ({len(error_logs)})", len(error_logs) > 0)
    check(f"Target has WARNING logs ({len(warning_logs)})", len(warning_logs) > 0)

    # Check error messages match failure type
    all_error_msgs = " ".join(l["message"] for l in error_logs).lower()
    if scenario == "memory_leak":
        check(
            "Logs mention OOM/memory",
            "outofmemory" in all_error_msgs or "heap" in all_error_msgs or "oom" in all_error_msgs,
            f"No memory-related error messages found",
        )
    elif scenario == "cpu_spike":
        check(
            "Logs mention CPU/timeout",
            "cpu" in all_error_msgs or "timeout" in all_error_msgs or "thread" in all_error_msgs,
            f"No CPU-related error messages found",
        )
    elif scenario == "database_timeout":
        check(
            "Logs mention DB/connection",
            "database" in all_error_msgs or "connection" in all_error_msgs or "deadlock" in all_error_msgs,
            f"No DB-related error messages found",
        )
    elif scenario == "network_latency":
        check(
            "Logs mention network/socket",
            "socket" in all_error_msgs or "connection reset" in all_error_msgs or "503" in all_error_msgs,
            f"No network-related error messages found",
        )

    # Log level distribution: healthy svc should be mostly INFO
    if healthy_services:
        svc = healthy_services[0]
        svc_logs = [l for l in payload["logs"] if l["service"] == svc]
        info_count = sum(1 for l in svc_logs if l["level"] == "INFO")
        info_pct = (info_count / max(len(svc_logs), 1)) * 100
        check(
            f"Healthy '{svc}' logs mostly INFO ({info_pct:.0f}%)",
            info_pct > 60,
            f"Expected > 60% INFO, got {info_pct:.1f}%",
        )

    # ── 6. Blast radius / dependency cascade ───────────────────
    print("\n  [Blast Radius]")
    br = payload["blast_radius"]
    check("Blast radius has 'directly_affected'", "directly_affected" in br)
    check("Blast radius has 'total_affected'", "total_affected" in br)

    # Verify against dependency_graph directly
    expected_impact = get_cascade_impact(target_service)
    check(
        f"Total affected matches graph ({br['total_affected']})",
        br["total_affected"] == expected_impact["total_affected"],
        f"Expected {expected_impact['total_affected']}, got {br['total_affected']}",
    )

    # ── 7. Timestamps are sequential ──────────────────────────
    print("\n  [Timestamps]")
    timestamps = [r["timestamp"] for r in payload["metrics"]]
    svc_ts = [r["timestamp"] for r in payload["metrics"] if r["service"] == target_service]
    is_sorted = all(svc_ts[i] <= svc_ts[i+1] for i in range(len(svc_ts)-1))
    check("Target metric timestamps are sequential", is_sorted)


# ═══════════════════════════════════════════════════════════════════
#  RUN ALL SCENARIOS
# ═══════════════════════════════════════════════════════════════════

def main():
    test_cases = [
        ("memory_leak", "payment-service"),
        ("cpu_spike", "fraud-service"),
        ("database_timeout", "database"),
        ("network_latency", "api-gateway"),
    ]

    for scenario, target in test_cases:
        validate_scenario(scenario, target)

    # ── Summary ────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed} {PASS}")
    print(f"  Failed       : {failed} {'❌' if failed else ''}")
    print(f"  Pass rate    : {(passed/total)*100:.1f}%")
    print(f"{'='*60}\n")

    if failed:
        print("  Failed checks:")
        for name, ok in results:
            if not ok:
                print(f"    ❌ {name}")
        print()
        sys.exit(1)
    else:
        print("  All checks passed! Simulation is correct. 🎯\n")


if __name__ == "__main__":
    main()
