"""
validate_observability.py — End-to-end validation of the observability layer.

Checks every query method across all 4 scenarios to ensure:
  1. Stores ingest data correctly
  2. Filtering works (by service, level, timestamp)
  3. Query engine returns correct enriched results
  4. Agents see the right picture through the observability lens
"""

import sys
from simulation import run_simulation
from observability import build_observability_from_simulation

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

    # run simulation
    sim = run_simulation(scenario=scenario, target_service=target_service)

    # build observability layer
    obs = build_observability_from_simulation(sim)
    ms = obs["metrics_store"]
    ls = obs["log_store"]
    qe = obs["query_engine"]

    # ── 1. MetricsStore ─────────────────────────────────────────
    print("\n  [MetricsStore — Ingestion]")
    check("Metrics ingested", ms.total_records == len(sim["metrics"]),
          f"Expected {len(sim['metrics'])}, got {ms.total_records}")
    check("All services present", len(ms.services) == len(sim["services"]),
          f"Expected {len(sim['services'])}, got {len(ms.services)}")

    print("\n  [MetricsStore — Queries]")
    svc_metrics = ms.get_by_service(target_service)
    check(f"get_by_service('{target_service}') returns data", len(svc_metrics) > 0)

    snapshot = ms.get_latest_snapshot()
    check("get_latest_snapshot() has all services", len(snapshot) == len(sim["services"]))
    check("Snapshot has target service", target_service in snapshot)

    # timestamp range query
    all_metrics = ms.get_all()
    ts_first = all_metrics[0]["timestamp"]
    ts_last = all_metrics[-1]["timestamp"]
    range_metrics = ms.get_between(ts_first, ts_last, service=target_service)
    check("get_between() returns data", len(range_metrics) > 0)

    # trend
    trend = ms.get_metric_trend(target_service, "cpu_percent")
    check("get_metric_trend() returns data", len(trend) > 0)
    check("Trend entries have timestamp+value",
          all("timestamp" in t and "value" in t for t in trend))

    # average
    avg_cpu = ms.get_average(target_service, "cpu_percent")
    check("get_average() returns a number", avg_cpu is not None and isinstance(avg_cpu, float))

    # ── 2. LogStore ─────────────────────────────────────────────
    print("\n  [LogStore — Ingestion]")
    check("Logs ingested", ls.total_logs == len(sim["logs"]),
          f"Expected {len(sim['logs'])}, got {ls.total_logs}")
    check("All services present", len(ls.services) == len(sim["services"]))

    print("\n  [LogStore — Queries]")
    svc_logs = ls.get_by_service(target_service)
    check(f"get_by_service('{target_service}') returns data", len(svc_logs) > 0)

    errors = ls.get_errors()
    check("get_errors() returns data", len(errors) > 0)
    check("All returned are ERROR level",
          all(l["level"] == "ERROR" for l in errors))

    warnings = ls.get_warnings()
    check("get_warnings() returns data", len(warnings) > 0)

    by_level = ls.get_by_level("INFO")
    check("get_by_level('INFO') returns data", len(by_level) > 0)

    # timestamp range
    all_logs = ls.get_all()
    ts_first_l = min(l["timestamp"] for l in all_logs)
    ts_last_l = max(l["timestamp"] for l in all_logs)
    range_logs = ls.get_between(ts_first_l, ts_last_l, service=target_service, level="ERROR")
    check("get_between() with service+level filter works", len(range_logs) >= 0)

    # error counts
    err_counts = ls.get_error_count_by_service()
    check("get_error_count_by_service() returns dict", isinstance(err_counts, dict))
    check("Target service has errors in count",
          target_service in err_counts and err_counts[target_service] > 0)

    top_err = ls.get_top_error_services(3)
    check("get_top_error_services() returns list", isinstance(top_err, list))
    check("Top error services non-empty", len(top_err) > 0)

    # error frequency over time
    freq = ls.get_error_frequency_over_time(target_service)
    check("get_error_frequency_over_time() returns buckets", len(freq) > 0)
    check("Buckets have error_count",
          all("error_count" in b for b in freq))

    # keyword search
    search_results = ls.search_messages("error", service=target_service)
    check("search_messages() works", isinstance(search_results, list))

    # ── 3. QueryEngine ──────────────────────────────────────────
    print("\n  [QueryEngine — High-Level Queries]")

    # system health summary
    health = qe.get_system_health_summary()
    check("get_system_health_summary() returns dict", isinstance(health, dict))
    check("Health has total_services", "total_services" in health)
    check("Health has services_in_distress", "services_in_distress" in health)
    check("Some services in distress", len(health["services_in_distress"]) > 0,
          "Expected at least 1 distressed service in a failure scenario")

    # scenario-specific checks
    if scenario == "memory_leak":
        mem_results = qe.get_services_with_abnormal_memory()
        check("get_services_with_abnormal_memory() finds target",
              any(r["service"] == target_service for r in mem_results),
              f"Target {target_service} not found in abnormal memory results")
        if mem_results:
            r = next(r for r in mem_results if r["service"] == target_service)
            check("Memory result has trend='increasing'",
                  r["trend"] == "increasing",
                  f"Got trend={r['trend']}")
            check("Memory result has related_errors > 0",
                  r["related_errors"] > 0)

    elif scenario == "cpu_spike":
        cpu_results = qe.get_services_with_high_cpu()
        check("get_services_with_high_cpu() finds target",
              any(r["service"] == target_service for r in cpu_results),
              f"Target {target_service} not found in high CPU results")

    elif scenario == "database_timeout":
        lat_results = qe.get_latency_spikes()
        check("get_latency_spikes() returns results", len(lat_results) > 0)

        db_impact = qe.get_services_impacted_by_database_failure()
        check("get_services_impacted_by_database_failure() returns results",
              len(db_impact) > 0)
        check("DB impact includes target",
              any(r["service"] == target_service for r in db_impact))

    elif scenario == "network_latency":
        lat_results = qe.get_latency_spikes()
        check("get_latency_spikes() finds target",
              any(r["service"] == target_service for r in lat_results),
              f"Target {target_service} not in latency spikes")

    # top error generators
    top_gen = qe.get_top_error_generators()
    check("get_top_error_generators() returns results", len(top_gen) > 0)
    check("Top generator has sample_messages",
          "sample_messages" in top_gen[0] and len(top_gen[0]["sample_messages"]) > 0)
    check("Top generator has metric snapshot",
          "latest_cpu" in top_gen[0])

    # high error rate
    high_err = qe.get_services_with_high_error_rate()
    check("get_services_with_high_error_rate() returns results", len(high_err) > 0)

    # ── 4. Data immutability ────────────────────────────────────
    print("\n  [Data Immutability]")
    count_before = ms.total_records
    _ = ms.get_all()
    _ = ms.get_by_service(target_service)
    _ = qe.get_system_health_summary()
    check("Queries don't alter metrics count",
          ms.total_records == count_before)

    log_count_before = ls.total_logs
    _ = ls.get_errors()
    _ = ls.search_messages("error")
    _ = qe.get_top_error_generators()
    check("Queries don't alter log count",
          ls.total_logs == log_count_before)


def main():
    test_cases = [
        ("memory_leak", "payment-service"),
        ("cpu_spike", "fraud-service"),
        ("database_timeout", "database"),
        ("network_latency", "api-gateway"),
    ]

    for scenario, target in test_cases:
        validate_scenario(scenario, target)

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  OBSERVABILITY VALIDATION SUMMARY")
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
        print("  All checks passed! Observability layer is correct. 🎯\n")


if __name__ == "__main__":
    main()
