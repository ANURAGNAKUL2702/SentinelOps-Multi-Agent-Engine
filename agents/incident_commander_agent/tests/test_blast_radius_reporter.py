"""Tests for core/blast_radius_reporter.py — Algorithm 3."""

from __future__ import annotations

from agents.incident_commander_agent.core.blast_radius_reporter import (
    calculate_blast_radius,
)
from agents.incident_commander_agent.config import BlastRadiusConfig
from agents.root_cause_agent.schema import CausalLink


class TestBlastRadiusReporter:
    def test_basic_calculation(self):
        br = calculate_blast_radius(["svc-a", "svc-b"], total_known_services=10)
        assert br.affected_service_count == 2
        assert br.estimated_users == 10_000  # 2 * 5000
        assert 0.0 <= br.availability_impact <= 1.0

    def test_customer_facing_detection(self):
        br = calculate_blast_radius(["api-gateway", "backend"])
        assert br.is_customer_facing is True

    def test_non_customer_facing(self):
        br = calculate_blast_radius(["data-processor", "batch-worker"])
        assert br.is_customer_facing is False

    def test_causal_chain_adds_services(self):
        chain = [
            CausalLink(cause="svc-a", effect="svc-c", confidence=0.8),
        ]
        br = calculate_blast_radius(["svc-a", "svc-b"], chain, total_known_services=10)
        assert "svc-c" in br.affected_services
        assert br.affected_service_count == 3

    def test_custom_config(self):
        cfg = BlastRadiusConfig(avg_users_per_service=1000, revenue_per_minute_per_service=50.0)
        br = calculate_blast_radius(["svc-a"], config=cfg)
        assert br.estimated_users == 1000
        assert br.revenue_impact_per_minute == 50.0

    def test_empty_services(self):
        br = calculate_blast_radius([])
        assert br.affected_service_count == 0
        assert br.estimated_users == 0
        assert br.availability_impact == 0.0
