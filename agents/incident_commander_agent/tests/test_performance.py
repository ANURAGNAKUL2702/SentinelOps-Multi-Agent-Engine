"""Performance tests — ensure pipeline meets SLA targets."""

from __future__ import annotations

import time

import pytest

from agents.incident_commander_agent.agent import IncidentCommanderAgent
from agents.incident_commander_agent.fallback import DeterministicFallback
from agents.incident_commander_agent.tests.conftest import (
    make_input,
    make_validation_report,
    make_verdict,
)


class TestPerformance:
    def test_full_pipeline_under_2s(self):
        agent = IncidentCommanderAgent()
        inp = make_input()

        start = time.perf_counter()
        response = agent.command(inp)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 2000, f"Pipeline took {elapsed:.1f}ms (>2s)"

    def test_fallback_under_100ms(self):
        fb = DeterministicFallback()
        verdict = make_verdict()
        report = make_validation_report()

        start = time.perf_counter()
        fb.execute(verdict, report)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Fallback took {elapsed:.1f}ms (>100ms)"

    def test_ten_iterations_under_2s(self):
        """10 back-to-back runs should each be well under 2s."""
        agent = IncidentCommanderAgent()
        inp = make_input()

        for _ in range(10):
            start = time.perf_counter()
            agent.command(inp)
            elapsed = (time.perf_counter() - start) * 1000
            assert elapsed < 2000
