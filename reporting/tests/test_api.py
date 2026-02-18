"""Tests for reporting.api — FastAPI endpoints via TestClient."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reporting.api.dependencies import (
    get_config,
    get_db,
    get_report_builder,
    get_repository,
    init_dependencies,
    shutdown_dependencies,
)
from reporting.api.server import app
from reporting.config import ReportingConfig
from reporting.database.connection import DatabaseConnection
from reporting.database.repository import IncidentRepository
from reporting.report_builder import ReportBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_deps(tmp_path):
    """Initialise dependencies with an in-memory SQLite DB."""
    cfg = ReportingConfig(
        database_url="sqlite:///:memory:",
        output_dir=str(tmp_path / "reports"),
        include_visualizations=False,
    )
    init_dependencies(cfg)
    yield
    shutdown_dependencies()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_pipeline_result() -> dict:
    return {
        "correlation_id": str(uuid.uuid4()),
        "status": "success",
        "execution_time": 5.0,
        "agent_outputs": {
            "root_cause_output": {
                "root_cause": "Connection pool exhaustion",
                "confidence": 0.85,
                "severity": "high",
            },
            "validation_output": {"accuracy": 0.9},
            "incident_response": {
                "runbook": {"title": "Fix pool", "steps": ["Step 1"]},
                "action_items": [],
            },
            "dependency_output": {"affected_services": ["api-gw"]},
            "log_output": {},
        },
        "telemetry": {
            "total_llm_cost": 0.05,
            "total_tokens": 5000,
            "total_llm_calls": 10,
            "agent_latencies": {"log_agent": 1.0},
            "parallel_speedup": 1.5,
        },
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/api/v1/health/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert "version" in data
        assert "uptime" in data

    def test_health_has_database_field(self, client: TestClient) -> None:
        data = client.get("/api/v1/health/").json()
        assert data["database"] in ("connected", "disconnected")


# ---------------------------------------------------------------------------
# Reports endpoints
# ---------------------------------------------------------------------------

class TestReportsEndpoints:
    def test_generate_report(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        r = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["json"],
            "save_to_database": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert "report_id" in data
        assert data["correlation_id"] == sample_pipeline_result["correlation_id"]
        assert "json" in data["formats"]

    def test_generate_report_html(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        r = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["html"],
        })
        assert r.status_code == 200

    def test_generate_report_invalid_format(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        r = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["invalid_format"],
        })
        assert r.status_code == 422

    def test_get_report_metadata(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        gen = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["json"],
        })
        report_id = gen.json()["report_id"]
        r = client.get(f"/api/v1/reports/{report_id}")
        assert r.status_code == 200
        assert r.json()["report_id"] == report_id

    def test_get_report_not_found(self, client: TestClient) -> None:
        r = client.get(f"/api/v1/reports/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_download_report(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        gen = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["json"],
        })
        report_id = gen.json()["report_id"]
        r = client.get(f"/api/v1/reports/{report_id}/download/json")
        assert r.status_code == 200
        # Should be valid JSON
        json.loads(r.content)

    def test_download_report_not_found(self, client: TestClient) -> None:
        r = client.get(f"/api/v1/reports/{uuid.uuid4()}/download/html")
        assert r.status_code == 404

    def test_download_report_invalid_format(
        self, client: TestClient, sample_pipeline_result: dict,
    ) -> None:
        gen = client.post("/api/v1/reports/generate", json={
            "pipeline_result": sample_pipeline_result,
            "formats": ["json"],
        })
        report_id = gen.json()["report_id"]
        r = client.get(f"/api/v1/reports/{report_id}/download/xlsx")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Incidents endpoints
# ---------------------------------------------------------------------------

class TestIncidentsEndpoints:
    def _seed_incidents(self, client: TestClient) -> None:
        for i in range(3):
            client.post("/api/v1/reports/generate", json={
                "pipeline_result": {
                    "correlation_id": f"inc-{i}",
                    "status": "success",
                    "execution_time": float(i + 1),
                    "agent_outputs": {
                        "root_cause_output": {
                            "root_cause": "DB pool" if i < 2 else "OOM",
                            "confidence": 0.8,
                            "severity": "high" if i == 0 else "medium",
                        },
                    },
                    "telemetry": {"total_llm_cost": 0.01 * (i + 1)},
                },
                "formats": ["json"],
                "save_to_database": True,
            })

    def test_list_incidents(self, client: TestClient) -> None:
        self._seed_incidents(client)
        r = client.get("/api/v1/incidents/")
        assert r.status_code == 200
        data = r.json()
        assert "incidents" in data
        assert data["total"] >= 1

    def test_list_incidents_with_limit(self, client: TestClient) -> None:
        self._seed_incidents(client)
        r = client.get("/api/v1/incidents/?limit=1")
        assert r.status_code == 200
        assert r.json()["total"] <= 1

    def test_list_incidents_filter_severity(self, client: TestClient) -> None:
        self._seed_incidents(client)
        r = client.get("/api/v1/incidents/?severity=P1_HIGH")
        assert r.status_code == 200
        for inc in r.json()["incidents"]:
            assert inc["severity"] == "P1_HIGH"

    def test_list_incidents_filter_root_cause(self, client: TestClient) -> None:
        self._seed_incidents(client)
        r = client.get("/api/v1/incidents/?root_cause=DB")
        assert r.status_code == 200

    def test_get_incident_by_id(self, client: TestClient) -> None:
        self._seed_incidents(client)
        r = client.get("/api/v1/incidents/inc-0")
        assert r.status_code == 200
        assert r.json()["incident"]["correlation_id"] == "inc-0"

    def test_get_incident_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/incidents/nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

class TestAnalyticsEndpoints:
    def _seed(self, client: TestClient) -> None:
        for i in range(3):
            client.post("/api/v1/reports/generate", json={
                "pipeline_result": {
                    "correlation_id": f"analytics-{i}",
                    "status": "success",
                    "execution_time": 5.0,
                    "agent_outputs": {
                        "root_cause_output": {"root_cause": "test", "confidence": 0.8},
                    },
                    "telemetry": {"total_llm_cost": 0.01},
                },
                "formats": ["json"],
                "save_to_database": True,
            })

    def test_dashboard(self, client: TestClient) -> None:
        self._seed(client)
        r = client.get("/api/v1/analytics/dashboard?days=30")
        assert r.status_code == 200
        data = r.json()
        assert "total_incidents" in data
        assert "mttr" in data
        assert "slo_compliance" in data

    def test_trends(self, client: TestClient) -> None:
        self._seed(client)
        r = client.get("/api/v1/analytics/trends?metric=duration&days=30")
        assert r.status_code == 200
        data = r.json()
        assert "direction" in data
        assert "slope" in data

    def test_costs(self, client: TestClient) -> None:
        self._seed(client)
        r = client.get("/api/v1/analytics/costs?days=30")
        assert r.status_code == 200
        data = r.json()
        assert "total_cost" in data

    def test_insights(self, client: TestClient) -> None:
        self._seed(client)
        r = client.get("/api/v1/analytics/insights?days=30")
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data
        assert "based_on_incidents" in data


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_prometheus_metrics(self, client: TestClient) -> None:
        r = client.get("/api/v1/metrics/prometheus")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_body(self, client: TestClient) -> None:
        r = client.post("/api/v1/reports/generate")
        assert r.status_code == 422

    def test_invalid_json(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/reports/generate",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_invalid_limit(self, client: TestClient) -> None:
        r = client.get("/api/v1/incidents/?limit=0")
        assert r.status_code == 422

    def test_invalid_days(self, client: TestClient) -> None:
        r = client.get("/api/v1/analytics/dashboard?days=0")
        assert r.status_code == 422
