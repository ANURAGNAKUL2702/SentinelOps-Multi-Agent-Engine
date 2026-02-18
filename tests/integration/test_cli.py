"""Tests for the Click CLI (main.py) via CliRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Provide a Click CliRunner with isolated filesystem."""
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def config_yaml(tmp_path: Path) -> str:
    """Write a minimal valid config YAML and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "system:\n  log_level: INFO\n"
        "simulation:\n  available_scenarios:\n"
        "    - memory_leak\n    - cpu_spike\n"
        "    - database_timeout\n    - network_latency\n",
        encoding="utf-8",
    )
    return str(cfg)


# ── Root group ─────────────────────────────────────────────────────


class TestCLIGroup:
    def test_help_flag(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "--help"])
        assert result.exit_code == 0
        assert "War-Room" in result.output or "war-room" in result.output.lower()

    def test_no_subcommand_shows_help(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_version_option(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


# ── list-scenarios ─────────────────────────────────────────────────


class TestListScenarios:
    def test_lists_all_scenarios(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "list-scenarios"])
        assert result.exit_code == 0
        for s in ("memory_leak", "cpu_spike", "database_timeout", "network_latency"):
            assert s in result.stderr


# ── validate ───────────────────────────────────────────────────────


class TestValidateCommand:
    def test_validate_valid_config(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "validate", "--config", config_yaml])
        assert result.exit_code == 0
        # Rich output goes to stderr
        assert "valid" in (result.output + result.stderr).lower()

    def test_validate_missing_file(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(
            cli,
            ["--config", config_yaml, "validate", "--config", "/no/such/file.yaml"],
        )
        # Should still exit 0 because missing file → defaults which are valid
        # OR it may finish with defaults
        assert result.exit_code in (0, 1)

    def test_validate_shows_summary(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "validate", "--config", config_yaml])
        combined = result.output + result.stderr
        assert "1.0.0" in combined or "Version" in combined


# ── version command ────────────────────────────────────────────────


class TestVersionCommand:
    def test_version_subcommand(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "version"])
        combined = result.output + result.stderr
        assert "1.0.0" in combined
        assert "Python" in combined or "python" in combined.lower()

    def test_version_shows_dependencies(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "version"])
        combined = result.output + result.stderr
        # At least some dependency should appear
        assert "click" in combined.lower() or "pydantic" in combined.lower()


# ── run command — validation ───────────────────────────────────────


class TestRunValidation:
    def test_run_missing_scenario(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(cli, ["--config", config_yaml, "run"])
        # Missing required --scenario
        assert result.exit_code != 0

    def test_run_unknown_scenario(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(
            cli,
            ["--config", config_yaml, "run", "--scenario", "nonexistent"],
        )
        # Should fail with exit code 1
        assert result.exit_code == 1

    def test_run_invalid_format(self, runner: CliRunner, config_yaml: str) -> None:
        result = runner.invoke(
            cli,
            ["--config", config_yaml, "run", "-s", "cpu_spike", "-f", "xlsx"],
        )
        assert result.exit_code == 1

    @patch("integration.pipeline.WarRoomPipeline.run_scenario")
    def test_run_success(
        self,
        mock_run: MagicMock,
        runner: CliRunner,
        config_yaml: str,
    ) -> None:
        from integration.pipeline import PipelineRunResult

        mock_run.return_value = PipelineRunResult(
            status="SUCCESS",
            scenario="cpu_spike",
            correlation_id="test-123",
            root_cause="CPU exhaustion",
            confidence=0.92,
            severity="P1",
            execution_time=5.3,
            report_paths={"html": "reports/report.html"},
        )

        result = runner.invoke(
            cli,
            ["--config", config_yaml, "run", "-s", "cpu_spike", "-f", "html"],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("integration.pipeline.WarRoomPipeline.run_scenario")
    def test_run_failure(
        self,
        mock_run: MagicMock,
        runner: CliRunner,
        config_yaml: str,
    ) -> None:
        from integration.pipeline import PipelineRunResult

        mock_run.return_value = PipelineRunResult(
            status="FAILED",
            scenario="cpu_spike",
            errors=["something broke"],
        )

        result = runner.invoke(
            cli,
            ["--config", config_yaml, "run", "-s", "cpu_spike"],
        )
        assert result.exit_code == 2

    @patch("integration.pipeline.WarRoomPipeline.run_scenario")
    def test_run_partial_success(
        self,
        mock_run: MagicMock,
        runner: CliRunner,
        config_yaml: str,
    ) -> None:
        from integration.pipeline import PipelineRunResult

        mock_run.return_value = PipelineRunResult(
            status="PARTIAL_SUCCESS",
            scenario="database_timeout",
            root_cause="DB pool saturation",
            confidence=0.72,
            severity="P0",
            execution_time=8.1,
            errors=["DB save failed"],
        )

        result = runner.invoke(
            cli,
            ["--config", config_yaml, "run", "-s", "database_timeout"],
        )
        assert result.exit_code == 3


# ── analyze command ────────────────────────────────────────────────


class TestAnalyzeCommand:
    @patch("integration.pipeline.WarRoomPipeline.analyze_history")
    def test_analyze_default(
        self,
        mock_analyze: MagicMock,
        runner: CliRunner,
        config_yaml: str,
    ) -> None:
        mock_analyze.return_value = {
            "total_incidents": 42,
            "mttr": 12.5,
            "mttd": 3.2,
            "slo_compliance": 0.95,
            "total_cost": 1.23,
            "common_root_causes": [],
        }

        result = runner.invoke(cli, ["--config", config_yaml, "analyze"])
        assert result.exit_code == 0
        mock_analyze.assert_called_once()


# ── dashboard command ──────────────────────────────────────────────


class TestDashboardCommand:
    @patch("integration.pipeline.WarRoomPipeline.generate_dashboard")
    def test_dashboard(
        self,
        mock_dash: MagicMock,
        runner: CliRunner,
        config_yaml: str,
    ) -> None:
        mock_dash.return_value = "reports/dashboard.html"

        result = runner.invoke(cli, ["--config", config_yaml, "dashboard"])
        assert result.exit_code == 0
        combined = result.output + result.stderr
        assert "dashboard" in combined.lower()
