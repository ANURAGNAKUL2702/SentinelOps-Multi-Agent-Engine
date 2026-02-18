"""Tests for integration.config_manager — YAML + env var config loading."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from integration.config_manager import (
    ConfigManager,
    SystemConfig,
    SystemSettings,
    SimulationSettings,
    AgentSettings,
    OrchestratorSettings,
    ReportingSettings,
    APISettings,
    DatabaseSettings,
    LLMSettings,
    _deep_merge,
)


# ── SystemConfig defaults ──────────────────────────────────────────


class TestSystemConfigDefaults:
    """Ensure every section has sensible defaults."""

    def test_default_instantiation(self) -> None:
        cfg = SystemConfig()
        assert cfg.system.log_level == "INFO"
        assert cfg.system.version == "1.0.0"

    def test_default_scenarios(self) -> None:
        cfg = SystemConfig()
        assert "database_timeout" in cfg.simulation.available_scenarios
        assert "memory_leak" in cfg.simulation.available_scenarios
        assert "cpu_spike" in cfg.simulation.available_scenarios
        assert "network_latency" in cfg.simulation.available_scenarios
        assert len(cfg.simulation.available_scenarios) == 4

    def test_default_agent_timeouts(self) -> None:
        cfg = SystemConfig()
        assert cfg.agents.log_agent.timeout_seconds == 5.0
        assert cfg.agents.hypothesis_agent.timeout_seconds == 10.0
        assert cfg.agents.root_cause_agent.timeout_seconds == 10.0

    def test_default_orchestrator(self) -> None:
        cfg = SystemConfig()
        assert cfg.orchestrator.pipeline_timeout_seconds == 60.0
        assert cfg.orchestrator.max_retries == 2
        assert cfg.orchestrator.enable_parallel_execution is True

    def test_default_reporting(self) -> None:
        cfg = SystemConfig()
        assert "html" in cfg.reporting.default_formats
        assert cfg.reporting.database.enable is True

    def test_default_api(self) -> None:
        cfg = SystemConfig()
        assert cfg.api.enabled is False
        assert cfg.api.port == 8000

    def test_frozen(self) -> None:
        cfg = SystemConfig()
        with pytest.raises(Exception):
            cfg.system = SystemSettings(log_level="DEBUG")


# ── Validators ─────────────────────────────────────────────────────


class TestValidators:
    def test_log_level_validation_valid(self) -> None:
        s = SystemSettings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_log_level_validation_invalid(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            SystemSettings(log_level="TRACE")

    def test_log_level_case_insensitive(self) -> None:
        for lvl in ("info", "INFO", "Info"):
            assert SystemSettings(log_level=lvl).log_level == "INFO"


# ── ConfigManager.load ─────────────────────────────────────────────


class TestConfigManagerLoad:
    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        p = str(tmp_path / "nonexistent.yaml")
        cfg = ConfigManager.load(p)
        assert isinstance(cfg, SystemConfig)
        assert cfg.system.log_level == "INFO"

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            system:
              log_level: DEBUG
            simulation:
              duration_minutes: 10
        """)
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml_content, encoding="utf-8")

        cfg = ConfigManager.load(str(f))
        assert cfg.system.log_level == "DEBUG"
        assert cfg.simulation.duration_minutes == 10
        # defaults still present
        assert cfg.orchestrator.pipeline_timeout_seconds == 60.0

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        cfg = ConfigManager.load(str(f))
        assert isinstance(cfg, SystemConfig)

    def test_load_partial_config(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            reporting:
              output_directory: /tmp/reports
        """)
        f = tmp_path / "partial.yaml"
        f.write_text(yaml_content, encoding="utf-8")

        cfg = ConfigManager.load(str(f))
        assert cfg.reporting.output_directory == "/tmp/reports"
        # defaults
        assert cfg.system.log_level == "INFO"


# ── ConfigManager.validate ─────────────────────────────────────────


class TestConfigManagerValidate:
    def test_valid_config_no_issues(self) -> None:
        cfg = SystemConfig()
        issues = ConfigManager.validate(cfg)
        assert issues == []

    def test_empty_scenario_name(self) -> None:
        cfg = SystemConfig(
            simulation=SimulationSettings(available_scenarios=["ok", "  "]),
        )
        issues = ConfigManager.validate(cfg)
        assert any("empty" in i.lower() for i in issues)

    def test_negative_pipeline_timeout(self) -> None:
        cfg = SystemConfig(
            orchestrator=OrchestratorSettings(pipeline_timeout_seconds=-1),
        )
        issues = ConfigManager.validate(cfg)
        assert any("pipeline_timeout" in i for i in issues)


# ── ConfigManager.merge_env_vars ───────────────────────────────────


class TestMergeEnvVars:
    def test_env_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WARROOM_LOG_LEVEL", "WARNING")
        cfg = SystemConfig()
        merged = ConfigManager.merge_env_vars(cfg)
        assert merged.system.log_level == "WARNING"

    def test_env_override_api_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WARROOM_API_PORT", "9090")
        cfg = SystemConfig()
        merged = ConfigManager.merge_env_vars(cfg)
        assert merged.api.port == 9090

    def test_env_override_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WARROOM_DATABASE_URL", "postgresql://db")
        cfg = SystemConfig()
        merged = ConfigManager.merge_env_vars(cfg)
        assert merged.reporting.database.url == "postgresql://db"

    def test_env_override_api_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WARROOM_API_ENABLED", "true")
        cfg = SystemConfig()
        merged = ConfigManager.merge_env_vars(cfg)
        assert merged.api.enabled is True

    def test_no_env_vars_returns_same(self) -> None:
        cfg = SystemConfig()
        merged = ConfigManager.merge_env_vars(cfg)
        assert merged == cfg


# ── ConfigManager.save / get_default_config ────────────────────────


class TestConfigManagerSaveAndDefaults:
    def test_get_default_config(self) -> None:
        cfg = ConfigManager.get_default_config()
        assert isinstance(cfg, SystemConfig)

    def test_save_and_reload(self, tmp_path: Path) -> None:
        cfg = SystemConfig(
            system=SystemSettings(log_level="ERROR"),
        )
        out = str(tmp_path / "saved.yaml")
        ConfigManager.save(cfg, out)
        assert Path(out).exists()

        reloaded = ConfigManager.load(out)
        assert reloaded.system.log_level == "ERROR"


# ── _deep_merge helper ─────────────────────────────────────────────


class TestDeepMerge:
    def test_shallow_merge(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"x": {"y": 1, "z": 2}}
        _deep_merge(base, {"x": {"z": 99}})
        assert base["x"]["y"] == 1
        assert base["x"]["z"] == 99

    def test_add_nested_key(self) -> None:
        base = {"x": {"y": 1}}
        _deep_merge(base, {"x": {"new_key": "hello"}})
        assert base["x"]["new_key"] == "hello"
