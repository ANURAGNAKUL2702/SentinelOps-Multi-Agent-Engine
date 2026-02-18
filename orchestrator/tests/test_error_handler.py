"""Tests for orchestrator.error_handler."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig
from orchestrator.error_handler import ErrorHandler
from orchestrator.schema import CircuitBreakerOpenError


class TestCategorizeError:
    def test_timeout(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        assert eh.categorize_error(asyncio.TimeoutError()) == "TIMEOUT"

    def test_validation_error(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        try:
            from orchestrator.schema import StageResult
            StageResult(stage_name="s", agents=[], duration=-1, status="X",
                        start_time="x", end_time="x")
        except ValidationError as ve:
            assert eh.categorize_error(ve) == "VALIDATION_ERROR"

    def test_circuit_open(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        assert eh.categorize_error(CircuitBreakerOpenError("a")) == "CIRCUIT_OPEN"

    def test_llm_error(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        assert eh.categorize_error(RuntimeError("Groq API rate_limit")) == "LLM_ERROR"

    def test_unknown(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        assert eh.categorize_error(RuntimeError("something random")) == "UNKNOWN"


class TestShouldAbort:
    def test_fail_fast_always_aborts(self) -> None:
        eh = ErrorHandler(OrchestratorConfig(fail_fast=True))
        assert eh.should_abort("log_agent") is True

    def test_non_critical_continues(self) -> None:
        eh = ErrorHandler(OrchestratorConfig(fail_fast=False))
        assert eh.should_abort("log_agent") is False
        assert eh.should_abort("metrics_agent") is False
        assert eh.should_abort("validation_agent") is False

    def test_critical_agent_aborts(self) -> None:
        eh = ErrorHandler(OrchestratorConfig(fail_fast=False))
        assert eh.should_abort("hypothesis_agent") is True
        assert eh.should_abort("root_cause_agent") is True


class TestIsCriticalAgent:
    def test_hypothesis_critical(self) -> None:
        assert ErrorHandler.is_critical_agent("hypothesis_agent") is True

    def test_root_cause_critical(self) -> None:
        assert ErrorHandler.is_critical_agent("root_cause_agent") is True

    def test_log_agent_not_critical(self) -> None:
        assert ErrorHandler.is_critical_agent("log_agent") is False


class TestTrackErrors:
    def test_handle_and_get(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        eh.handle_agent_error("log_agent", RuntimeError("boom"), stage="stage_1")
        eh.handle_agent_error("metrics_agent", asyncio.TimeoutError(), stage="stage_1")
        errors = eh.get_errors()
        assert len(errors) == 2
        assert errors[0].agent_name == "log_agent"
        assert errors[0].error_type == "UNKNOWN"
        assert errors[1].error_type == "TIMEOUT"

    def test_reset_clears_errors(self) -> None:
        eh = ErrorHandler(OrchestratorConfig())
        eh.handle_agent_error("a", RuntimeError("x"))
        eh.reset()
        assert eh.get_errors() == []
