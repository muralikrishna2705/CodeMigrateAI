"""Tests for the runtime agents."""

import pytest

from models.state import MigrationState
from runtime.agent_dispatcher import DispatcherAgent
from runtime.agent_observer import ObserverAgent
from runtime.agent_providers import ProviderAgent
from runtime.agent_recovery import RecoveryAgent


def _state(**overrides) -> MigrationState:
    base = dict(
        source_code="x",
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
    )
    base.update(overrides)
    return MigrationState(**base)


class TestProviderAgent:
    @pytest.mark.asyncio
    async def test_register_and_get(self):
        agent = ProviderAgent(None)
        agent.register("test_key", {"foo": "bar"})
        assert agent.get("test_key") == {"foo": "bar"}
        assert agent.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_run_returns_registered_providers(self):
        agent = ProviderAgent(None)
        agent.register("cache", object())
        result = await agent.run(_state())
        assert result.success
        assert "cache" in result.details.get("providers", [])


class TestDispatcherAgent:
    @pytest.mark.asyncio
    async def test_low_complexity_routes_correctly(self):
        agent = DispatcherAgent(None)
        result = await agent.run(_state(code_metrics={"complexity": "low"}))
        assert result.success
        assert result.details["complexity"] == "low"
        assert "DeepAnalyzerAgent" not in result.details["pipeline"]

    @pytest.mark.asyncio
    async def test_high_complexity_routes_correctly(self):
        agent = DispatcherAgent(None)
        result = await agent.run(_state(code_metrics={"complexity": "high"}))
        assert result.success
        assert result.details["complexity"] == "high"
        assert "DeepAnalyzerAgent" in result.details["pipeline"]


class TestObserverAgent:
    @pytest.mark.asyncio
    async def test_collects_metrics(self):
        ObserverAgent.reset_metrics()
        agent = ObserverAgent(None)
        state = _state()
        state.record_success("AnalyzerAgent", "ok", duration_ms=10)
        state.record_error("MigratorAgent", "boom", duration_ms=5)

        result = await agent.run(state)
        metrics = result.details
        assert metrics["agents_run"] == 2
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 1
        assert metrics["total_duration_ms"] == 15
        assert metrics["by_agent"]["MigratorAgent"]["errors"] == 1


class TestRecoveryAgent:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_three_failures(self):
        from runtime import agent_recovery

        agent_recovery._circuit_state.clear()
        agent = RecoveryAgent(None)
        state = _state(errors=["[FlakyAgent] boom"])

        await agent.run(state)
        await agent.run(state)
        assert not agent._is_circuit_open("FlakyAgent")

        result = await agent.run(state)
        assert agent._is_circuit_open("FlakyAgent")
        assert result.success
