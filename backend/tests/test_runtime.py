"""Tests for the runtime helpers: DI provider, executor, observer, breaker."""

import contextvars

import pytest

from agents.base import BaseAgent
from models.state import MigrationState
from runtime import agent_recovery
from runtime.agent_observer import ObserverAgent
from runtime.agent_providers import Provider
from runtime.agent_runtime import Runtime


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


def _graph_state(**overrides) -> dict:
    """A minimal graph-state dict of the shape the Runtime hydrates from."""
    base = {
        "source_code": "x = 1",
        "source_language": "python",
        "source_version": "3.8",
        "target_language": "python",
        "target_version": "3.12",
        "reports": [],
        "errors": [],
        "agents_completed": [],
    }
    base.update(overrides)
    return base


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


class TestCircuitBreaker:
    def test_circuit_opens_after_three_failures(self):
        agent_recovery.reset()
        assert not agent_recovery.is_circuit_open("FlakyAgent")

        agent_recovery.record_failure("FlakyAgent")
        agent_recovery.record_failure("FlakyAgent")
        assert not agent_recovery.is_circuit_open("FlakyAgent")

        agent_recovery.record_failure("FlakyAgent")
        assert agent_recovery.is_circuit_open("FlakyAgent")

    def test_snapshot_reports_failures(self):
        agent_recovery.reset()
        agent_recovery.record_failure("A")
        snap = agent_recovery.snapshot()
        assert snap["A"]["failures"] == 1


class TestProvider:
    def test_register_and_get_instance(self):
        p = Provider()
        assert p.get("x") is None
        assert not p.has("x")

        sentinel = object()
        p.register("x", sentinel)
        assert p.has("x")
        assert p.get("x") is sentinel

    def test_factory_is_resolved_on_every_get(self):
        p = Provider()
        seq = iter(range(1, 100))
        p.register_factory("counter", lambda: next(seq))
        assert p.has("counter")
        assert p.get("counter") == 1
        assert p.get("counter") == 2  # re-resolved each call, not cached

    def test_factory_wins_over_instance(self):
        p = Provider()
        p.register("k", "instance")
        p.register_factory("k", lambda: "factory")
        assert p.get("k") == "factory"

    def test_resolve_omits_none_values(self):
        p = Provider()
        p.register("present", 1)
        p.register("absent", None)  # e.g. an unset RAG pipeline
        p.register_factory("callback", lambda: None)  # e.g. no active stream
        config = p.resolve(("present", "absent", "callback", "never_registered"))
        assert config == {"present": 1}

    def test_contextvar_factory_reflects_task_scope(self):
        # Mirrors how the SSE stream_callback flows through the Provider: a
        # task-scoped ContextVar read afresh on each resolve.
        var: contextvars.ContextVar = contextvars.ContextVar("t", default=None)
        p = Provider()
        p.register_factory("stream_callback", var.get)
        assert p.get("stream_callback") is None
        token = var.set("cb")
        try:
            assert p.get("stream_callback") == "cb"
        finally:
            var.reset(token)


class TestRuntime:
    @pytest.mark.asyncio
    async def test_skips_when_circuit_open(self):
        agent_recovery.reset()
        for _ in range(3):  # trip the breaker
            agent_recovery.record_failure("DispatcherAgent")
        assert agent_recovery.is_circuit_open("DispatcherAgent")

        runtime = Runtime(Provider())
        result = await runtime.execute("DispatcherAgent", _graph_state())
        # Returned unchanged: the agent never ran, so it isn't recorded.
        assert "DispatcherAgent" not in result.get("agents_completed", [])
        agent_recovery.reset()

    @pytest.mark.asyncio
    async def test_unknown_agent_records_error(self):
        runtime = Runtime(Provider())
        result = await runtime.execute("NoSuchAgent", _graph_state())
        assert any("NoSuchAgent" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_delegates_to_agent_and_writes_back(self):
        # DispatcherAgent (requires_llm=False, routing off) is offline-safe.
        from graph import nodes

        runtime = Runtime(Provider())
        nodes.set_llm_client(object())  # a placeholder the dispatcher never calls
        try:
            state = _graph_state(code_metrics={"complexity": "high"})
            result = await runtime.execute("DispatcherAgent", state)
        finally:
            nodes.set_llm_client(None)

        assert "DispatcherAgent" in result["agents_completed"]
        assert result["route_plan"]["deep_analyze"] is True

    @pytest.mark.asyncio
    async def test_records_failure_when_agent_errors(self):
        from graph import nodes

        agent_recovery.reset()

        # A throwaway agent that always errors; auto-registered on definition,
        # removed from the registry afterwards so it can't leak into other tests.
        class _BoomAgent(BaseAgent):
            name = "_BoomAgent"
            requires_llm = False

            async def run(self, state):
                raise RuntimeError("boom")

        runtime = Runtime(Provider())
        nodes.set_llm_client(object())
        try:
            result = await runtime.execute("_BoomAgent", _graph_state())
        finally:
            nodes.set_llm_client(None)
            BaseAgent._registry.pop("_BoomAgent", None)

        # BaseAgent.__call__ swallows the exception into an error report; the
        # Runtime turns that into a circuit-breaker failure.
        assert any("_BoomAgent" in str(e) or "boom" in str(e) for e in result["errors"])
        assert agent_recovery.snapshot().get("_BoomAgent", {}).get("failures") == 1
        agent_recovery.reset()
