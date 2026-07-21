"""Latency budgets, measured in model calls rather than seconds.

Wall-clock against a stub model measures this machine, not this system, so the
number that actually predicts production latency is **how many times we call the
model**. On the Gemini free tier (~10 RPM) each extra call is roughly six
seconds of user-visible wait, so a silent +2 in the call count is a real
regression even though every test still passes and the output is unchanged.

These budgets exist to make that regression loud. They are ceilings with a
little headroom, not exact counts — a change that legitimately needs another
call should raise the ceiling in the same commit that spends it, which is
precisely the review conversation worth forcing.

The one wall-clock assertion is deliberately generous: it is aimed at
pathological regressions (an accidental O(n^2) state copy, a retry loop that
spins), not at benchmarking.
"""

import asyncio
import json
import time

import pytest
from graph import nodes as graph_nodes
from models.state import MigrationState

VALID_PY = "def greet():\n    return 'hello'\n"

JAVA_SOURCE = (
    "import java.util.*;\n"
    "public class Report {\n"
    "    private final List<String> rows = new ArrayList<String>();\n"
    "    public void add(String r) { rows.add(r); }\n"
    "}\n"
)


class CountingLLM:
    """Stub model that records every call instead of making one.

    Mirrors ``StubLLM`` in tests_orchestrator.py, plus counting. It deliberately
    exposes no ``chat_model`` attribute, so agents take the ``call_llm``
    compatibility path and every model interaction lands in one counter.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append({"model": model, "fmt": fmt, "chars": len(prompt)})
        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps(
                {"plan_summary": "Upgrade the code.", "steps": [], "risk_areas": []}
            )
        if "failed validation" in prompt:
            return VALID_PY
        return json.dumps({"plan_summary": "Migrated.", "migrated_code": VALID_PY})

    async def stream_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ):
        yield await self.call_llm(prompt, system_prompt, fmt=fmt, model=model)


def _state() -> MigrationState:
    return MigrationState(
        source_code=JAVA_SOURCE,
        source_language="java",
        source_version="8",
        target_language="python",
        target_version="3.12",
    )


async def _migrate(settings, llm=None, cache=None, state=None):
    """Run one migration through the real pipeline; return (llm, result)."""
    from pipeline.orchestrator import Pipeline

    llm = llm or CountingLLM()
    pipeline = Pipeline(llm, cache_manager=cache, settings=settings, memory=None)
    try:
        result = await pipeline.run(state or _state())
    finally:
        await pipeline.aclose()
    return llm, result


# The agents that consult the *global* settings singleton rather than the
# injected one (agent_dispatcher.py, orchestrator_agent.py) can only be
# reconfigured through the environment, so every budget test goes through
# ``settings_override`` for consistency.
_OFFLINE = {
    "cache_enabled": "false",
    "enable_validation": "false",
    "memory_enabled": "false",
    "enable_rag": "false",
}

# The three flags that decide whether the *model* drives control flow.
_DECISION_FLAGS = (
    "dispatcher_llm_routing",
    "orchestrator_llm_planning",
    "enable_reflection",
)


def _config(**enabled: bool) -> dict:
    """A complete offline configuration with every decision flag pinned.

    ``settings_override`` sets environment variables, and monkeypatch's setenv
    accumulates for the duration of a test — so a partial override called twice
    in one test silently inherits the first call's flags. Naming all three every
    time makes each configuration total rather than differential.
    """
    config = dict(_OFFLINE)
    for flag in _DECISION_FLAGS:
        config[flag] = "true" if enabled.get(flag) else "false"
    return config


_FULLY_AGENTIC = _config(**{flag: True for flag in _DECISION_FLAGS})


class TestModelCallBudget:
    """How many times one migration talks to the model."""

    def test_shipped_defaults_stay_within_budget(self, settings_override):
        settings = settings_override(**_config())
        llm, result = asyncio.run(_migrate(settings))

        assert result.migrated_code, "budget is meaningless if nothing was produced"
        # Measured: 3 (analyze, plan, migrate). The ceiling allows one more so a
        # justified addition is not a test edit, but a fourth is the last free one.
        assert len(llm.calls) <= 4, [c["chars"] for c in llm.calls]

    def test_every_agentic_flag_on_stays_within_budget(self, settings_override):
        settings = settings_override(**_FULLY_AGENTIC)
        llm, result = asyncio.run(_migrate(settings))

        assert result.migrated_code
        # Measured: 8. This is the configuration that matters for real latency —
        # on a free tier it is already the better part of a minute, so the
        # ceiling is tight on purpose.
        assert len(llm.calls) <= 10, [c["chars"] for c in llm.calls]

    def test_llm_routing_costs_at_most_one_call_each(self, settings_override):
        """Routing and decomposition are *fast-role* calls; they must stay cheap.

        Each is a single yes/no or short-list decision. If either starts costing
        more than one call it has grown a loop, which is the expensive failure
        mode for a decision that was supposed to be a latency-neutral upgrade
        over a rule.
        """
        base, _ = asyncio.run(_migrate(settings_override(**_config())))
        routed, _ = asyncio.run(
            _migrate(settings_override(**_config(dispatcher_llm_routing=True)))
        )
        planned, _ = asyncio.run(
            _migrate(settings_override(**_config(orchestrator_llm_planning=True)))
        )

        assert len(routed.calls) - len(base.calls) <= 1
        assert len(planned.calls) - len(base.calls) <= 1


class TestCacheShortCircuit:
    """A cache hit must cost nothing — it is the whole point of the cache."""

    def test_cache_hit_makes_no_model_calls(self, settings_override):
        from cache.manager import CacheManager

        settings = settings_override(**{**_config(), "cache_enabled": "true"})
        cache = CacheManager()
        cache.clear()

        async def _both():
            first_llm, _ = await _migrate(settings, cache=cache)
            second_llm, second = await _migrate(settings, cache=cache)
            return first_llm, second_llm, second

        first_llm, second_llm, second = asyncio.run(_both())

        assert len(first_llm.calls) > 0, "first run should populate the cache"
        # The regression this guards is subtle: the cache-hit path returns early
        # in Pipeline.run, so a change that moves work above that return is
        # invisible in output and only shows up as latency.
        assert second_llm.calls == []
        assert second.migrated_code
        cache.clear()


class TestGraphWork:
    """Bounds on how much the graph itself does per migration."""

    def test_node_count_is_bounded(self, settings_override):
        settings = settings_override(**_FULLY_AGENTIC)
        _, result = asyncio.run(_migrate(settings))

        # Measured: 9 agents with every flag on. Each is a superstep with state
        # serialization on both sides, so this bounds graph overhead the same
        # way the call budget bounds model overhead.
        assert len(result.agents_done) <= 11, result.agents_done
        assert len(result.agents_done) == len(set(result.agents_done)), (
            f"an agent ran twice without a loop to justify it: {result.agents_done}"
        )

    def test_state_written_back_per_node_is_bounded(
        self, settings_override, monkeypatch
    ):
        """Total bytes each node hands back to LangGraph.

        Every node currently returns the *whole* accumulated state, so this grows
        quadratically with the number of nodes: superstep N re-serializes
        everything the first N-1 produced. Phase 3's delta writeback is what
        collapses it. Recorded here so the improvement is measurable rather than
        asserted, with a ceiling that catches it getting worse.
        """
        real_writeback = graph_nodes.writeback
        sizes: list[int] = []

        def _measured(state, mig_state):
            result = real_writeback(state, mig_state)
            sizes.append(len(json.dumps(result, default=str)))
            return result

        monkeypatch.setattr(graph_nodes, "writeback", _measured)

        settings = settings_override(**_config())
        _, result = asyncio.run(_migrate(settings))

        assert sizes, "no node wrote back; the probe missed"
        assert result.migrated_code
        total_kb = sum(sizes) / 1024
        # Measured at ~20 KB for an 8-node run over a 5-line source file. The
        # ceiling is generous because the payload scales with source size; what
        # it catches is the growth *rate* changing.
        assert total_kb < 200, f"{total_kb:.1f} KB written back across {len(sizes)} nodes"


class TestWallClock:
    """A backstop for pathological regressions, not a benchmark."""

    def test_stub_migration_finishes_promptly(self, settings_override):
        settings = settings_override(**_config())

        started = time.perf_counter()
        _, result = asyncio.run(_migrate(settings))
        elapsed = time.perf_counter() - started

        assert result.migrated_code
        # Measured at ~15 ms. Three orders of magnitude of headroom: this fires
        # only for something structurally wrong (a runaway loop, a deepcopy per
        # token), never for ordinary machine-to-machine variance.
        assert elapsed < 10.0, f"stub migration took {elapsed:.2f}s"


@pytest.mark.parametrize("flag", ["dispatcher_llm_routing", "orchestrator_llm_planning"])
def test_llm_decision_flags_are_reachable(settings_override, flag):
    """Guard against a decision flag that silently does nothing.

    Both of these read ``get_settings()`` directly rather than the injected
    settings, so passing ``Settings(flag=True)`` to the Pipeline does *not*
    enable them — a trap that made an earlier measurement of this exact budget
    read zero. If the flag ever stops costing a call, it has stopped working.
    """
    base, _ = asyncio.run(_migrate(settings_override(**_config())))
    enabled, _ = asyncio.run(
        _migrate(settings_override(**_config(**{flag: True})))
    )
    assert len(enabled.calls) > len(base.calls), f"{flag} made no model call"
