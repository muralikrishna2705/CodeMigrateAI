"""Tests for Dimension 3 — agent reflection + decision-making.

Covers the shared reflection engine, the ReflectionTool, the BaseAgent.reflect
hook, the CriticAgent (heuristic + LLM critique), the ReflectorAgent graph node,
the reflect_condition routing, the reflect/migrate node wrappers, tool wiring, and
an offline end-to-end reflect -> re-migrate loop through the compiled graph.

Everything runs offline against scripted stub LLMs — no Ollama/validator service.
"""

import json

import pytest

from agents.base import BaseAgent, ReflectionResult
from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.reflector_agent import ReflectorAgent
from agents.tools import ReflectionTool, build_registry
from agents.tools.base import ToolRegistry
from agents.tools.reflection import CODE_CRITERIA, evaluate_output
from config import Settings
from graph import nodes
from graph.conditions import reflect_condition
from graph.migration_graph import build_migration_graph
from models.state import MigrationState

VALID_PY = "def greet():\n    return 'hello'\n"


def _state(**overrides) -> MigrationState:
    base = dict(
        source_code="print('hello')",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
    )
    base.update(overrides)
    return MigrationState(**base)


class ScriptedLLM:
    """A stub whose call_llm returns a fixed JSON string (or by prompt marker)."""

    def __init__(self, response: str, *, fast_model: str | None = None):
        self._response = response
        self.calls: list[str] = []
        if fast_model is not None:
            self.fast_model = fast_model

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None) -> str:
        self.calls.append(prompt)
        return self._response

    def extract_json(self, raw: str) -> dict:
        return json.loads(raw)


# --- ReflectionResult -------------------------------------------------------


class TestReflectionResult:
    def test_neutral_is_a_passing_verdict(self):
        r = ReflectionResult.neutral()
        assert r.passed
        assert r.recommendation == "pass"
        assert r.confidence == 1.0

    def test_passed_property_tracks_recommendation(self):
        assert ReflectionResult(0.9, "pass").passed
        assert not ReflectionResult(0.2, "re-generate").passed
        assert not ReflectionResult(0.5, "gather-more-info").passed


# --- evaluate_output (the shared engine) ------------------------------------


class TestEvaluateOutput:
    @pytest.mark.asyncio
    async def test_no_llm_degrades_to_pass(self):
        result = await evaluate_output(object(), output="x = 1")
        assert result.passed and result.details.get("degraded")

    @pytest.mark.asyncio
    async def test_empty_output_degrades_to_pass(self):
        llm = ScriptedLLM(json.dumps({"confidence": 0.1, "recommendation": "re-generate"}))
        result = await evaluate_output(llm, output="   ")
        assert result.passed
        assert not llm.calls  # short-circuited before calling the model

    @pytest.mark.asyncio
    async def test_parses_verdict(self):
        llm = ScriptedLLM(
            json.dumps(
                {
                    "confidence": 0.25,
                    "recommendation": "re-generate",
                    "feedback": "missing error handling",
                }
            )
        )
        result = await evaluate_output(llm, output="def f(): pass", criteria=CODE_CRITERIA)
        assert result.confidence == 0.25
        assert result.recommendation == "re-generate"
        assert result.feedback == "missing error handling"

    @pytest.mark.asyncio
    async def test_unparseable_response_degrades_to_pass(self):
        llm = ScriptedLLM("I think it looks fine, honestly")
        result = await evaluate_output(llm, output="x = 1")
        assert result.passed

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_is_rejected_not_clamped(self):
        # The Critique schema bounds confidence to [0.0, 1.0], so a value outside
        # it fails validation and the critique degrades to a passing verdict.
        # Previously this was clamped after parsing, which silently turned a
        # model that misunderstood the scale into a confident-looking answer.
        llm = ScriptedLLM(json.dumps({"confidence": 5, "recommendation": "pass"}))
        result = await evaluate_output(llm, output="x = 1")
        assert result.passed
        assert result.details == {"degraded": True}

    @pytest.mark.asyncio
    async def test_recommendation_outside_the_enum_degrades_to_pass(self):
        # The three actions are a Literal on the schema rather than a fuzzy
        # string match over "regenerate"/"redo"/"needs more context". Anything
        # else is now a validation failure, and a failed critique must never
        # block a migration.
        llm = ScriptedLLM(json.dumps({"confidence": 0.2, "recommendation": "redo it"}))
        result = await evaluate_output(llm, output="x = 1")
        assert result.passed

    @pytest.mark.asyncio
    async def test_valid_recommendation_is_carried_through(self):
        llm = ScriptedLLM(
            json.dumps(
                {"confidence": 0.2, "recommendation": "re-generate", "feedback": "stub"}
            )
        )
        result = await evaluate_output(llm, output="x = 1")
        assert result.recommendation == "re-generate"
        assert result.confidence == 0.2
        assert result.feedback == "stub"


# --- ReflectionTool ---------------------------------------------------------


class TestReflectionTool:
    @pytest.mark.asyncio
    async def test_returns_verdict_payload(self):
        llm = ScriptedLLM(
            json.dumps({"confidence": 0.3, "recommendation": "re-generate", "feedback": "fix"})
        )
        tool = ReflectionTool(llm)
        result = await tool(output="def f(): pass", stage="code")
        assert result.success
        assert result.data["recommendation"] == "re-generate"
        assert result.data["confidence"] == 0.3
        assert "re-generate" in result.summary


# --- BaseAgent.reflect hook -------------------------------------------------


class TestBaseReflectHook:
    @pytest.mark.asyncio
    async def test_falls_back_to_llm_when_no_tool(self):
        llm = ScriptedLLM(json.dumps({"confidence": 0.9, "recommendation": "pass"}))
        agent = PlannerAgent(llm, {})  # no tools in config
        result = await agent.reflect(_state(), "some plan")
        assert result.passed and result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_prefers_the_reflect_output_tool(self):
        tool_llm = ScriptedLLM(
            json.dumps({"confidence": 0.2, "recommendation": "re-generate", "feedback": "x"})
        )
        agent_llm = ScriptedLLM(json.dumps({"confidence": 1.0, "recommendation": "pass"}))
        registry = ToolRegistry([ReflectionTool(tool_llm)])
        agent = PlannerAgent(agent_llm, {"tools": registry})
        result = await agent.reflect(_state(), "some plan")
        # The verdict came from the tool's LLM, not the agent's own.
        assert result.recommendation == "re-generate"
        assert not agent_llm.calls


# --- CriticAgent ------------------------------------------------------------


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_stub_marker_floors_to_regenerate(self):
        # Model over-optimistically passes, but the code has a TODO -> forced regen.
        llm = ScriptedLLM(json.dumps({"confidence": 0.95, "recommendation": "pass"}))
        critic = CriticAgent(llm, {})
        code = "def greet():\n    # TODO: implement\n    pass\n"
        result = await critic.critique(code, state=_state())
        assert result.recommendation == "re-generate"
        assert result.confidence <= 0.4
        assert "stub_markers" in (result.details or {})

    @pytest.mark.asyncio
    async def test_clean_code_keeps_model_verdict(self):
        llm = ScriptedLLM(json.dumps({"confidence": 0.9, "recommendation": "pass"}))
        critic = CriticAgent(llm, {})
        result = await critic.critique(VALID_PY, state=_state())
        assert result.passed and result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_empty_code_is_neutral(self):
        critic = CriticAgent(ScriptedLLM("{}"), {})
        result = await critic.critique("   ", state=_state())
        assert result.passed

    @pytest.mark.asyncio
    async def test_run_writes_reflection_fields_to_state(self):
        llm = ScriptedLLM(
            json.dumps({"confidence": 0.3, "recommendation": "re-generate", "feedback": "f"})
        )
        critic = CriticAgent(llm, {})
        state = _state(migrated_code=VALID_PY)
        result = await critic.run(state)
        assert result.success
        assert state.reflection_recommendation == "re-generate"
        assert state.reflection_score == 0.3
        assert state.reflection_feedback == "f"


# --- ReflectorAgent (graph node) --------------------------------------------


class TestReflectorAgent:
    @pytest.mark.asyncio
    async def test_reflects_on_code_and_writes_state(self):
        llm = ScriptedLLM(
            json.dumps({"confidence": 0.4, "recommendation": "re-generate", "feedback": "y"})
        )
        agent = ReflectorAgent(llm, {})
        state = _state(migrated_code=VALID_PY)
        await agent(state)
        assert state.reflection_recommendation == "re-generate"
        assert "ReflectorAgent" in state.agents_done

    @pytest.mark.asyncio
    async def test_nothing_to_reflect_passes_through(self):
        agent = ReflectorAgent(ScriptedLLM("{}"), {})
        state = _state()  # no migrated_code, no plan
        await agent(state)
        assert state.reflection_recommendation == "pass"


# --- reflect_condition ------------------------------------------------------


class TestReflectCondition:
    def test_pass_routes_to_validate(self):
        assert reflect_condition({"reflection_recommendation": "pass"}) == "validate"

    def test_missing_recommendation_routes_to_validate(self):
        # Reflection disabled -> field absent -> proceed to validate.
        assert reflect_condition({}) == "validate"

    def test_low_confidence_with_budget_routes_to_re_migrate(self):
        assert (
            reflect_condition(
                {
                    "reflection_recommendation": "re-generate",
                    "reflection_count": 0,
                    "max_reflections": 1,
                }
            )
            == "re_migrate"
        )

    def test_budget_spent_routes_to_validate(self):
        assert (
            reflect_condition(
                {
                    "reflection_recommendation": "re-generate",
                    "reflection_count": 1,
                    "max_reflections": 1,
                }
            )
            == "validate"
        )


# --- node wrappers ----------------------------------------------------------


class TestNodeWrappers:
    @pytest.mark.asyncio
    async def test_reflect_node_self_skips_when_disabled(self):
        # enable_reflection absent -> the node returns state untouched, unrun.
        state = {"migrated_code": VALID_PY, "reflection_recommendation": "pass"}
        result = await nodes.reflect_node(dict(state))
        assert "ReflectorAgent" not in result.get("agents_completed", [])

    @pytest.mark.asyncio
    async def test_migrate_node_counts_and_clears_reflection_regeneration(self):
        stub = ScriptedLLM(json.dumps({"plan_summary": "m", "migrated_code": VALID_PY}))
        nodes.set_llm_client(stub)
        try:
            state = {
                "source_code": "print('hi')",
                "source_language": "python",
                "source_version": "2.7",
                "target_language": "python",
                "target_version": "3.12",
                "migration_type": "upgrade_version",
                "reports": [],
                "errors": [],
                "agents_completed": [],
                "reflection_feedback": "make it idiomatic",
                "reflection_recommendation": "re-generate",
                "reflection_count": 0,
            }
            result = await nodes.migrate_node(state)
        finally:
            nodes.set_llm_client(None)

        assert result["reflection_count"] == 1
        assert result["reflection_feedback"] == ""
        assert result["reflection_recommendation"] == "pass"


# --- build_registry wiring --------------------------------------------------


class TestReflectionToolWiring:
    def test_tool_absent_by_default(self):
        registry = build_registry(llm_client=ScriptedLLM("{}"), settings=Settings())
        assert "reflect_output" not in registry.names()

    def test_tool_present_when_enabled_with_llm(self):
        registry = build_registry(
            llm_client=ScriptedLLM("{}"), settings=Settings(enable_reflection=True)
        )
        assert "reflect_output" in registry.names()

    def test_tool_absent_without_llm(self):
        registry = build_registry(
            llm_client=None, settings=Settings(enable_reflection=True)
        )
        assert "reflect_output" not in registry.names()


# --- end-to-end through the compiled graph ----------------------------------


def _graph_state(**overrides) -> dict:
    base = {
        "source_code": "print('hello')",
        "source_language": "python",
        "source_version": "2.7",
        "target_language": "python",
        "target_version": "3.12",
        "migration_type": "upgrade_version",
        "retry_count": 0,
        "max_retries": 2,
        "best_effort_code": "",
        "reports": [],
        "errors": [],
        "agents_completed": [],
    }
    base.update(overrides)
    return base


def test_graph_includes_reflect_node():
    app = build_migration_graph()
    node_names = set(getattr(app, "nodes", {}) or {})
    assert "reflect" in node_names


class ReflectLoopLLM:
    """Drives one reflect -> re-migrate loop.

    The first migration emits a REGEN_ME sentinel that the (graph-level)
    reflection flags for regeneration; the second is clean. Reflection routes on
    the code embedded in the critique prompt, so it needs no call counting.
    """

    def __init__(self):
        self.migrator_calls = 0

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None) -> str:
        if prompt.startswith("Analyze this"):
            return json.dumps(
                {
                    "deprecated_patterns": [],
                    "migration_challenges": [],
                    "key_constructs": [],
                    "summary": "Stub analysis.",
                }
            )
        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps({"plan_summary": "plan", "steps": [], "risk_areas": []})
        if "Critically review" in prompt:
            if "REGEN_ME" in prompt:
                return json.dumps(
                    {
                        "confidence": 0.9,  # high, so INTERNAL reflection won't fire
                        "recommendation": "re-generate",
                        "feedback": "Replace the REGEN_ME sentinel with real logic.",
                    }
                )
            return json.dumps({"confidence": 0.95, "recommendation": "pass", "feedback": ""})
        if "failed validation" in prompt:
            return VALID_PY
        self.migrator_calls += 1
        code = (
            "def greet():\n    return 'REGEN_ME'\n"
            if self.migrator_calls == 1
            else VALID_PY
        )
        return json.dumps({"plan_summary": "m", "migrated_code": code})

    def extract_json(self, raw):
        return json.loads(raw)


@pytest.mark.asyncio
async def test_reflection_loop_regenerates_low_confidence_code():
    """Graph-level reflection: low-confidence code -> re-migrate -> pass, bounded."""
    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(ReflectLoopLLM())
    try:
        app = build_migration_graph()
        state = _graph_state(
            # Enable the graph reflect node at the state level (settings stay off,
            # so the in-agent reflection doesn't also fire and muddy the count).
            enable_reflection=True,
            reflection_count=0,
            max_reflections=1,
            reflection_recommendation="pass",
            reflection_feedback="",
        )
        result = await app.ainvoke(state)
    finally:
        nodes.set_llm_client(None)

    assert result["reflection_count"] == 1  # exactly one regeneration
    assert result["migrated_code"].strip() == VALID_PY.strip()
    assert "ReflectorAgent" in result["agents_completed"]
    assert result["validation_result"]["valid"] is True


@pytest.mark.asyncio
async def test_reflection_disabled_does_not_loop_or_run_reflector():
    """With reflection off (default), the reflect node self-skips end-to-end."""
    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(ReflectLoopLLM())
    try:
        app = build_migration_graph()
        result = await app.ainvoke(_graph_state())  # no enable_reflection
    finally:
        nodes.set_llm_client(None)

    assert result.get("reflection_count", 0) == 0
    assert "ReflectorAgent" not in result["agents_completed"]
