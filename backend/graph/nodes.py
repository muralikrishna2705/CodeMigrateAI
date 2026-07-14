"""Agent wrapper node functions for the migration graph.

Each node hydrates a :class:`MigrationState` from the graph's dict-based state,
runs the corresponding agent, then writes the agent's outputs back into the
graph state so downstream nodes and edge conditions can read them.

The project's agents require a real LLM client at construction time
(``BaseAgent.__init__(llm_client, config)``), so we inject a shared
:class:`~llm.client.LLMClient` here rather than the ``None`` placeholder. Tests
can substitute a stub via :func:`set_llm_client`.
"""

import logging

# Importing the agent modules triggers AgentMeta auto-registration so that
# BaseAgent.get_registry() can resolve them by name below.
import agents.analyzer_agent  # noqa: F401
import agents.deep_analyzer_agent  # noqa: F401
import agents.fixer_agent  # noqa: F401
import agents.migrator_agent  # noqa: F401
import agents.planner_agent  # noqa: F401
import agents.validator_agent  # noqa: F401
from agents.base import BaseAgent
from models.state import AgentReport, MigrationState, MigrationType

log = logging.getLogger("CodeMigrateAI.GraphNodes")


# --- Shared LLM client (injectable for tests) -----------------------------

_llm_client = None


def get_llm_client():
    """Return the shared LLM client, creating it lazily on first use."""
    global _llm_client
    if _llm_client is None:
        from llm.client import LLMClient

        _llm_client = LLMClient()
    return _llm_client


def set_llm_client(client) -> None:
    """Override the shared LLM client (used by tests to inject a stub)."""
    global _llm_client
    _llm_client = client


# --- Shared token-stream callback (SSE token-by-token, MigratorAgent only) -

_stream_token_callback = None


def set_stream_callback(callback) -> None:
    """Set/clear the callback MigratorAgent streams generated tokens through.

    Mirrors the pre-graph orchestrator's ``agent.stream_callback = ...`` wiring:
    the graph builds a fresh ``MigratorAgent`` per invocation instead of reusing
    a persistent instance, so the callback is threaded through module state
    instead of set directly on an instance.
    """
    global _stream_token_callback
    _stream_token_callback = callback


# --- Helpers --------------------------------------------------------------


def _normalize_validation(vr: dict | None) -> dict | None:
    """Give validation results a consistent shape for conditions/FixerAgent.

    ``ValidatorAgent`` emits ``{syntax_valid, syntax_errors: {...}}`` while the
    external validator service emits ``{valid, errors, warnings}``. Downstream
    routing (:func:`conditions.validate_condition`) and the ``FixerAgent`` read
    ``valid`` / ``errors`` / ``warnings``, so surface those keys either way.
    """
    if not vr:
        return vr
    normalized = dict(vr)
    syntax_errors = vr.get("syntax_errors")
    nested = syntax_errors if isinstance(syntax_errors, dict) else {}
    normalized.setdefault("valid", vr.get("syntax_valid", True))
    normalized.setdefault("errors", nested.get("errors", []))
    normalized.setdefault("warnings", nested.get("warnings", []))
    return normalized


def hydrate_state(state: dict) -> MigrationState:
    """Build a MigrationState from the graph dict, carrying forward history."""
    mig_state = MigrationState(
        source_code=state["source_code"],
        source_language=state["source_language"],
        source_version=state["source_version"],
        target_language=state["target_language"],
        target_version=state["target_version"],
        migration_type=state.get("migration_type") or MigrationType.UPGRADE_VERSION,
    )
    # Copy existing analysis/plan/outputs if present.
    if state.get("code_metrics"):
        mig_state.code_metrics = state["code_metrics"]
    if state.get("inline_plan"):
        mig_state.inline_plan = state["inline_plan"]
    if state.get("migrated_code"):
        mig_state.migrated_code = state["migrated_code"]
    if state.get("rag_context"):
        mig_state.rag_context = state["rag_context"]
    if state.get("validation_result"):
        mig_state.validation_result = state["validation_result"]
    # Carry accumulated run history so reports/errors/agents grow across nodes
    # instead of being overwritten with only the latest node's output.
    mig_state.reports = [AgentReport(**r) for r in state.get("reports", [])]
    mig_state.errors = list(state.get("errors", []))
    mig_state.agents_done = list(state.get("agents_completed", []))
    return mig_state


def writeback(state: dict, mig_state: MigrationState) -> dict:
    """Write a MigrationState's outputs back into the graph state dict."""
    state["migration_type"] = mig_state.migration_type.value
    state["code_metrics"] = mig_state.code_metrics
    state["inline_plan"] = mig_state.inline_plan
    state["migrated_code"] = mig_state.migrated_code
    state["rag_context"] = mig_state.rag_context
    state["validation_result"] = _normalize_validation(mig_state.validation_result)
    state["reports"] = [r.model_dump() for r in mig_state.reports]
    state["errors"] = mig_state.errors
    state["agents_completed"] = mig_state.agents_done
    return state


def _make_node(agent_name: str):
    """Factory: creates a node function for a given agent."""

    async def node(state: dict) -> dict:
        agent_cls = BaseAgent.get_registry().get(agent_name)
        if not agent_cls:
            log.error("Agent %s not found in registry", agent_name)
            state["errors"] = state.get("errors", []) + [
                f"Agent {agent_name} not found"
            ]
            return state

        mig_state = hydrate_state(state)
        config = None
        if agent_name == "MigratorAgent" and _stream_token_callback is not None:
            config = {"stream_callback": _stream_token_callback}
        agent = agent_cls(get_llm_client(), config)  # LLM client injected here
        mig_state = await agent(mig_state)
        return writeback(state, mig_state)

    return node


# --- Individual node functions --------------------------------------------

analyze_node = _make_node("AnalyzerAgent")
deep_analyze_node = _make_node("DeepAnalyzerAgent")
plan_node = _make_node("PlannerAgent")
migrate_node = _make_node("MigratorAgent")
validate_node = _make_node("ValidatorAgent")

_fixer_node = _make_node("FixerAgent")


async def fix_node(state: dict) -> dict:
    """Run the FixerAgent and advance the retry counter.

    Incrementing ``retry_count`` here is what makes the
    ``validate -> fix -> migrate -> validate`` loop terminate: without it the
    loop would run forever because ``validate_condition`` would always observe
    ``retry_count < max_retries``.
    """
    # Preserve the current attempt as the best effort before re-migrating.
    if state.get("migrated_code"):
        state["best_effort_code"] = state["migrated_code"]

    state = await _fixer_node(state)
    state["retry_count"] = state.get("retry_count", 0) + 1
    log.info("Fix applied; retry_count now %d", state["retry_count"])
    return state
