"""Agent wrapper node functions for the migration graph.

Each node hydrates a :class:`MigrationState` from the graph's dict-based state,
runs the corresponding agent, then writes the agent's outputs back into the
graph state so downstream nodes and edge conditions can read them.

The project's agents require a real LLM client at construction time
(``BaseAgent.__init__(llm_client, config)``), so we inject a shared
:class:`~llm.client.LLMClient` here rather than the ``None`` placeholder. Tests
can substitute a stub via :func:`set_llm_client`.
"""

import contextvars
import logging

# Importing the agent modules triggers AgentMeta auto-registration so that
# BaseAgent.get_registry() can resolve them by name below.
import agents.analyzer_agent  # noqa: F401
import agents.critic_agent  # noqa: F401
import agents.deep_analyzer_agent  # noqa: F401
import agents.fixer_agent  # noqa: F401
import agents.migrator_agent  # noqa: F401
import agents.planner_agent  # noqa: F401
import agents.reflector_agent  # noqa: F401
import agents.retriever_agent  # noqa: F401
import agents.validator_agent  # noqa: F401
import runtime.agent_dispatcher  # noqa: F401
import runtime.agent_observer  # noqa: F401
import runtime.agent_validator  # noqa: F401
from models.state import AgentReport, MigrationState, MigrationType
from runtime.agent_providers import Provider
from runtime.agent_runtime import Runtime

log = logging.getLogger("CodeMigrateAI.GraphNodes")


# --- Dynamic wiring: Provider (DI container) + Runtime (executor) ----------
#
# Shared dependencies (LLM client, RAG pipeline, per-request stream callback) are
# registered into a single Provider; the Runtime resolves each agent's declared
# ``needs`` from it when a graph node runs. This replaces the old scatter of
# module globals and the hardcoded per-agent config block.

_provider = Provider()
_runtime = Runtime(_provider)


def get_llm_client():
    """Return the shared LLM client, creating it lazily if none is registered.

    A registered ``None`` (tests clear the client this way between runs) is
    treated as "unset", so lazy creation still kicks in — matching the old
    ``if _llm_client is None`` behaviour.
    """
    client = _provider.get("llm")
    if client is None:
        from llm.client import LLMClient

        client = LLMClient()
        _provider.register("llm", client)
    return client


def set_llm_client(client) -> None:
    """Override the shared LLM client (used by tests to inject a stub)."""
    _provider.register("llm", client)
    # The reflect_output tool is built with this client, so rebuild the registry
    # whenever it changes — without this, reflection would have no tool in a live
    # app that never re-registers the RAG pipeline (e.g. RAG disabled).
    rebuild_tools()


def set_rag_pipeline(pipeline) -> None:
    """Register the shared RAG pipeline used by the ``retrieve`` node.

    Built lazily in the app lifespan, after this module is imported. Until then
    (and whenever ``None`` is registered) the Provider resolves it to nothing, so
    the RetrieverAgent simply skips retrieval.
    """
    _provider.register("rag_pipeline", pipeline)
    rebuild_tools()


def set_migration_memory(memory) -> None:
    """Register the cross-session migration memory backing ``semantic_search``."""
    _provider.register("migration_memory", memory)
    rebuild_tools()


# --- Tools ----------------------------------------------------------------
#
# The tool registry is a Provider dependency like any other; agents opt in with
# `needs = ("tools",)`. It is rebuilt whenever a backing service is registered
# because VectorDBTool/SemanticSearchTool are constructed with the pipeline and
# memory respectively — both of which are wired lazily in the app lifespan, after
# this module is imported. Rebuilding (rather than registering once at import)
# is what lets those tools exist at all in a live app, while the dependency-free
# tools (metrics, syntax, source reader) work from import time in offline runs.


def rebuild_tools() -> None:
    """Rebuild the tool registry from the Provider's current services."""
    from agents.tools import build_registry

    registry = build_registry(
        rag_pipeline=_provider.get("rag_pipeline"),
        migration_memory=_provider.get("migration_memory"),
        llm_client=_provider.get("llm"),
    )
    _provider.register_tools(registry)
    log.info("Tools available: %s", ", ".join(registry.names()) or "(none)")


def set_tools(registry) -> None:
    """Override the tool registry (used by tests to inject stub tools)."""
    _provider.register_tools(registry)


rebuild_tools()


# --- Per-request token-stream callback (SSE token-by-token, MigratorAgent) -
#
# The graph builds a fresh ``MigratorAgent`` per invocation instead of reusing a
# persistent instance, so the SSE callback can't live on an agent instance. A
# plain module global would be shared across every in-flight request on the
# worker, letting two concurrent /migrate/stream calls clobber each other's
# callback and cross-wire tokens between clients. A ``ContextVar`` scopes the
# callback to the asyncio task running each request: ``.set()`` only mutates the
# calling task's context, and LangGraph's per-superstep tasks inherit it at
# creation time, so concurrent requests stay isolated. It is exposed through the
# Provider as a factory (resolved per ``get``) so ``stream_callback`` flows
# through the same needs-based DI as every other dependency.

_stream_token_callback: contextvars.ContextVar = contextvars.ContextVar(
    "codemigrate_stream_token_callback", default=None
)
_provider.register_factory("stream_callback", _stream_token_callback.get)


def set_stream_callback(callback) -> "contextvars.Token":
    """Set the per-request token callback; returns a token to reset it with."""
    return _stream_token_callback.set(callback)


def reset_stream_callback(token: "contextvars.Token") -> None:
    """Restore the callback to its previous (per-request) value."""
    _stream_token_callback.reset(token)


# --- Helpers --------------------------------------------------------------


def _normalize_validation(vr: dict | None) -> dict | None:
    """Ensure validation results carry the keys conditions/FixerAgent read.

    Both the ``ValidatorAgent`` and the external validator service now emit the
    unified ``{valid, errors, warnings}`` shape, so this is just a defensive
    pass-through that fills in defaults for any missing key.
    """
    if not vr:
        return vr
    normalized = dict(vr)
    normalized.setdefault("valid", True)
    normalized.setdefault("errors", [])
    normalized.setdefault("warnings", [])
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
    if "rag_context" in state:
        mig_state.rag_context = state["rag_context"]
    if state.get("validation_result"):
        mig_state.validation_result = state["validation_result"]
    # Carry accumulated run history so reports/errors/agents grow across nodes
    # instead of being overwritten with only the latest node's output.
    mig_state.reports = [AgentReport(**r) for r in state.get("reports", [])]
    mig_state.errors = list(state.get("errors", []))
    mig_state.agents_done = list(state.get("agents_completed", []))
    mig_state.retrieval_requests = list(state.get("retrieval_requests", []))
    mig_state.reretrieval_count = state.get("reretrieval_count", 0)
    mig_state.route_plan = state.get("route_plan") or {}
    mig_state.reflection_score = state.get("reflection_score", 0.0)
    mig_state.reflection_feedback = state.get("reflection_feedback", "")
    mig_state.reflection_recommendation = state.get("reflection_recommendation", "pass")
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
    state["retrieval_requests"] = mig_state.retrieval_requests
    state["reretrieval_count"] = mig_state.reretrieval_count
    state["route_plan"] = mig_state.route_plan
    state["reflection_score"] = mig_state.reflection_score
    state["reflection_feedback"] = mig_state.reflection_feedback
    state["reflection_recommendation"] = mig_state.reflection_recommendation
    return state


def last_report_status(mig_state: MigrationState, agent_name: str) -> str | None:
    """Status of the most recent report this agent produced, if any.

    Public because the Runtime executor (``runtime.agent_runtime``) reads it to
    decide whether a run should trip the circuit breaker.
    """
    report = next(
        (r for r in reversed(mig_state.reports) if r.agent == agent_name), None
    )
    return report.status if report else None


def _make_node(agent_name: str):
    """Factory: a graph node that delegates the agent's execution to the Runtime.

    All the per-run mechanics — the circuit-breaker guard, needs-based dependency
    injection from the Provider, failure recording, and hydrate/writeback — live
    in ``Runtime.execute`` so every node shares one implementation instead of
    inlining it.
    """

    async def node(state: dict) -> dict:
        return await _runtime.execute(agent_name, state)

    return node


# --- Individual node functions --------------------------------------------

analyze_node = _make_node("AnalyzerAgent")
dispatch_node = _make_node("DispatcherAgent")
deep_analyze_node = _make_node("DeepAnalyzerAgent")
plan_node = _make_node("PlannerAgent")
validate_node = _make_node("ValidatorAgent")
observe_node = _make_node("ObserverAgent")

_migrate_node = _make_node("MigratorAgent")
_retriever_node = _make_node("RetrieverAgent")
_reflect_node = _make_node("ReflectorAgent")
_fixer_node = _make_node("FixerAgent")
_service_validate_node = _make_node("RuntimeValidatorAgent")


async def migrate_node(state: dict) -> dict:
    """Run the MigratorAgent; consume a reflection-driven regeneration.

    On the first pass (from ``plan``) or a fix-driven re-migration, the reflection
    fields are unset/passing and this is a plain migration. When the ``reflect``
    node has sent the run back with a non-pass verdict, the MigratorAgent
    regenerates *using* ``reflection_feedback``; this wrapper then advances
    ``reflection_count`` (which bounds the reflect -> migrate loop, mirroring how
    ``retry_count`` bounds the fix loop) and clears the verdict so the next
    reflection starts fresh. Detecting the re-entry from the feedback signal
    mirrors how ``retrieve_node`` detects a re-retrieval.
    """
    was_regeneration = (
        bool(state.get("reflection_feedback"))
        and (state.get("reflection_recommendation") or "pass") != "pass"
    )
    state = await _migrate_node(state)
    if was_regeneration:
        state["reflection_count"] = state.get("reflection_count", 0) + 1
        state["reflection_feedback"] = ""
        state["reflection_recommendation"] = "pass"
        log.info(
            "Reflection-driven re-migration done; reflection_count now %d",
            state["reflection_count"],
        )
    return state


async def reflect_node(state: dict) -> dict:
    """Optional self-reflection on the migrated code (Reflexion outer loop).

    Self-skips unless the orchestrator seeded ``enable_reflection`` — so offline /
    direct-graph runs (which build the graph without seeding it) never reflect —
    and there is code worth reflecting on. When it runs, the ReflectorAgent writes
    ``reflection_score``/``reflection_feedback``/``reflection_recommendation``,
    which ``reflect_condition`` routes on.
    """
    if not state.get("enable_reflection") or not state.get("migrated_code"):
        return state
    return await _reflect_node(state)


async def service_validate_node(state: dict) -> dict:
    """Optional deep validation against the external validator *service*.

    The in-graph ``validate`` node (ValidatorAgent) is an offline syntax gate that
    drives the fix loop; this terminal node adds an authoritative service check
    once that loop settles. It self-skips unless the orchestrator seeded
    ``enable_validation`` — so offline/unit runs (which build the graph directly
    without seeding it) never touch the network — and there is clean code worth
    validating.
    """
    if not state.get("enable_validation") or not state.get("migrated_code"):
        return state
    if state.get("errors"):
        return state
    return await _service_validate_node(state)


async def retrieve_node(state: dict) -> dict:
    """Run the RetrieverAgent; count + consume a re-retrieval request.

    On the first pass ``retrieval_requests`` is empty and this simply provisions
    RAG context. When the MigratorAgent has enqueued ungrounded imports it is a
    *re*-retrieval: advance ``reretrieval_count`` (which bounds the
    migrate -> retrieve loop, mirroring the fix loop's ``retry_count``) and clear
    the queue so the loop can't spin on the same request.
    """
    was_reretrieval = bool(state.get("retrieval_requests"))
    state = await _retriever_node(state)
    if was_reretrieval:
        state["reretrieval_count"] = state.get("reretrieval_count", 0) + 1
        state["retrieval_requests"] = []
        log.info(
            "Re-retrieval done; reretrieval_count now %d", state["reretrieval_count"]
        )
    return state


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
