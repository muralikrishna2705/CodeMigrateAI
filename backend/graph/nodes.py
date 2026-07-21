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
import time

# Importing the agent modules triggers AgentMeta auto-registration so that
# BaseAgent.get_registry() can resolve them by name below.
import agents.analyzer_agent  # noqa: F401
import agents.critic_agent  # noqa: F401
import agents.deep_analyzer_agent  # noqa: F401
import agents.fixer_agent  # noqa: F401
import agents.migrator_agent  # noqa: F401
import agents.orchestrator_agent  # noqa: F401
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

from .state import ACCUMULATING_FIELDS

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
    mig_state.parallel_tasks = list(state.get("parallel_tasks", []))
    mig_state.subgraph_results = list(state.get("subgraph_results", []))
    mig_state.reflection_score = state.get("reflection_score", 0.0)
    mig_state.reflection_feedback = state.get("reflection_feedback", "")
    mig_state.reflection_recommendation = state.get("reflection_recommendation", "pass")
    mig_state.memory_hits = list(state.get("memory_hits", []))
    mig_state.session_id = state.get("session_id", "")
    return mig_state


# Scalar graph fields: (key, how to read it off a MigrationState, the value
# hydrate_state produces when the key is absent from the graph dict).
#
# The default is what makes "changed" mean the same thing on a partially seeded
# state as on a fully seeded one. Comparing against a bare ``state.get(field)``
# would return None for an absent key, differ from the hydrated default, and
# report a change no agent made — so a direct graph.ainvoke() with a minimal
# state would see its first node emit half the schema.
_SCALAR_WRITEBACK = (
    ("inline_plan", lambda s: s.inline_plan, ""),
    ("migrated_code", lambda s: s.migrated_code, ""),
    ("rag_context", lambda s: s.rag_context, ""),
    ("route_plan", lambda s: s.route_plan, {}),
    ("parallel_tasks", lambda s: s.parallel_tasks, []),
    ("retrieval_requests", lambda s: s.retrieval_requests, []),
    ("reflection_score", lambda s: s.reflection_score, 0.0),
    ("reflection_feedback", lambda s: s.reflection_feedback, ""),
    ("reflection_recommendation", lambda s: s.reflection_recommendation, "pass"),
)


def writeback(state: dict, mig_state: MigrationState) -> dict:
    """Return the *delta* an agent produced, not the accumulated state.

    LangGraph folds each node's return value into the channels using the
    reducers declared on :class:`~graph.state.GraphState`. That makes returning
    the whole state actively wrong under an additive reducer — every node would
    re-append the entire run history — and it is what previously made native
    fan-out impossible, since two branches returning the full state conflict on
    every key rather than only on what they each touched.

    ``state`` is read but never mutated: it belongs to the channels, and a node
    that writes through to it would bypass the reducers entirely.

    Accumulating lists are diffed by length. The agent hydrated its
    ``MigrationState`` from this same ``state`` (see :func:`hydrate_state`), so
    everything past the inbound length is this agent's own contribution. The
    ``>`` guard covers an agent that replaced a list with a shorter one, which
    has no meaningful delta to take.
    """
    delta: dict = {}

    # Mirrors hydrate_state's own fallback, so an absent key is not a change.
    migration_type = mig_state.migration_type.value
    if migration_type != (
        state.get("migration_type") or MigrationType.UPGRADE_VERSION.value
    ):
        delta["migration_type"] = migration_type

    for field, read, default in _SCALAR_WRITEBACK:
        value = read(mig_state)
        if value != state.get(field, default):
            delta[field] = value

    metrics = mig_state.code_metrics
    if metrics is not None and metrics != state.get("code_metrics"):
        delta["code_metrics"] = metrics

    validation = _normalize_validation(mig_state.validation_result)
    if validation is not None and validation != state.get("validation_result"):
        delta["validation_result"] = validation

    # Counters carry a `max` reducer, so emitting the absolute value is
    # idempotent — the same delta applied twice is not two retries.
    if mig_state.reretrieval_count != state.get("reretrieval_count", 0):
        delta["reretrieval_count"] = mig_state.reretrieval_count

    for field, items in (
        ("reports", [r.model_dump() for r in mig_state.reports]),
        ("errors", mig_state.errors),
        ("agents_completed", mig_state.agents_done),
        ("subgraph_results", mig_state.subgraph_results),
    ):
        inbound = len(state.get(field) or [])
        if len(items) > inbound:
            delta[field] = list(items[inbound:])

    return delta


def state_delta(base: dict, updated: dict) -> dict:
    """Reduce a full graph state down to what changed relative to ``base``.

    The bridge for code that still thinks in whole states — currently the
    parallel fan-out's merge step. Accumulating fields yield only their new
    tail, because their reducers append whatever they are handed.
    """
    delta: dict = {}
    for key, value in updated.items():
        if key in ACCUMULATING_FIELDS:
            inbound = len(base.get(key) or [])
            if len(value or []) > inbound:
                delta[key] = list(value[inbound:])
        elif value != base.get(key):
            delta[key] = value
    return delta


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
orchestrate_node = _make_node("OrchestratorAgent")
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
    delta = await _migrate_node(state)
    if was_regeneration:
        # Counted from the *inbound* state. Reading it back off the delta would
        # always see 0 — the delta only carries what the agent changed, and the
        # agent does not touch this counter — so the loop would never reach
        # max_reflections and would spin until the recursion limit.
        delta["reflection_count"] = state.get("reflection_count", 0) + 1
        delta["reflection_feedback"] = ""
        delta["reflection_recommendation"] = "pass"
        log.info(
            "Reflection-driven re-migration done; reflection_count now %d",
            delta["reflection_count"],
        )
    return delta


async def reflect_node(state: dict) -> dict:
    """Optional self-reflection on the migrated code (Reflexion outer loop).

    Self-skips unless the orchestrator seeded ``enable_reflection`` — so offline /
    direct-graph runs (which build the graph without seeding it) never reflect —
    and there is code worth reflecting on. When it runs, the ReflectorAgent writes
    ``reflection_score``/``reflection_feedback``/``reflection_recommendation``,
    which ``reflect_condition`` routes on.
    """
    if not state.get("enable_reflection") or not state.get("migrated_code"):
        return {}
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
        return {}
    if state.get("errors"):
        return {}
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
    delta = await _retriever_node(state)
    if was_reretrieval:
        # Counted from the inbound state, as in migrate_node. Clearing the queue
        # relies on `retrieval_requests` having no reducer: under operator.add
        # an empty list is a no-op and the queue could never drain.
        delta["reretrieval_count"] = state.get("reretrieval_count", 0) + 1
        delta["retrieval_requests"] = []
        log.info(
            "Re-retrieval done; reretrieval_count now %d", delta["reretrieval_count"]
        )
    return delta


async def subgraph_branch_node(state: dict) -> dict:
    """Run one fanned-out sub-task as a compiled subgraph.

    The "execute" half of Plan-and-Execute. ``orchestrate_condition`` emits one
    ``Send`` per independent sub-task the OrchestratorAgent planned, each
    carrying the task name in ``branch_task``; LangGraph runs them concurrently
    in a single superstep and folds their returns through ``GraphState``'s
    reducers. That fan-in is what replaced the hand-rolled merge — the conflict
    policies it applied by hand are now the channel annotations.

    The branches share one inbound state without copying it. That is safe only
    because of the delta contract: nodes read the dict and return what changed
    rather than mutating in place, so there are no partial writes for a sibling
    to observe. The previous implementation deep-copied precisely because that
    was not yet true.

    A branch degrades rather than fails. LangGraph aborts the whole superstep on
    an unhandled exception, so a branch that blows up would take its siblings
    and the migration with it — where the contract is that a failed sub-task
    contributes nothing and the run continues with whatever context it has.

    ``subgraphs`` is imported here rather than at module scope because that
    module imports these node functions — a top-level import would be circular.
    """
    from .subgraphs import SUBGRAPH_TASKS

    task = state.get("branch_task") or ""
    started = time.perf_counter()

    def _record(ok: bool, agents: list, error: str | None = None) -> dict:
        record = {
            "task": task,
            "ok": ok,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "agents": list(agents),
        }
        if error:
            record["error"] = error
        return record

    builder = SUBGRAPH_TASKS.get(task)
    if builder is None:
        # The task names can come from an LLM decomposition, so one hallucinated
        # entry must not abort a migration the other branches can still finish.
        log.warning("Unknown subgraph task %r; contributing nothing", task)
        return {"subgraph_results": [_record(False, [], "unknown task")]}

    try:
        result = await builder().ainvoke(state)
    except Exception as exc:  # noqa: BLE001 — a branch must never abort the run
        log.exception("Parallel branch %r failed", task)
        return {"subgraph_results": [_record(False, [], str(exc))]}

    # ainvoke returns the subgraph's whole accumulated state; reduce it to this
    # branch's contribution before handing it to the parent's channels.
    delta = state_delta(state, result)
    delta["subgraph_results"] = [_record(True, delta.get("agents_completed", []))]
    log.info(
        "Parallel branch %r finished in %dms", task, delta["subgraph_results"][0]["duration_ms"]
    )
    return delta


async def fix_node(state: dict) -> dict:
    """Run the FixerAgent and advance the retry counter.

    Incrementing ``retry_count`` here is what makes the
    ``validate -> fix -> migrate -> validate`` loop terminate: without it the
    loop would run forever because ``validate_condition`` would always observe
    ``retry_count < max_retries``.
    """
    delta = await _fixer_node(state)

    # Preserve the current attempt as the best effort before re-migrating. Read
    # from the inbound state rather than the delta: the fixer does not rewrite
    # migrated_code, so the delta has no entry for it.
    if state.get("migrated_code"):
        delta["best_effort_code"] = state["migrated_code"]

    delta["retry_count"] = state.get("retry_count", 0) + 1
    log.info("Fix applied; retry_count now %d", delta["retry_count"])
    return delta
