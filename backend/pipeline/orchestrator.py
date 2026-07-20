import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from cache.keys import generate_key
from cache.manager import CacheManager
from config import get_settings
from graph import nodes as graph_nodes
from graph.migration_graph import build_migration_graph
from llm.streaming import SSEStreamHandler
from memory.checkpointer import build_checkpointer
from memory.migration_memory import summarize_hits
from models.state import MigrationState

log = logging.getLogger("CodeMigrateAI.Pipeline")


class Pipeline:
    def __init__(
        self,
        llm_client,
        cache_manager: Optional[CacheManager] = None,
        settings=None,
        memory=None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache_manager
        # Persistent memory (Dimension 5). Optional: None means every run starts
        # cold, which is exactly the pre-Dimension-5 behaviour.
        self.memory = memory
        # Opt in to checkpointing: the pipeline is the caller that owns session
        # ids, so it is the one that can honour the thread_id contract.
        self._checkpointer = build_checkpointer(self.settings)
        self._graph = build_migration_graph(
            checkpointer=self._checkpointer, settings=self.settings
        )
        # Graph nodes build a fresh agent instance per call rather than reusing
        # a persistent one, so the shared LLM client is threaded through
        # module state instead of instance state.
        graph_nodes.set_llm_client(llm_client)

    async def run(self, state: MigrationState) -> MigrationState:
        if self.cache and self.settings.cache_enabled:
            cache_key = generate_key(state)
            cached = await self.cache.get(cache_key)
            if cached:
                log.info("Cache hit for %s...", cache_key[:16])
                # A streaming client is blocked on the SSE queue waiting for a
                # terminal event. Without this, a cache hit returns here and the
                # stream never emits `complete`, so the browser hangs forever.
                if self.settings.enable_streaming and hasattr(state, "_stream_queue"):
                    await SSEStreamHandler(state._stream_queue).send_complete(cached)
                return cached

        log.info("=" * 60)
        log.info(
            "Pipeline START: %s %s to %s %s",
            state.source_language,
            state.source_version,
            state.target_language,
            state.target_version,
        )
        log.info("Source size: %d chars", len(state.source_code))
        log.info("=" * 60)

        stream_handler = None
        if self.settings.enable_streaming and hasattr(state, "_stream_queue"):
            stream_handler = SSEStreamHandler(state._stream_queue)

        # A session id is required, not optional: it is the LangGraph checkpoint
        # thread_id, and the compiled graph refuses to run without one. A caller
        # supplying a previous run's id resumes that run's checkpoint.
        if not state.session_id:
            state.session_id = uuid.uuid4().hex

        # Cross-session recall, before any agent runs, so the plan and migration
        # can both be grounded in prior art.
        await self._recall(state)

        # Everything — analysis, dynamic dispatch, RAG retrieval, planning,
        # migration, validation (offline + optional external service), the fix
        # retry loop, and observability — runs inside the compiled LangGraph now
        # (see graph/migration_graph.py). There is no linear prelude anymore.
        state = await self._run_graph(state, stream_handler)

        state.completed_at = datetime.utcnow()

        await self._persist(state)

        if (
            self.cache
            and self.settings.cache_enabled
            and state.migrated_code
            and not state.errors
        ):
            cache_key = generate_key(state)
            await self.cache.set(cache_key, state)

        if stream_handler:
            await stream_handler.send_complete(state)

        log.info(
            "Pipeline END. Agents done: %s | Errors: %d",
            state.agents_done,
            len(state.errors),
        )
        return state

    async def _run_graph(
        self, state: MigrationState, stream_handler: Optional[SSEStreamHandler]
    ) -> MigrationState:
        graph_state = {
            "source_code": state.source_code,
            "source_language": state.source_language,
            "source_version": state.source_version,
            "target_language": state.target_language,
            "target_version": state.target_version,
            "migration_type": state.migration_type.value,
            "code_metrics": state.code_metrics,
            "inline_plan": state.inline_plan,
            "migrated_code": state.migrated_code,
            "rag_context": state.rag_context,
            "validation_result": state.validation_result,
            "reports": [r.model_dump() for r in state.reports],
            "errors": list(state.errors),
            "agents_completed": list(state.agents_done),
            "retrieval_requests": list(state.retrieval_requests),
            "reretrieval_count": 0,
            "max_reretrievals": self.settings.max_reretrievals,
            "route_plan": dict(state.route_plan),
            # Persistent memory (Dimension 5): recalled before the graph starts
            # so every node sees the same prior art without re-querying.
            "memory_hits": list(state.memory_hits),
            "session_id": state.session_id,
            # Dynamic orchestration (Dimension 4): seeded from settings so the
            # parallel node self-skips unless enabled. The OrchestratorAgent
            # fills parallel_tasks in-graph; subgraph_results accumulates one
            # record per fanned-out branch.
            "parallel_tasks": list(state.parallel_tasks),
            "subgraph_results": list(state.subgraph_results),
            "parallel_enabled": self.settings.parallel_enabled,
            "max_parallel_tasks": self.settings.max_parallel_tasks,
            # Reflection (Dimension 3): seeded from settings so the reflect node
            # self-skips unless enabled, bounded by max_reflections like the fix loop.
            "reflection_score": 0.0,
            "reflection_feedback": "",
            "reflection_recommendation": "pass",
            "reflection_count": 0,
            "max_reflections": self.settings.max_reflections,
            "enable_reflection": self.settings.enable_reflection,
            "retry_count": 0,
            "max_retries": self.settings.max_retries,
            "best_effort_code": "",
            "final_result": None,
            # Gates the in-graph service_validate node; seeded from settings so
            # offline runs (which never set this) never touch the validator.
            "enable_validation": self.settings.enable_validation,
        }

        callback_token = None
        if stream_handler:
            callback_token = graph_nodes.set_stream_callback(stream_handler.send_token)

        # thread_id keys the checkpoint. The compiled graph raises without it
        # whenever a checkpointer is attached, so it is always supplied.
        run_config = {"configurable": {"thread_id": state.session_id or uuid.uuid4().hex}}

        final_graph_state = graph_state
        try:
            async for chunk in self._graph.astream(
                graph_state, config=run_config, stream_mode="updates"
            ):
                for _node_name, node_state in chunk.items():
                    final_graph_state = node_state
                    reports = node_state.get("reports") or []
                    if not reports or not stream_handler:
                        continue
                    # astream("updates") only yields once a node has finished,
                    # so start/complete fire back-to-back rather than framing
                    # the node's actual runtime.
                    report = reports[-1]
                    agent_name = report.get("agent", _node_name)
                    await stream_handler.send_agent_start(
                        agent_name, f"Starting {agent_name}..."
                    )
                    await stream_handler.send_agent_complete(agent_name, report)
        finally:
            if callback_token is not None:
                graph_nodes.reset_stream_callback(callback_token)

        return graph_nodes.hydrate_state(final_graph_state)

    async def aclose(self) -> None:
        """Release the checkpointer's database connection.

        ``AsyncSqliteSaver`` holds an aiosqlite connection backed by its own
        thread. Left open, it is finalized after the event loop has gone and
        raises "Event loop is closed" from the garbage collector — noise in
        tests, and a leaked connection per Pipeline anywhere a process builds
        more than one.
        """
        conn = getattr(self._checkpointer, "conn", None)
        if conn is None:
            return
        try:
            await conn.close()
        except Exception as exc:  # noqa: BLE001 — shutdown must not raise
            log.debug("Checkpointer close failed: %s", exc)

    # ------------------------------------------------------------------ memory

    async def _recall(self, state: MigrationState) -> None:
        """Look up similar past migrations and fold them into the RAG context.

        Hits are appended to ``rag_context`` rather than handed to the agents on
        a separate channel: every agent already consumes that context, so prior
        art reaches all of them without touching an agent signature. They are
        appended *after* the retrieved documentation so official docs keep
        precedence — a past migration is evidence of what we did, not of what is
        correct.
        """
        if self.memory is None or not self.settings.memory_enabled:
            return
        try:
            hits = await asyncio.to_thread(
                self.memory.recall,
                source_code=state.source_code,
                source_language=state.source_language,
                target_language=state.target_language,
                k=self.settings.memory_top_k,
            )
        except Exception as exc:  # noqa: BLE001 — recall is an optimisation
            log.warning("Memory recall failed; continuing without it: %s", exc)
            return

        if not hits:
            log.info("Memory: no prior migrations for this language pair")
            return

        state.memory_hits = hits
        context = summarize_hits(hits, limit=self.settings.memory_top_k)
        if context:
            state.rag_context = (
                f"{state.rag_context}\n\n## Similar past migrations\n{context}"
                if state.rag_context
                else f"## Similar past migrations\n{context}"
            )
        log.info(
            "Memory: %d prior migration(s) recalled (best similarity %.2f)",
            len(hits),
            hits[0].get("similarity", 0.0),
        )

    async def _persist(self, state: MigrationState) -> None:
        """Record this run's outcome for future sessions.

        Successes and failures both go in, to different tables. The failure row
        is never retrieved as grounding (see ``MemoryStore``) — it exists so a
        later run can be warned off a construct that has burned us before.
        """
        if self.memory is None or not self.settings.memory_enabled:
            return

        failed = bool(state.errors) or not state.migrated_code
        validation = state.validation_result or {}
        if not validation.get("valid", True):
            failed = True

        try:
            if failed:
                await asyncio.to_thread(
                    self.memory.record_failure,
                    source_code=state.source_code,
                    source_language=state.source_language,
                    target_language=state.target_language,
                    reason=state.errors[0] if state.errors else "no code produced",
                    details={"errors": state.errors, "validation": validation},
                    session_id=state.session_id,
                )
                return

            await asyncio.to_thread(
                self.memory.remember,
                source_code=state.source_code,
                source_language=state.source_language,
                source_version=state.source_version,
                target_language=state.target_language,
                target_version=state.target_version,
                migrated_code=state.migrated_code,
                plan=state.inline_plan,
                score=_quality_score(state),
                session_id=state.session_id,
            )
        except Exception as exc:  # noqa: BLE001 — never fail a run over memory
            log.warning("Could not persist migration memory: %s", exc)


def _quality_score(state: MigrationState) -> float:
    """Confidence in a completed migration, in [0.0, 1.0].

    Ranks one stored migration against another during recall, so it only has to
    be ordinally sensible. Starts from whether validation passed (the one
    objective signal available), then nudges on the model's self-assessment and
    down for a run that needed retries to get there.
    """
    validation = state.validation_result or {}
    score = 0.8 if validation.get("valid", False) else 0.5
    if state.reflection_score:
        # Blend rather than replace: a model's confidence in its own output is
        # weaker evidence than a validator actually running the code.
        score = 0.7 * score + 0.3 * float(state.reflection_score)
    retries = sum(1 for report in state.reports if report.agent == "FixerAgent")
    score -= 0.1 * retries
    return round(max(0.0, min(1.0, score)), 4)


async def run_migration_pipeline(state: MigrationState) -> MigrationState:
    from cache.manager import CacheManager
    from llm.client import LLMClient
    from memory.migration_memory import build_memory

    settings = get_settings()
    llm = LLMClient()
    cache = CacheManager()
    memory = (
        build_memory(
            settings.memory_db_path, min_similarity=settings.memory_min_similarity
        )
        if settings.memory_enabled
        else None
    )
    pipeline = Pipeline(llm, cache, memory=memory)

    try:
        return await pipeline.run(state)
    finally:
        await pipeline.aclose()
        if memory is not None:
            memory.store.close()
        await llm.close()
