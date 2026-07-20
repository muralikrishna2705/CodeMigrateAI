import logging
from datetime import datetime
from typing import Optional

from cache.keys import generate_key
from cache.manager import CacheManager
from config import get_settings
from graph import nodes as graph_nodes
from graph.migration_graph import build_migration_graph
from llm.streaming import SSEStreamHandler
from models.state import MigrationState

log = logging.getLogger("CodeMigrateAI.Pipeline")


class Pipeline:
    def __init__(
        self,
        llm_client,
        cache_manager: Optional[CacheManager] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache_manager
        self._graph = build_migration_graph()
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

        # Everything — analysis, dynamic dispatch, RAG retrieval, planning,
        # migration, validation (offline + optional external service), the fix
        # retry loop, and observability — runs inside the compiled LangGraph now
        # (see graph/migration_graph.py). There is no linear prelude anymore.
        state = await self._run_graph(state, stream_handler)

        state.completed_at = datetime.utcnow()

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

        final_graph_state = graph_state
        try:
            async for chunk in self._graph.astream(graph_state, stream_mode="updates"):
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


async def run_migration_pipeline(state: MigrationState) -> MigrationState:
    from cache.manager import CacheManager
    from llm.client import LLMClient

    llm = LLMClient()
    cache = CacheManager()
    pipeline = Pipeline(llm, cache)

    try:
        return await pipeline.run(state)
    finally:
        await llm.close()
