"""
CodeMigrateAI backend.

A compiled LangGraph workflow (see graph/migration_graph.py) runs analysis,
dynamic dispatch, orchestration, agentic RAG retrieval, planning, migration,
reflection, validation, and the fix retry loop.

The chat model is provider-agnostic (llm/providers.py). Gemini is the default;
Ollama is selectable via LLM_PROVIDER for local inference, in which case it runs
on the host and is reached from Docker through host.docker.internal.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from cache.keys import key_prefix
from cache.manager import CacheManager
from config import ENV_FILE, get_settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from graph import nodes as graph_nodes
from llm import providers
from llm.client import LLMClient
from llm.language_profiles import get_supported_profiles
from llm.streaming import sse_event_generator
from models.requests import MigrateRequest, MigrateResponse
from models.state import MigrationState
from pipeline.orchestrator import Pipeline
from memory.migration_memory import build_memory
from rag import (
    CachedEmbeddings,
    IngestionPipeline,
    RAGPipeline,
    SemanticMigrationMemory,
    VectorStore,
)
from runtime.agent_observer import ObserverAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CodeMigrateAI")

llm_client: LLMClient | None = None
pipeline: Pipeline | None = None
cache_manager: CacheManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, pipeline, cache_manager

    settings = get_settings()
    main_model = providers.resolve_model_name("main", settings)
    fast_model = providers.resolve_model_name("fast", settings)
    hosted = providers.is_hosted(settings)
    log.info("PROVIDER   : %s%s", settings.llm_provider, " (hosted)" if hosted else "")
    log.info("LLM_MODEL  : %s | fast: %s", main_model, fast_model)
    log.info("EMBEDDINGS : %s/%s", settings.embedding_provider,
             providers.resolve_embedding_model(settings))
    if settings.llm_requests_per_second > 0:
        log.info("RATE LIMIT : %.2f req/s", settings.llm_requests_per_second)
    if settings.embed_requests_per_second > 0:
        # Reported separately because it is a separate quota, in a different
        # unit — texts per second, not calls per second.
        log.info("EMBED LIMIT: %.2f texts/s", settings.embed_requests_per_second)

    llm_client = LLMClient()
    alive = await llm_client.health_check()
    if alive:
        log.info("LLM provider is configured and ready")
    elif hosted:
        log.warning(
            "No API key configured for %s; set GOOGLE_API_KEY to enable the LLM",
            settings.llm_provider,
        )
    else:
        log.warning("Ollama is not reachable; check that Ollama is running")

    # Only a local provider has models to provision. Hosted ones resolve
    # server-side, so ensure_model is a no-op there.
    if alive and not hosted and settings.ollama_auto_pull:
        await llm_client.ensure_model(main_model)
        if fast_model != main_model:
            await llm_client.ensure_model(fast_model)

    cache_manager = CacheManager()

    # Persistent memory (Dimension 5). Built before the pipeline and outside the
    # RAG task on purpose: it is a local SQLite file with no model or network
    # dependency, so it is ready on the first request rather than whenever
    # background ingestion finishes. The semantic leg attaches later if it comes
    # up. build_memory never raises — a bad path yields None and no recall.
    app.state.persistent_memory = None
    if settings.memory_enabled:
        app.state.persistent_memory = build_memory(
            settings.memory_db_path,
            min_similarity=settings.memory_min_similarity,
        )
        if app.state.persistent_memory is not None:
            log.info(
                "Persistent memory ready (%d past migrations)",
                app.state.persistent_memory.store.count("migrations"),
            )

    pipeline = Pipeline(llm_client, cache_manager, memory=app.state.persistent_memory)

    # RAG pipeline (Phase 2): embed reference docs into the vector store and
    # expose retrieval to the RetrieverAgent. This runs as a BACKGROUND task so
    # that embedding ingestion (which can take minutes) never blocks app
    # startup — the API serves /health and /migrate immediately, and the
    # RetrieverAgent simply skips RAG until it becomes ready. Wrapped so that an
    # unreachable embedding model or vector store can never break the app.
    app.state.rag_pipeline = None
    rag_state = {"ingestion": None}

    async def _init_rag() -> None:
        try:
            # A local embedding model is separate from the chat model and is
            # often not pulled on a fresh Ollama install — without it every
            # embed call 404s and RAG silently disables itself. Pull it once on
            # startup; a hosted embedding provider needs no provisioning.
            if settings.embedding_provider == "ollama" and settings.ollama_auto_pull:
                await llm_client.ensure_model(
                    providers.resolve_embedding_model(settings)
                )
            rag_embeddings = CachedEmbeddings(
                providers.get_embeddings(settings),
                max_cache=settings.local_cache_max_entries,
            )
            rag_vector_store = VectorStore(rag_embeddings)
            rag_vector_store.initialize()
            if settings.rag_ingest_on_startup:
                ingestion = IngestionPipeline(rag_vector_store, rag_embeddings)
                rag_state["ingestion"] = ingestion
                await ingestion.run(
                    [lang["id"] for lang in settings.supported_languages]
                )
            else:
                # Serving a prebuilt index (backend/scripts/build_index.py).
                # Ingestion is the only thing here that spends embedding quota,
                # so skipping it makes a restart free.
                log.info(
                    "Startup ingestion disabled; serving prebuilt index "
                    "(%d chunks)",
                    rag_vector_store.count(),
                )
            # Pass the LLM client so the agentic RAG strategies (HyDE, CRAG,
            # multi-query, …) can reason; single-hop retrieval never touches it.
            app.state.rag_pipeline = RAGPipeline(
                rag_vector_store, rag_embeddings, llm_client
            )
            # The graph builds a fresh RetrieverAgent per call, so the pipeline is
            # threaded through module state (like the LLM client) rather than an
            # instance — see graph/nodes.set_rag_pipeline. Registering it also
            # rebuilds the tool registry, which is what brings VectorDBTool up.
            graph_nodes.set_rag_pipeline(app.state.rag_pipeline)
            log.info("RAG pipeline ready")

            # Cross-session migration memory shares the embedding service but
            # lives in its own Chroma collection, so past migrations can never
            # outrank official documentation during reference retrieval.
            if settings.enable_migration_memory:
                memory = SemanticMigrationMemory(rag_embeddings)
                memory.initialize()
                app.state.migration_memory = memory
                # The SemanticSearchTool talks to the Chroma store directly, so
                # the graph keeps receiving that object; the SQL facade gets it
                # as its semantic leg for merged recall.
                graph_nodes.set_migration_memory(memory)
                if app.state.persistent_memory is not None:
                    app.state.persistent_memory.attach_semantic(memory)
                log.info("Semantic migration memory ready (%d entries)", memory.count())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("RAG initialization failed; continuing without RAG: %s", exc)

    rag_task: asyncio.Task | None = None
    if settings.enable_rag:
        rag_task = asyncio.create_task(_init_rag())
    else:
        log.info("RAG disabled via settings")

    yield

    if rag_task is not None and not rag_task.done():
        rag_task.cancel()
        try:
            await rag_task
        except asyncio.CancelledError:
            pass
    ingestion = rag_state.get("ingestion")
    if ingestion is not None:
        await ingestion.close()
    # Release the checkpointer's aiosqlite connection and the memory store's
    # SQLite handle before the loop goes away.
    if pipeline is not None:
        await pipeline.aclose()
    if app.state.persistent_memory is not None:
        app.state.persistent_memory.store.close()
    await llm_client.close()
    log.info("Shutdown complete")


app = FastAPI(
    title="CodeMigrateAI",
    description="AI-driven code migration platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    alive = await llm_client.health_check()
    profiles = get_supported_profiles()
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": providers.resolve_model_name("main", settings),
        "fast_model": providers.resolve_model_name("fast", settings),
        "llm": "ready" if alive else "unavailable",
        # Retained under its original key so existing clients keep parsing; it
        # now reports the configured provider, whichever that is.
        "ollama": "connected" if alive else "unavailable",
        "prompt_composer": "ready",
        "rag": "ready" if getattr(app.state, "rag_pipeline", None) else "unavailable",
        "language_profiles": sorted(profiles),
        "language_profile_count": len(profiles),
        "metrics": ObserverAgent.get_metrics(),
    }


@app.get("/languages")
async def get_languages():
    return {"languages": get_settings().supported_languages}


def _llm_unavailable_detail() -> str:
    """The 503 body when the configured provider can't serve a migration.

    Provider-specific because the fix is: a hosted provider needs a key, a local
    one needs a running daemon and a pulled model. A single generic message
    would send the operator looking in the wrong place.
    """
    settings = get_settings()
    if providers.is_hosted(settings):
        return (
            f"No API key configured for provider '{settings.llm_provider}'. "
            f"Set GOOGLE_API_KEY in the environment or in {ENV_FILE}."
        )
    return (
        f"Ollama is not reachable at {settings.ollama_url}. Make sure Ollama is "
        f"running and model '{providers.resolve_model_name('main', settings)}' "
        "is pulled."
    )


@app.post("/migrate", response_model=MigrateResponse)
async def migrate(request: MigrateRequest):
    if not await llm_client.health_check():
        raise HTTPException(status_code=503, detail=_llm_unavailable_detail())

    state = MigrationState(
        source_code=request.source_code,
        source_language=request.source_language,
        source_version=request.source_version,
        target_language=request.target_language,
        target_version=request.target_version,
    )

    try:
        final_state = await pipeline.run(state)
    except Exception as exc:
        log.exception("Pipeline crashed")
        raise HTTPException(status_code=500, detail=str(exc))

    return MigrateResponse(
        success=bool(final_state.migrated_code) and not final_state.errors,
        migrated_code=final_state.migrated_code,
        inline_plan=final_state.inline_plan,
        migration_type=final_state.migration_type.value,
        source_language=final_state.source_language,
        source_version=final_state.source_version,
        target_language=final_state.target_language,
        target_version=final_state.target_version,
        reports=[r.model_dump() for r in final_state.reports],
        errors=final_state.errors,
        agents_completed=final_state.agents_done,
        validation_result=final_state.validation_result,
    )


@app.post("/migrate/stream")
async def migrate_stream(request: MigrateRequest):
    if not await llm_client.health_check():
        raise HTTPException(status_code=503, detail=_llm_unavailable_detail())

    state = MigrationState(
        source_code=request.source_code,
        source_language=request.source_language,
        source_version=request.source_version,
        target_language=request.target_language,
        target_version=request.target_version,
    )

    queue = asyncio.Queue()
    state._stream_queue = queue

    async def event_generator():
        async for event in sse_event_generator(queue):
            yield event

    async def run_pipeline():
        try:
            await pipeline.run(state)
        except Exception as exc:
            log.exception("Streaming pipeline error")
            await queue.put({"type": "error", "message": str(exc)})

    asyncio.create_task(run_pipeline())

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/cache/stats")
async def cache_stats():
    return cache_manager.stats()


@app.post("/cache/clear")
async def clear_cache(
    source_language: str = "",
    source_version: str = "",
    target_language: str = "",
    target_version: str = "",
):
    """Clear the migration cache, optionally scoped to one migration family.

    Re-indexing the RAG corpus for a single target, or fixing one language
    profile, only invalidates the migrations that touch it — dropping the whole
    cache would force every unrelated migration to be recomputed too. The
    filters are hierarchical: a source language alone covers every target under
    it, and omitting all of them clears everything.
    """
    if not source_language:
        cache_manager.clear()
        return {"status": "cleared", "scope": "all"}

    removed = cache_manager.invalidate(
        source_language, source_version, target_language, target_version
    )
    return {
        "status": "cleared",
        "scope": key_prefix(
            source_language, source_version, target_language, target_version
        ),
        "local_entries_removed": removed,
    }
