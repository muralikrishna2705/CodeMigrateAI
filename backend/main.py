"""
CodeMigrateAI backend.

Two-agent LLM-first pipeline:
  AnalyzerAgent -> MigratorAgent

Ollama runs on the Windows host and is reached from Docker through
host.docker.internal by default.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from cache.manager import CacheManager
from config import get_settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llm.client import LLMClient
from llm.language_profiles import get_supported_profiles
from llm.streaming import sse_event_generator
from models.requests import MigrateRequest, MigrateResponse
from models.state import MigrationState
from pipeline.orchestrator import Pipeline
from rag import CachedEmbeddings, IngestionPipeline, RAGPipeline, VectorStore

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
    log.info("OLLAMA_URL : %s", settings.ollama_url)
    log.info("LLM_MODEL  : %s", settings.llm_model)
    log.info("PIPELINE   : AnalyzerAgent -> MigratorAgent")

    llm_client = LLMClient()
    alive = await llm_client.health_check()
    if alive:
        log.info("Ollama is reachable and ready")
        # Pull the chat model if it isn't present yet, so switching LLM_MODEL
        # (e.g. to a stronger coder model) works on the next startup without a
        # manual `ollama pull`.
        if settings.ollama_auto_pull:
            await llm_client.ensure_model(settings.llm_model)
            # Pull the optional fast analysis/planning model too, but only when
            # it's a distinct model — otherwise routing reuses llm_model.
            if (
                settings.fast_llm_model
                and settings.fast_llm_model != settings.llm_model
            ):
                await llm_client.ensure_model(settings.fast_llm_model)
    else:
        log.warning("Ollama is not reachable; check that Ollama is running")

    cache_manager = CacheManager()
    pipeline = Pipeline(llm_client, cache_manager)

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
            # The embedding model is separate from the chat model and is often
            # not pulled on a fresh Ollama install — without it every embed call
            # 404s and RAG silently disables itself. Pull it once on startup.
            if settings.ollama_auto_pull:
                await llm_client.ensure_model(settings.embedding_model)
            rag_embeddings = CachedEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_url,
                max_cache=settings.local_cache_max_entries,
            )
            rag_vector_store = VectorStore(rag_embeddings)
            rag_vector_store.initialize()
            ingestion = IngestionPipeline(rag_vector_store, rag_embeddings)
            rag_state["ingestion"] = ingestion
            await ingestion.run(
                [lang["id"] for lang in settings.supported_languages]
            )
            app.state.rag_pipeline = RAGPipeline(rag_vector_store, rag_embeddings)
            pipeline.registry.attach_rag_pipeline(app.state.rag_pipeline)
            log.info("RAG pipeline ready")
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
        "model": settings.llm_model,
        "fast_model": settings.fast_llm_model or settings.llm_model,
        "ollama": "connected" if alive else "unavailable",
        "prompt_composer": "ready",
        "rag": "ready" if getattr(app.state, "rag_pipeline", None) else "unavailable",
        "language_profiles": sorted(profiles),
        "language_profile_count": len(profiles),
    }


@app.get("/languages")
async def get_languages():
    return {"languages": get_settings().supported_languages}


@app.post("/migrate", response_model=MigrateResponse)
async def migrate(request: MigrateRequest):
    if not await llm_client.health_check():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {get_settings().ollama_url}. "
                f"Make sure Ollama is running and model "
                f"'{get_settings().llm_model}' is pulled."
            ),
        )

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
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {get_settings().ollama_url}. "
                f"Make sure Ollama is running and model "
                f"'{get_settings().llm_model}' is pulled."
            ),
        )

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
async def clear_cache():
    cache_manager.clear()
    return {"status": "cleared"}
