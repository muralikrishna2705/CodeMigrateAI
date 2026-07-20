from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_url: str = "http://host.docker.internal:11434"
    llm_model: str = "deepseek-coder:1.3b"
    # Optional lighter/faster model for analysis + planning (not code generation).
    # Empty -> reuse llm_model, so routing is a no-op until an operator sets and
    # pulls a distinct model (e.g. "llama3.2:1b"), keeping default startup safe.
    fast_llm_model: str = ""
    embedding_model: str = "nomic-embed-text"
    ollama_auto_pull: bool = True  # pull missing models on startup
    llm_timeout_sec: float = 120.0
    # num_ctx must comfortably exceed the composed prompt (~1500+ tokens with
    # RAG/planner context) plus num_predict, otherwise Ollama silently truncates
    # the prompt/response and JSON output comes back malformed.
    llm_num_predict: int = 2048
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.0
    llm_num_threads: int = 8
    llm_top_p: float = 0.9

    # Shared truncation limit for LLM input
    max_llm_code_chars: int = 4000

    # Pipeline
    max_code_chars: int = 50_000
    enable_semantic_analysis: bool = True

    # Redis Cache
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True
    redis_ttl_seconds: int = 86400
    redis_max_entries: int = 10000

    # Local Cache (fallback)
    local_cache_max_entries: int = 500

    # Validators
    validator_url: str = "http://validator:8000"
    enable_validation: bool = True
    validator_timeout_sec: int = 30
    enable_syntax_validation: bool = True
    enable_logic_validation: bool = False

    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 1

    # LangGraph (Phase 3)
    max_retries: int = 2
    # Dynamic routing: when on, the DispatcherAgent asks the fast model whether a
    # migration needs deep analysis (falling back to the complexity rule on any
    # failure). Off by default so routing stays deterministic and offline-safe.
    dispatcher_llm_routing: bool = False
    # Dynamic orchestration (Dimension 4)
    #
    # The OrchestratorAgent decomposes a migration into sub-tasks and the
    # `parallel` node fans the independent ones out concurrently (deep analysis
    # and RAG retrieval both depend only on AnalyzerAgent's metrics, never on
    # each other), then a merge node folds the branches back into one state.
    # On by default because the decomposition is rule-based and deterministic:
    # the orchestrator only plans a parallel batch when it finds >= 2 genuinely
    # independent tasks, and anything else routes down the untouched sequential
    # path. Set False to force the original linear flow.
    parallel_enabled: bool = True
    # Concurrency ceiling for the fan-out. Each parallel branch is a subgraph
    # invocation that can make LLM calls, so this bounds simultaneous load on
    # the Ollama endpoint rather than just CPU.
    max_parallel_tasks: int = 4
    # When on, the OrchestratorAgent asks the fast model to decompose the
    # migration instead of applying its rules (falling back to the rules on any
    # failure), mirroring dispatcher_llm_routing. Off by default so the
    # decomposition stays deterministic and offline-safe.
    orchestrator_llm_planning: bool = False

    # Adaptive RAG: when the MigratorAgent emits at least this many ungrounded
    # imports, it enqueues them as targeted retrieval queries and the graph loops
    # migrate -> retrieve -> plan -> migrate, bounded by max_reretrievals (kept
    # to 1 by default so a single corrective pass never spirals).
    rag_grounding_reretrieval_threshold: int = 1
    max_reretrievals: int = 1

    # Agent reflection + decision-making (Dimension 3)
    #
    # When on, agents self-critique their own output (Reflexion pattern): the
    # PlannerAgent reflects on its plan and refines it, the MigratorAgent reflects
    # on the migrated code's correctness and regenerates low-confidence output, and
    # a dedicated ReflectorAgent runs as a `reflect` graph node between migrate and
    # validate, routing low-confidence code back to migrate with actionable
    # feedback (bounded by max_reflections, mirroring the fix loop's max_retries).
    # Off by default so decisions stay deterministic/offline-safe; every reflection
    # degrades to "pass" when no LLM is wired, so switching this on can improve
    # quality but never blocks a migration.
    enable_reflection: bool = False
    # Graph-level regeneration budget: how many times reflect -> migrate may loop
    # for a single migration. Kept to 1 so a single corrective pass never spirals.
    max_reflections: int = 1
    # Confidence floor the in-agent reflection uses to decide an in-place refine
    # (the model's own 0.0-1.0 self-rating). The graph ReflectorAgent routes on the
    # model's explicit recommendation instead, so this only gates internal refines.
    reflection_min_confidence: float = 0.6

    # Agent Tools
    #
    # Agents call tools on demand (see agents/tools/). `tools_enabled=False` is
    # the kill switch: build_registry returns an empty registry and every agent
    # falls back to its pre-tool behaviour.
    tools_enabled: bool = True
    tool_timeout_sec: float = 20.0
    tool_vector_db_enabled: bool = True
    tool_code_metrics_enabled: bool = True
    tool_syntax_check_enabled: bool = True
    tool_source_reader_enabled: bool = True
    # Off by default: the only tool that reaches the public internet, and the
    # DuckDuckGo HTML endpoint it uses rate-limits datacenter IPs.
    tool_web_search_enabled: bool = False
    # Off by default: the memory collection is empty until migrations have run.
    tool_semantic_search_enabled: bool = False

    # Persistent memory (Dimension 5)
    #
    # Two stores, deliberately separate:
    #
    #   memory_enabled          -> the SQLite system of record (MemoryStore /
    #                              MigrationMemory / PatternStore). On by default:
    #                              it is a local file, needs no network or model,
    #                              and starts empty, so the worst case is zero
    #                              recall rather than a failure.
    #   enable_migration_memory -> the Chroma semantic leg (SemanticMigrationMemory,
    #                              backs the semantic_search tool). Off by default:
    #                              it needs a live embedding service.
    #
    # With both on, MigrationMemory queries the trie/cosine leg and merges the
    # semantic hits; with only the first, recall stays fully offline.
    memory_enabled: bool = True
    memory_db_path: str = "memory/codemigrate.db"
    # Minimum lexical cosine for a past migration to count as a hit at all. Below
    # this the "precedent" is noise, and injecting it costs prompt budget while
    # grounding the model in unrelated code.
    memory_min_similarity: float = 0.25
    # Bloom filter sizing for PatternStore. Saturating past capacity degrades the
    # false positive rate (never correctness — hits are confirmed against SQLite).
    pattern_bloom_capacity: int = 10_000
    pattern_bloom_error_rate: float = 0.01

    # LangGraph checkpointing: persists in-run graph state per thread_id so an
    # interrupted migration can resume. Uses AsyncSqliteSaver (the sync
    # SqliteSaver raises on the astream path the pipeline uses) and degrades to
    # MemorySaver when no event loop is running, e.g. tests.
    checkpointer_enabled: bool = True
    checkpoint_db_path: str = "memory/checkpoints.db"

    # Cross-session migration memory (backs the semantic_search tool). Recorded
    # by the ObserverAgent for successful, validated migrations only.
    enable_migration_memory: bool = False
    memory_top_k: int = 3
    # Higher than rag_min_score: a weak "similar past migration" is worse than
    # none, since it grounds new code in a precedent that doesn't really apply.
    memory_min_score: float = 0.8

    # Retriever tool loop: when on, the RetrieverAgent asks the LLM which tool to
    # call and with what query, instead of running one heuristic retrieval pass.
    # Off by default — deepseek-coder:1.3b is not tool-call trained, so selection
    # is prompt-driven and unreliable; any failure falls back to the heuristic
    # pass, which is exactly the pre-tool behaviour.
    retriever_tool_loop: bool = False
    retriever_max_tool_calls: int = 3

    # RAG Pipeline (Phase 2)
    enable_rag: bool = True
    rag_top_k: int = 4
    rag_min_score: float = 0.7
    # Grounding: the retrieval query is built from the actual source code's
    # imports/APIs/type names (not just the language pair) so retrieved examples
    # are code-specific. Cap how many symbols and how much of a code excerpt feed
    # the query so a large blob can't wash out the symbol signal.
    rag_query_max_symbols: int = 12
    rag_query_code_chars: int = 600
    # Prefer reference examples written in the TARGET language (the ones that
    # actually ground target-language API usage). Falls back to an unfiltered
    # search when the target-language corpus yields nothing.
    rag_filter_by_target_language: bool = True
    # Version-aware retrieval: a phased ladder retrieves target-VERSION docs
    # first, broadens to any-version target-language docs if that is sparse, then
    # falls back to an unfiltered search. Docs tagged with the wildcard version
    # (unversioned corpus content) always match the version leg, so this degrades
    # gracefully to language-only behavior until versioned docs are ingested.
    rag_filter_by_target_version: bool = True
    rag_version_wildcard: str = "any"
    # Metadata-weighted ranking: after retrieval, boost each hit's base relevance
    # by small authority signals so exact-version / official / migration docs win
    # ties without overriding a genuinely more relevant (higher-cosine) example.
    # Kept small relative to cosine scores (~0.7-1.0) on purpose.
    rag_rank_weight_version: float = 0.15
    rag_rank_weight_official: float = 0.10
    rag_rank_weight_migration: float = 0.10
    # Doc types treated as migration authority for ranking (guides/notes that
    # describe how to move between versions or flag removed APIs).
    rag_migration_doc_types: list[str] = [
        "migration-guide",
        "release-notes",
        "deprecation",
    ]
    # Hybrid retrieval: combine the dense vector leg with a keyword leg (exact
    # symbol matches via Chroma's where_document) merged by reciprocal rank
    # fusion. Improves recall for exact API/symbol names that dense embeddings
    # miss. Degrades gracefully to vector-only if the store lacks keyword search.
    rag_hybrid_enabled: bool = True
    rag_rrf_k: int = 60

    # Anti-hallucination: flag library imports in migrated code that are not
    # grounded by the source, retrieved context, or target stdlib (advisory).
    enable_grounding_check: bool = True
    # When the web_search tool is enabled, how many flagged imports the
    # MigratorAgent checks against official docs before giving up. Each check is
    # a network round trip, so this is deliberately small.
    migrator_max_import_checks: int = 3

    # Optional LLM-based query expansion: reformulate the extracted code signals
    # into a targeted natural-language retrieval query via the fast model. Off by
    # default (adds one LLM call per retrieval); falls back to the concatenated
    # signal query on any error.
    rag_query_expansion: bool = False

    # Agentic RAG strategies (Dimension 2)
    #
    # Which retrieval strategy enrich_prompt / the VectorDBTool use. Options:
    #   single_hop (default) · hyde · multi_query · multi_hop ·
    #   contextual_compression · parent_document · corrective · self_rag
    # Default is single_hop = the original one-shot behaviour: offline-safe and
    # free of per-retrieval LLM calls. The LLM-driven strategies activate only
    # when a client is wired, and every one degrades to single_hop on any failure,
    # so switching this can improve grounding but never break a migration.
    rag_strategy: str = "single_hop"
    # HyDE: retrieve against an LLM-written hypothetical target-language answer.
    rag_hyde_enabled: bool = True
    # Multi-Query: how many LLM-generated query paraphrases to fan out over.
    rag_multi_query_count: int = 5
    # Multi-hop: bounds on the query-graph BFS (nodes expanded, graph depth).
    rag_multi_hop_max_subqueries: int = 4
    rag_multi_hop_max_depth: int = 2
    # Contextual compression: LLM-extract only query-relevant lines per doc. Off by
    # default — one LLM call per retrieved doc.
    rag_compression_enabled: bool = False
    # CRAG: min relevance for the offline grade heuristic; web fallback reaches the
    # live internet so it stays opt-in (like tool_web_search_enabled).
    rag_crag_relevance_threshold: float = 0.5
    rag_crag_web_fallback: bool = False
    # Self-RAG: honour the model's "skip retrieval" decision. Off by default so the
    # anti-hallucination retrieval always runs; reflection filtering is unaffected.
    rag_self_rag_enabled: bool = False

    # Web Document Fetching (Phase 2)
    enable_web_docs: bool = True
    web_docs_refresh_days: int = 7

    # Supported Languages (drives frontend dropdowns)
    supported_languages: list[dict] = [
        {
            "id": "java",
            "name": "Java",
            "versions": ["7", "8", "11", "17", "21"],
        },
        {
            "id": "python",
            "name": "Python",
            "versions": ["2.7", "3.8", "3.10", "3.12"],
        },
        {
            "id": "javascript",
            "name": "JavaScript",
            "versions": ["ES5", "ES6", "ES2020", "ES2022"],
        },
        {
            "id": "typescript",
            "name": "TypeScript",
            "versions": ["3.x", "4.x", "5.x"],
        },
        {
            "id": "csharp",
            "name": "C#",
            "versions": ["6", "8", "10", "12"],
        },
        {
            "id": "go",
            "name": "Go",
            "versions": ["1.18", "1.20", "1.22"],
        },
        {
            "id": "kotlin",
            "name": "Kotlin",
            "versions": ["1.7", "1.9", "2.0"],
        },
        {
            "id": "rust",
            "name": "Rust",
            "versions": ["1.70", "1.80"],
        },
        {
            "id": "cpp",
            "name": "C++",
            "versions": ["14", "17", "20", "23"],
        },
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
