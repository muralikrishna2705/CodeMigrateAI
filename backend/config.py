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

    # RAG Pipeline (Phase 2)
    enable_rag: bool = True
    rag_top_k: int = 4
    rag_min_score: float = 0.7
    chroma_url: str = "http://chromadb:8002"
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
