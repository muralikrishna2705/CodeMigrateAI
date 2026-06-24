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
    llm_timeout_sec: float = 120.0
    llm_num_predict: int = 512
    llm_num_ctx: int = 2048
    llm_temperature: float = 0.0
    llm_num_threads: int = 8
    llm_top_p: float = 0.9

    # Pipeline
    max_code_chars: int = 50_000
    enable_semantic_analysis: bool = False

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
    enable_validation: bool = False
    validator_timeout_sec: int = 30
    enable_syntax_validation: bool = True
    enable_logic_validation: bool = False

    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 1

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
