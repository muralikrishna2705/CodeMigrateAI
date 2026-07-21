"""Provider-agnostic chat-model and embedding construction.

Everything that needs a model asks here instead of instantiating a client, so
swapping Gemini for Ollama (or anything else LangChain speaks) is one setting
rather than an edit in every agent.

Two roles exist, because the work splits cleanly in two:

``main``
    Code generation. Quality matters more than speed; gets the stronger model.
``fast``
    Routing, grading, decomposition, query reformulation — short structured
    answers on a hot path. Gets the cheap model with thinking disabled, since a
    reasoning budget on a yes/no routing call is latency spent for nothing.

**Why the provider is always passed explicitly**: ``init_chat_model("gemini-…")``
infers ``model_provider="google_vertexai"`` (with a DeprecationWarning) rather
than the AI Studio / Gemini API path this project uses. Leaving inference to
guess would silently route to a different backend that needs different
credentials, so ``model_provider`` is never omitted.

The rate limiter is deliberately **one shared instance across every role and
model**: API quota is billed per project, not per model, so two limiters would
each think they owned the whole budget and together burst to double it.
"""

import logging
import threading
from typing import Literal

from config import get_settings
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

log = logging.getLogger("CodeMigrateAI.Providers")

Role = Literal["main", "fast"]

#: Per-provider model defaults, used when the corresponding setting is left
#: empty. Keeping them here (rather than as literal defaults in Settings) is what
#: lets one `llm_provider` switch carry the whole model triple with it — an
#: Ollama run must not inherit `gemini-2.5-flash` as its model name.
# Verified against the live generateContent endpoint, not the /models listing:
# that listing still advertises the 2.5 flash models, which 404 for accounts
# created after their retirement ("no longer available to new users"). A model
# id is only real if a call to it succeeds.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "google_genai": {
        "main": "gemini-3.5-flash",
        "fast": "gemini-3.1-flash-lite",
        "embed": "gemini-embedding-001",
    },
    "ollama": {
        "main": "deepseek-coder:1.3b",
        "fast": "llama3.2:3b",
        "embed": "nomic-embed-text",
    },
}

# Providers whose models run on someone else's hardware. Used to decide whether
# "ensure this model is pulled locally" means anything (it doesn't) and whether
# an API key is required (it is).
HOSTED_PROVIDERS = frozenset({"google_genai", "google_vertexai", "openai", "anthropic", "groq"})

_lock = threading.Lock()
_model_cache: dict[tuple, BaseChatModel] = {}
_embeddings_cache: dict[tuple, Embeddings] = {}
_rate_limiter: InMemoryRateLimiter | None = None


def resolve_model_name(role: Role = "main", settings=None) -> str:
    """The concrete model id for ``role`` under the configured provider.

    An explicit setting always wins; an empty one falls back to the provider's
    default. Exposed separately from :func:`get_chat_model` because /health and
    the startup logs want to report the name without building a client.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider
    configured = settings.llm_model if role == "main" else settings.fast_llm_model
    if configured:
        return configured
    defaults = DEFAULT_MODELS.get(provider, {})
    name = defaults.get(role) or defaults.get("main", "")
    if not name:
        raise ValueError(
            f"No model configured for provider {provider!r} role {role!r}. "
            f"Set llm_model (and fast_llm_model) explicitly."
        )
    return name


def resolve_embedding_model(settings=None) -> str:
    settings = settings or get_settings()
    if settings.embedding_model:
        return settings.embedding_model
    provider = settings.embedding_provider or settings.llm_provider
    return DEFAULT_MODELS.get(provider, {}).get("embed", "")


def get_rate_limiter(settings=None) -> InMemoryRateLimiter | None:
    """The process-wide limiter, or None when throttling is disabled.

    Hosted free tiers are measured in requests per *minute* and this graph makes
    8-15 model calls per migration, so without this the parallel fan-out 429s
    long before it saturates anything. ``max_bucket_size`` allows a short burst
    (the fan-out) while the steady-state rate still holds.
    """
    global _rate_limiter
    settings = settings or get_settings()
    if settings.llm_requests_per_second <= 0:
        return None
    with _lock:
        if _rate_limiter is None:
            _rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.llm_requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=max(1.0, settings.llm_max_burst),
            )
            log.info(
                "Rate limiter: %.2f req/s (burst %d)",
                settings.llm_requests_per_second,
                settings.llm_max_burst,
            )
    return _rate_limiter


def _provider_kwargs(provider: str, role: Role, json_mode: bool, settings) -> dict:
    """Translate this project's settings into one provider's constructor kwargs.

    JSON mode is the reason this exists: every backend spells "constrain the
    decoder to valid JSON" differently (``response_mime_type`` on Gemini,
    ``format`` on Ollama), and it is a constructor field rather than a call-time
    argument on both, which is why models are cached per ``json_mode``.
    """
    kwargs: dict = {"temperature": settings.llm_temperature}

    if provider == "google_genai":
        if settings.google_api_key:
            kwargs["api_key"] = settings.google_api_key
        kwargs["max_output_tokens"] = settings.llm_max_tokens
        kwargs["timeout"] = settings.llm_timeout_sec
        kwargs["max_retries"] = settings.llm_max_retries
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        # Gemini 2.5+ reasons before answering by default. That is worth paying
        # for on code generation and pure overhead on a routing or grading call,
        # so the fast role opts out entirely.
        if role == "fast" and settings.llm_fast_thinking_budget >= 0:
            kwargs["thinking_budget"] = settings.llm_fast_thinking_budget

    elif provider == "ollama":
        kwargs["base_url"] = settings.ollama_url
        kwargs["num_predict"] = settings.llm_max_tokens
        kwargs["num_ctx"] = settings.llm_num_ctx
        kwargs["top_p"] = settings.llm_top_p
        if json_mode:
            kwargs["format"] = "json"

    else:
        # Unknown provider: pass only what every chat model accepts and let
        # LangChain's own defaults cover the rest, rather than guessing at
        # provider-specific field names that would raise on construction.
        if settings.google_api_key and provider in HOSTED_PROVIDERS:
            kwargs["api_key"] = settings.google_api_key

    return kwargs


def get_chat_model(
    role: Role = "main",
    *,
    json_mode: bool = False,
    model: str | None = None,
    settings=None,
) -> BaseChatModel:
    """Build (or return a cached) chat model for ``role``.

    ``model`` overrides the role's configured name — used by call sites that
    already know which model they want, e.g. the LLMClient adapter honouring an
    explicit ``model=`` argument.

    Models are cached because construction parses schemas and opens a transport;
    rebuilding one per node call showed up as measurable overhead in a graph that
    runs a dozen of them. Cache identity covers everything that changes the
    client's behaviour, so a settings change can't be masked by a stale entry.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider
    name = model or resolve_model_name(role, settings)

    kwargs = _provider_kwargs(provider, role, json_mode, settings)
    # Key on the *resolved kwargs*, not on a hand-listed subset of settings.
    # Anything that changes how the client behaves — api key, base url, thinking
    # budget, token ceiling — is already in there, so no settings change can be
    # masked by a stale entry, and adding a provider kwarg later can't silently
    # forget to extend the key.
    key = (provider, name, role, json_mode, repr(sorted(kwargs.items())))

    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter(settings)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter

    chat = init_chat_model(name, model_provider=provider, **kwargs)
    with _lock:
        _model_cache[key] = chat
    log.info(
        "Chat model ready: %s/%s (role=%s%s)",
        provider,
        name,
        role,
        ", json" if json_mode else "",
    )
    return chat


def get_embeddings(settings=None) -> Embeddings:
    """Build (or return a cached) embedding service for the configured provider.

    Separate from the chat provider on purpose: embedding a corpus locally while
    generating with a hosted model is a reasonable split, and forcing them to
    match would make that impossible.
    """
    settings = settings or get_settings()
    provider = settings.embedding_provider or settings.llm_provider
    name = resolve_embedding_model(settings)
    key = (provider, name)

    cached = _embeddings_cache.get(key)
    if cached is not None:
        return cached

    if provider == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings: Embeddings = GoogleGenerativeAIEmbeddings(
            model=name if name.startswith("models/") else f"models/{name}",
            google_api_key=settings.google_api_key or None,
        )
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=name, base_url=settings.ollama_url)
    else:
        raise ValueError(f"Unsupported embedding_provider {provider!r}")

    with _lock:
        _embeddings_cache[key] = embeddings
    log.info("Embeddings ready: %s/%s", provider, name)
    return embeddings


def is_hosted(settings=None) -> bool:
    """Whether the chat provider runs remotely (no local model to pull)."""
    settings = settings or get_settings()
    return settings.llm_provider in HOSTED_PROVIDERS


def reset_models() -> None:
    """Drop every cached model, limiter included.

    Tests mutate settings between cases; without this they would keep getting a
    client built from the previous case's configuration.
    """
    global _rate_limiter
    with _lock:
        _model_cache.clear()
        _embeddings_cache.clear()
        _rate_limiter = None
