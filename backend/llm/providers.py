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

The chat rate limiter is deliberately **one shared instance across every role
and model**: chat quota is billed per project, not per model, so two limiters
would each think they owned the whole budget and together burst to double it.

Embeddings get a **separate** bucket, because they are a separate quota:
Google meters ``embed_content_free_tier_requests`` (100/min) independently of
generateContent (~10 RPM). Putting them on the chat bucket throttled corpus
ingestion to a fraction of its own allowance while still 429ing, because the
chat limiter counts *calls* and the embedding quota counts *content items* — a
50-text batch cost 50 units against a budget that believed it had spent 1. So
:class:`_ResilientEmbeddings` acquires one token **per text**, from a bucket
sized in items per second.

Both limiters are per-process. Anything that runs the app with more than one
worker process multiplies the effective rate by the worker count — see the
single-worker note in ``backend/Dockerfile``.
"""

import asyncio
import logging
import re
import threading
import time
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
_embed_rate_limiter: InMemoryRateLimiter | None = None


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


def get_embed_rate_limiter(settings=None) -> InMemoryRateLimiter | None:
    """The process-wide **embedding** limiter, or None when throttling is off.

    Separate from the chat limiter because it is a separate quota measured in a
    different unit — see the module docstring. Denominated in *texts* per
    second, since :class:`_ResilientEmbeddings` takes one token per text rather
    than one per API call.
    """
    global _embed_rate_limiter
    settings = settings or get_settings()
    if settings.embed_requests_per_second <= 0:
        return None
    with _lock:
        if _embed_rate_limiter is None:
            _embed_rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.embed_requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=max(1.0, settings.embed_max_burst),
            )
            log.info(
                "Embedding rate limiter: %.2f texts/s (burst %d)",
                settings.embed_requests_per_second,
                settings.embed_max_burst,
            )
    return _embed_rate_limiter


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


#: gemini-embedding-001 returns 3072-dim vectors. Pinned explicitly on the
#: backend (``output_dimensionality``) so the width is a stated contract rather
#: than an API default that could drift, and so the zero-vector fallback in
#: :mod:`rag.embedding_service` can match it even before the first successful
#: embed. Kept in step with ``embedding_service.GEMINI_DIMENSIONS``.
GEMINI_EMBED_DIMENSIONS = 3072


#: Google reports how long to wait in two places in the same error — the
#: structured ``RetryInfo`` detail and the prose message. Either is worth far
#: more than a guess: exponential backoff from 2s against an advertised 58s wait
#: is three guaranteed failures that each cost quota and push the reset further
#: out.
_RETRY_DELAY_PATTERNS = (
    re.compile(r"['\"]retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)


def _server_retry_delay(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, or None if it did not say."""
    text = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:  # pragma: no cover — regex already constrains it
                return None
    return None


class _ResilientEmbeddings(Embeddings):
    """Rate-limited, self-retrying wrapper around a hosted embedding backend.

    Three gaps this closes, all invisible until a free-tier quota is actually hit:

    * LangChain's :class:`InMemoryRateLimiter` throttles *chat* models only — it
      is a callback the chat client invokes and has no hook into
      ``embed_documents`` / ``embed_query``. So the startup corpus ingestion fires
      embedding batches completely unthrottled and 429s almost immediately. The
      tokens are acquired here instead, before every embed.
    * The quota counts **content items, not calls**. A limiter that takes one
      token per ``embed_documents`` call lets a 50-text batch spend 50 units of a
      100/min budget while believing it spent 1 — which is why throttling to
      0.16 req/s still 429'd within four calls. One token is taken per *text*,
      from a bucket denominated in texts per second.
    * ``GoogleGenerativeAIEmbeddings`` has no ``max_retries`` of its own — the
      field does not exist on it and its ``request_options`` is inert in this
      version — so a transient 429/5xx would go straight to the caller's
      zero-vector fallback and, on the *first* batch, poison the collection at the
      wrong width. Here the call is retried first, re-acquiring tokens each
      attempt so retries respect the same quota.

    Backoff prefers the delay the server itself advertises (``RetryInfo``) over
    the exponential schedule. Retrying a quota error sooner than the reset cannot
    succeed by construction, and each premature attempt spends more of the quota
    it is waiting on.

    Any exception is retried up to ``max_retries`` rather than only status codes
    parsed out of a provider-specific error string: the bound is small, and a
    genuinely permanent error simply exhausts the retries and reaches the
    caller's graceful fallback — the same place it would have reached
    immediately.
    """

    #: Ceiling on an exponential backoff sleep, so a large ``max_retries`` cannot
    #: schedule an absurd wait.
    _MAX_BACKOFF_SEC = 30.0
    #: Higher ceiling for a delay the *server* asked for: free-tier resets are
    #: routinely 30-60s, and truncating that back to 30 reintroduces the very
    #: premature retry this exists to stop.
    _MAX_SERVER_BACKOFF_SEC = 120.0

    def __init__(
        self,
        inner: Embeddings,
        *,
        limiter: InMemoryRateLimiter | None = None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ):
        self.inner = inner
        self.rate_limiter = limiter
        self._max_retries = max(0, max_retries)
        self._base_delay = base_delay

    def _backoff(self, attempt: int, exc: Exception) -> float:
        advised = _server_retry_delay(exc)
        if advised is not None:
            # +1s so we wake up after the reset rather than exactly on it.
            return min(self._MAX_SERVER_BACKOFF_SEC, advised + 1.0)
        return min(self._MAX_BACKOFF_SEC, self._base_delay * (2 ** attempt))

    def _acquire(self, cost: int) -> None:
        """Take ``cost`` tokens — one per text the call will actually embed."""
        if self.rate_limiter is None:
            return
        for _ in range(max(1, cost)):
            self.rate_limiter.acquire(blocking=True)

    async def _aacquire(self, cost: int) -> None:
        if self.rate_limiter is None:
            return
        for _ in range(max(1, cost)):
            await self.rate_limiter.aacquire(blocking=True)

    def _call(self, fn, arg, *, cost: int = 1):
        attempt = 0
        while True:
            self._acquire(cost)
            try:
                return fn(arg)
            except Exception as exc:  # noqa: BLE001 — bounded retry, then propagate
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt, exc)
                attempt += 1
                log.warning(
                    "Embedding call failed (attempt %d/%d), backing off %.1fs: %s",
                    attempt, self._max_retries, delay, exc,
                )
                time.sleep(delay)

    async def _acall(self, fn, arg, *, cost: int = 1):
        attempt = 0
        while True:
            await self._aacquire(cost)
            try:
                return await fn(arg)
            except Exception as exc:  # noqa: BLE001 — bounded retry, then propagate
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt, exc)
                attempt += 1
                log.warning(
                    "Async embedding call failed (attempt %d/%d), backing off %.1fs: %s",
                    attempt, self._max_retries, delay, exc,
                )
                await asyncio.sleep(delay)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(self.inner.embed_documents, texts, cost=len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._call(self.inner.embed_query, text, cost=1)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._acall(
            self.inner.aembed_documents, texts, cost=len(texts)
        )

    async def aembed_query(self, text: str) -> list[float]:
        return await self._acall(self.inner.aembed_query, text, cost=1)


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

        backend = GoogleGenerativeAIEmbeddings(
            model=name if name.startswith("models/") else f"models/{name}",
            google_api_key=settings.google_api_key or None,
            # Pin the width instead of trusting the API default so it is a stated
            # contract: the zero-vector fallback downstream must match it or Chroma
            # rejects the insert ("expecting dimension 3072, got 768").
            output_dimensionality=GEMINI_EMBED_DIMENSIONS,
        )
        # Embeddings are their own metered quota, and the startup corpus
        # ingestion is the burstiest caller of it. Wrap so every embed acquires
        # one token *per text* from the embedding bucket first, and so transient
        # 429/5xx are retried — at the server's advertised delay — before the
        # fallback ever substitutes a zero vector.
        embeddings: Embeddings = _ResilientEmbeddings(
            backend,
            limiter=get_embed_rate_limiter(settings),
            max_retries=settings.llm_max_retries,
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
    global _rate_limiter, _embed_rate_limiter
    with _lock:
        _model_cache.clear()
        _embeddings_cache.clear()
        _rate_limiter = None
        _embed_rate_limiter = None
