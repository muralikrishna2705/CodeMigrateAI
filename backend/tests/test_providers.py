"""Tests for the provider-agnostic model layer (llm/providers.py).

These cover the seams where a mistake is silent rather than loud: a provider
switch that half-applies, a cache that serves a client built from stale
settings, or a rate limiter that two roles each think they own.
"""

import pytest
from config import Settings
from langchain_core.rate_limiters import InMemoryRateLimiter
from llm import providers


def _settings(**overrides) -> Settings:
    # Pin the model fields empty by default so provider defaults are what's under
    # test, regardless of any LLM_MODEL in the environment or .env.
    base = {"llm_model": "", "fast_llm_model": "", "embedding_model": ""}
    base.update(overrides)
    return Settings(**base)


class TestModelResolution:
    def test_roles_resolve_to_distinct_provider_defaults(self):
        # Asserted against DEFAULT_MODELS rather than against literal model ids.
        # Pinning the strings here bought nothing and aged badly: when Google
        # retired the 2.5 flash models for new accounts this test kept passing
        # while every real call 404'd, which is the exact inversion of what a
        # test is for.
        s = _settings(llm_provider="google_genai")
        defaults = providers.DEFAULT_MODELS["google_genai"]
        assert providers.resolve_model_name("main", s) == defaults["main"]
        assert providers.resolve_model_name("fast", s) == defaults["fast"]
        assert defaults["main"] != defaults["fast"], (
            "the fast role exists to be cheaper than the main one"
        )

    def test_explicit_setting_beats_the_provider_default(self):
        s = _settings(llm_provider="google_genai", llm_model="some-pinned-model")
        assert providers.resolve_model_name("main", s) == "some-pinned-model"

    def test_switching_provider_carries_the_whole_model_set(self):
        # The failure this guards: an Ollama run inheriting a Gemini model id,
        # which only surfaces as a 404 at call time.
        s = _settings(llm_provider="ollama")
        assert providers.resolve_model_name("main", s) == "deepseek-coder:1.3b"
        assert "gemini" not in providers.resolve_model_name("fast", s)

    def test_unknown_provider_without_an_explicit_model_raises(self):
        # Better to fail at construction than to send an empty model name to an
        # API and get back an opaque error.
        s = _settings(llm_provider="not-a-provider")
        with pytest.raises(ValueError, match="No model configured"):
            providers.resolve_model_name("main", s)

    def test_embedding_provider_is_independent_of_chat_provider(self):
        s = _settings(llm_provider="google_genai", embedding_provider="ollama")
        assert providers.resolve_embedding_model(s) == "nomic-embed-text"


class TestProviderKwargs:
    def test_gemini_json_mode_sets_response_mime_type(self):
        s = _settings(llm_provider="google_genai", google_api_key="k")
        model = providers.get_chat_model("main", json_mode=True, settings=s)
        assert model.response_mime_type == "application/json"

    def test_ollama_json_mode_sets_format(self):
        s = _settings(llm_provider="ollama")
        assert providers.get_chat_model("main", json_mode=True, settings=s).format == "json"

    def test_fast_role_disables_thinking_on_gemini(self):
        # Reasoning before a yes/no routing answer is latency spent for nothing;
        # this is one of the larger per-call savings available.
        s = _settings(llm_provider="google_genai", google_api_key="k")
        assert providers.get_chat_model("fast", settings=s).thinking_budget == 0
        assert providers.get_chat_model("main", settings=s).thinking_budget is None

    def test_negative_thinking_budget_leaves_the_provider_default(self):
        s = _settings(
            llm_provider="google_genai", google_api_key="k", llm_fast_thinking_budget=-1
        )
        assert providers.get_chat_model("fast", settings=s).thinking_budget is None


class TestModelCache:
    def test_same_request_returns_the_same_instance(self):
        s = _settings(llm_provider="ollama")
        assert providers.get_chat_model("main", settings=s) is providers.get_chat_model(
            "main", settings=s
        )

    def test_json_mode_is_cached_separately(self):
        s = _settings(llm_provider="ollama")
        plain = providers.get_chat_model("main", json_mode=False, settings=s)
        json_mode = providers.get_chat_model("main", json_mode=True, settings=s)
        assert plain is not json_mode

    def test_a_changed_setting_is_not_masked_by_the_cache(self):
        # The cache keys on resolved kwargs rather than a hand-listed subset of
        # settings, so a field nobody remembered to add to the key still busts it.
        first = providers.get_chat_model(
            "main", settings=_settings(llm_provider="ollama", ollama_url="http://a:1")
        )
        second = providers.get_chat_model(
            "main", settings=_settings(llm_provider="ollama", ollama_url="http://b:2")
        )
        assert first is not second
        assert second.base_url == "http://b:2"

    def test_reset_clears_the_cache(self):
        s = _settings(llm_provider="ollama")
        first = providers.get_chat_model("main", settings=s)
        providers.reset_models()
        assert providers.get_chat_model("main", settings=s) is not first


class TestRateLimiter:
    def test_one_limiter_is_shared_across_roles(self):
        # Quota is billed per project, not per model. Two limiters would each
        # think they owned the whole budget and together burst to double it.
        s = _settings(llm_provider="google_genai", google_api_key="k",
                      llm_requests_per_second=1.0)
        main = providers.get_chat_model("main", settings=s)
        fast = providers.get_chat_model("fast", settings=s)
        assert main.rate_limiter is fast.rate_limiter
        assert isinstance(main.rate_limiter, InMemoryRateLimiter)

    def test_zero_rate_disables_throttling(self):
        # Correct for Ollama, where the only limit is local CPU.
        s = _settings(llm_provider="ollama", llm_requests_per_second=0)
        assert providers.get_rate_limiter(s) is None
        assert providers.get_chat_model("main", settings=s).rate_limiter is None


class TestHostedDetection:
    @pytest.mark.parametrize(
        "provider,expected",
        [("google_genai", True), ("openai", True), ("ollama", False)],
    )
    def test_is_hosted(self, provider, expected):
        assert providers.is_hosted(_settings(llm_provider=provider)) is expected


class TestEmbeddings:
    """The embedding path had no throttle or retry — the seam a free-tier 429
    slips through to poison the vector store."""

    def test_hosted_embeddings_get_their_own_bucket(self):
        # These used to share the chat limiter, on the theory that one project
        # means one quota. Google meters embeddings separately
        # (embed_content_free_tier_requests, 100/min, versus ~10 RPM for chat),
        # so sharing throttled ingestion to a fraction of its allowance AND
        # still 429'd — the buckets count different things.
        s = _settings(
            llm_provider="google_genai", google_api_key="k",
            llm_requests_per_second=1.0, embed_requests_per_second=2.0,
        )
        emb = providers.get_embeddings(s)
        assert isinstance(emb, providers._ResilientEmbeddings)
        assert emb.rate_limiter is providers.get_embed_rate_limiter(s)
        assert emb.rate_limiter is not providers.get_rate_limiter(s)

    def test_a_batch_costs_one_token_per_text(self):
        # The bug this pins: the quota counts content items, so a limiter taking
        # one token per *call* let a 50-text batch spend 50 units believing it
        # spent 1 — which is how 0.16 req/s still 429'd within four calls.
        taken = []

        class _CountingLimiter:
            def acquire(self, blocking=True):
                taken.append(1)
                return True

        class _Backend:
            def embed_documents(self, texts):
                return [[1.0] for _ in texts]

        wrap = providers._ResilientEmbeddings(_Backend(), limiter=_CountingLimiter())
        wrap.embed_documents(["a", "b", "c", "d", "e"])
        assert len(taken) == 5, "one token per text, not one per call"

    def test_backoff_obeys_the_delay_the_server_asked_for(self):
        # Retrying a quota error before its reset cannot succeed, and each
        # premature attempt spends more of the quota it is waiting on. Google
        # states the reset in the error; exponential backoff from 2s ignored it.
        wrap = providers._ResilientEmbeddings(object(), max_retries=2, base_delay=2.0)
        structured = RuntimeError(
            "429 RESOURCE_EXHAUSTED {'error': {...}, 'details': "
            "[{'@type': '...RetryInfo', 'retryDelay': '58s'}]}"
        )
        prose = RuntimeError("You exceeded your current quota. Please retry in 45.9s.")
        # +1s so we wake after the reset rather than exactly on it.
        assert wrap._backoff(0, structured) == 59.0
        assert wrap._backoff(0, prose) == 46.9
        # No advice -> the exponential schedule still applies.
        assert wrap._backoff(0, RuntimeError("connection reset")) == 2.0
        assert wrap._backoff(1, RuntimeError("connection reset")) == 4.0

    def test_a_server_delay_longer_than_the_exponential_cap_is_honoured(self):
        # Truncating an advertised 58s back to the 30s exponential ceiling would
        # reintroduce the premature retry this exists to stop.
        wrap = providers._ResilientEmbeddings(object(), max_retries=2)
        delay = wrap._backoff(0, RuntimeError("'retryDelay': '58s'"))
        assert delay > providers._ResilientEmbeddings._MAX_BACKOFF_SEC

    def test_hosted_embeddings_pin_the_output_width(self):
        # A pinned width is what lets the zero-vector fallback match the collection
        # even before the first successful embedding.
        s = _settings(llm_provider="google_genai", google_api_key="k")
        emb = providers.get_embeddings(s)
        assert emb.inner.output_dimensionality == providers.GEMINI_EMBED_DIMENSIONS

    def test_zero_rate_leaves_hosted_embeddings_unthrottled(self):
        s = _settings(
            llm_provider="google_genai", google_api_key="k",
            embed_requests_per_second=0,
        )
        emb = providers.get_embeddings(s)
        assert isinstance(emb, providers._ResilientEmbeddings)
        assert emb.rate_limiter is None

    def test_local_embeddings_are_not_wrapped(self):
        # Ollama is local: no quota to throttle and no hosted 429s to retry, so
        # wrapping it would only slow ingestion to the hosted rate for nothing.
        s = _settings(llm_provider="ollama", embedding_provider="ollama")
        emb = providers.get_embeddings(s)
        assert not isinstance(emb, providers._ResilientEmbeddings)

    def test_a_transient_failure_is_retried_not_surfaced(self):
        # base_delay=0 keeps the backoff instant for the test.
        class _Flaky:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED")
                return [[1.0] for _ in texts]

        flaky = _Flaky()
        wrap = providers._ResilientEmbeddings(flaky, max_retries=2, base_delay=0)
        out = wrap.embed_documents(["a", "b"])
        assert flaky.calls == 3
        assert out == [[1.0], [1.0]]

    def test_retries_are_bounded_then_the_error_propagates(self):
        # Exhausted retries must re-raise so the caller's fallback (zero vector at
        # the right width) takes over — the graceful-degradation contract.
        class _Broken:
            def embed_query(self, text):
                raise RuntimeError("permanent")

        wrap = providers._ResilientEmbeddings(_Broken(), max_retries=2, base_delay=0)
        with pytest.raises(RuntimeError, match="permanent"):
            wrap.embed_query("x")
