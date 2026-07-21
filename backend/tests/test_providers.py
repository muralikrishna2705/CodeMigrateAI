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
