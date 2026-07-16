"""Shared pytest fixtures for the backend suite."""

import pytest

from runtime import agent_recovery
from runtime.agent_observer import ObserverAgent


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    """Isolate the process-global circuit breaker + observer metrics per test.

    Both live in module-level state so their effects can leak across tests (a
    breaker tripped by one test would skip an agent a later test needs). Reset
    around every test to keep them deterministic.
    """
    agent_recovery.reset()
    ObserverAgent.reset_metrics()
    yield
    agent_recovery.reset()
    ObserverAgent.reset_metrics()


@pytest.fixture
def settings_override(monkeypatch):
    """Override Settings fields (via env) for the duration of one test.

    ``get_settings`` is ``lru_cache``d, so a settings change made mid-suite would
    otherwise persist into every later test in the process. Clearing the cache on
    both sides of the yield scopes the override to this test — including when the
    test fails, which an inline cache_clear() at the end of the test body would
    silently skip.
    """
    from config import get_settings

    def _apply(**overrides):
        for key, value in overrides.items():
            monkeypatch.setenv(key.upper(), str(value))
        get_settings.cache_clear()
        return get_settings()

    get_settings.cache_clear()
    yield _apply
    get_settings.cache_clear()
