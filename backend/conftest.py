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
