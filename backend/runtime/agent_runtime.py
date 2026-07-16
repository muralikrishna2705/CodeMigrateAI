"""Runtime — the dynamic executor that runs one agent inside the graph.

Formerly ``RuntimeAgent``, a prelude agent that only logged counts. It is now the
single place that actually *runs* an agent: every graph node delegates to
``Runtime.execute`` (see ``graph.nodes._make_node``). Centralizing execution here
means the cross-cutting concerns — the circuit breaker and dependency injection —
live in one place instead of being copy-pasted into each node.

Flow per agent:

  circuit open? -> skip (return the state unchanged)
  resolve the agent class by name from ``BaseAgent``'s registry
  hydrate a ``MigrationState`` from the graph dict
  build the agent's config from the ``Provider`` via its declared ``needs``
  construct it with the shared LLM client and run it
  a run whose latest report is an error trips the circuit breaker
  write the agent's outputs back into the graph dict
"""

import logging

from agents.base import BaseAgent
from runtime import agent_recovery
from runtime.agent_providers import Provider

log = logging.getLogger("CodeMigrateAI.Runtime")


class Runtime:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    async def execute(self, agent_name: str, state: dict) -> dict:
        # Imported lazily: graph.nodes imports this module at load time, so a
        # top-level import back into it would be circular.
        from graph.nodes import (
            get_llm_client,
            hydrate_state,
            last_report_status,
            writeback,
        )

        if agent_recovery.is_circuit_open(agent_name):
            log.warning("Circuit open for %s; skipping node", agent_name)
            return state

        agent_cls = BaseAgent.get_registry().get(agent_name)
        if not agent_cls:
            log.error("Agent %s not found in registry", agent_name)
            state["errors"] = state.get("errors", []) + [
                f"Agent {agent_name} not found"
            ]
            return state

        mig_state = hydrate_state(state)
        # Data-driven DI: resolve exactly the dependencies the agent declares it
        # needs from the Provider, instead of a hardcoded per-agent config block.
        config = self.provider.resolve(getattr(agent_cls, "needs", ()))
        agent = agent_cls(get_llm_client(), config)
        mig_state = await agent(mig_state)
        if last_report_status(mig_state, agent_name) == "error":
            agent_recovery.record_failure(agent_name)
        return writeback(state, mig_state)
