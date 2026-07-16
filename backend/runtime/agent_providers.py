"""Provider — the dynamic dependency-injection container for agents.

Formerly ``ProviderAgent``, a no-op prelude agent whose registry nothing read.
It is now the DI container the graph runtime pulls agent dependencies from,
replacing the hand-wired module globals and the hardcoded per-agent config that
used to live in ``graph.nodes``. Shared services (the LLM client, the RAG
pipeline) are registered here once at startup; agents declare what they need via
``BaseAgent.needs`` and the runtime resolves those names against this container
when it constructs each agent — so adding a new dependency is a ``register`` call
plus a ``needs`` entry, not a new ``if`` branch in the node factory.

Two kinds of entries:

- **instances** — a fixed object shared across requests (``llm``, ``rag_pipeline``).
- **factories** — a zero-arg callable resolved at ``get`` time. Used for the
  per-request SSE token callback, which is task-scoped in a ``ContextVar`` so
  concurrent ``/migrate/stream`` requests can't clobber each other's callback;
  the factory reads the ``ContextVar`` on each resolve.
"""

import logging
from typing import Any, Callable, Optional

log = logging.getLogger("CodeMigrateAI.Provider")


class Provider:
    """A minimal service container: register instances/factories, resolve by name."""

    def __init__(self) -> None:
        self._instances: dict[str, Any] = {}
        self._factories: dict[str, Callable[[], Any]] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register a shared instance under ``name`` (overwrites any prior)."""
        self._instances[name] = instance
        log.info("Provider registered: %s", name)

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a zero-arg factory resolved lazily on every ``get``."""
        self._factories[name] = factory

    def get(self, name: str) -> Optional[Any]:
        """Resolve ``name``: a factory (called now) wins, else the registered
        instance, else ``None``."""
        if name in self._factories:
            return self._factories[name]()
        return self._instances.get(name)

    def has(self, name: str) -> bool:
        return name in self._factories or name in self._instances

    def resolve(self, names) -> dict[str, Any]:
        """Build a config dict for the given dependency names.

        Names that resolve to ``None`` (e.g. an unset RAG pipeline, or the stream
        callback when no request is streaming) are omitted, so an agent receives
        exactly the dependencies that are actually wired — matching the old
        "only include it when present" behaviour of the hardcoded config.
        """
        config: dict[str, Any] = {}
        for name in names:
            value = self.get(name)
            if value is not None:
                config[name] = value
        return config
