"""Checkpointer factory for the migration graph.

LangGraph persists graph state per ``thread_id`` through a
``BaseCheckpointSaver``. That is a different axis from the stores in this
package: checkpoints let *one* migration resume mid-flight (or be inspected
step by step), while ``MemoryStore``/``MigrationMemory`` carry outcomes between
unrelated runs.

**The saver must be the async one.** The pipeline drives the graph with
``astream``, and the synchronous ``SqliteSaver`` raises ``NotImplementedError``
on every async checkpoint call — it would break every migration, not degrade
gracefully. So SQLite persistence here means ``AsyncSqliteSaver``.

``AsyncSqliteSaver.__init__`` calls ``asyncio.get_running_loop()`` and binds to
it, which constrains *where* it can be built: inside a running loop, and used
only from that same loop. The app satisfies this (``Pipeline`` is constructed
inside the async ``lifespan``), but tests and scripts that build a graph from
synchronous code do not — hence the fallback to ``MemorySaver``, which is
loop-agnostic. The graph stays checkpointed either way; only durability across
process restarts is lost.
"""

import asyncio
import logging
from typing import Optional

from .memory_store import resolve_data_path

log = logging.getLogger("CodeMigrateAI.Checkpointer")


def build_checkpointer(settings=None) -> Optional[object]:
    """Return a checkpointer for ``workflow.compile``, or None if disabled.

    Never raises: checkpointing is an enhancement, and a migration must still
    run when the checkpoint database cannot be opened.
    """
    if settings is None:
        from config import get_settings

        settings = get_settings()

    if not getattr(settings, "memory_enabled", True):
        return None
    if not getattr(settings, "checkpointer_enabled", True):
        return None

    saver = _build_sqlite_saver(settings)
    if saver is not None:
        return saver
    return _build_memory_saver()


def _build_sqlite_saver(settings) -> Optional[object]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop: AsyncSqliteSaver cannot bind. Expected for tests and any
        # synchronous graph build — not an error worth warning about.
        log.debug("No running event loop; using in-memory checkpointer")
        return None

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        log.info("langgraph-checkpoint-sqlite unavailable (%s); using MemorySaver", exc)
        return None

    try:
        # Same anchoring as the memory store: a CWD-relative checkpoint file
        # would resume from a different database depending on where the process
        # was launched.
        db_path = resolve_data_path(settings.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # aiosqlite.connect() is lazy — the connection opens on first await,
        # which AsyncSqliteSaver.setup() performs. Safe to build synchronously.
        saver = AsyncSqliteSaver(
            aiosqlite.connect(str(db_path), check_same_thread=False)
        )
        log.info("Graph checkpointing to %s", db_path)
        return saver
    except Exception as exc:  # noqa: BLE001 — never block graph construction
        log.warning("SQLite checkpointer unavailable (%s); using MemorySaver", exc)
        return None


def _build_memory_saver() -> Optional[object]:
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception as exc:  # noqa: BLE001
        log.warning("No checkpointer available (%s); running without one", exc)
        return None
