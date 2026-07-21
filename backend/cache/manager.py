import logging
from typing import Optional

import redis
from cache.keys import KEY_NAMESPACE, key_prefix
from config import get_settings
from dsa import PrefixLRU
from models.state import MigrationState

log = logging.getLogger("CodeMigrateAI.Cache")


class LRUCache(PrefixLRU[MigrationState]):
    """Local migration-state cache.

    Recency comes from the ``OrderedDict`` in :class:`~dsa.lru.PrefixLRU`; the
    trie beside it is what lets :meth:`CacheManager.invalidate` drop one
    migration family — every entry into Java 21, say — without touching the rest
    of the cache or walking every key to find them.
    """


class CacheManager:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.local = LRUCache(maxsize=self.settings.local_cache_max_entries)
        self._redis = None
        self._redis_failed = False
        self._ttl = self.settings.redis_ttl_seconds

    @property
    def redis(self):
        if self._redis_failed or not self.settings.redis_enabled:
            return None
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    self.settings.redis_url,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    retry_on_timeout=True,
                    decode_responses=True,
                )
                self._redis.ping()
                log.info("Redis connection established")
            except Exception as e:
                log.warning(f"Redis unavailable, using local cache only: {e}")
                self._redis_failed = True
                self._redis = None
        return self._redis

    async def get(self, key: str) -> Optional[MigrationState]:
        local_val = self.local.get(key)
        if local_val:
            log.debug(f"Cache hit (local): {key[:16]}...")
            return local_val.model_copy(deep=True)

        r = self.redis
        if r:
            try:
                data = r.get(key)
                if data:
                    state = MigrationState.model_validate_json(data)
                    self.local.set(key, state)
                    log.debug(f"Cache hit (redis): {key[:16]}...")
                    return state.model_copy(deep=True)
            except Exception as e:
                log.warning(f"Redis get failed: {e}")
        return None

    async def set(self, key: str, state: MigrationState):
        self.local.set(key, state.model_copy(deep=True))

        r = self.redis
        if r:
            try:
                r.setex(key, self._ttl, state.model_dump_json())
                log.debug(f"Cache set (redis): {key[:16]}...")
            except Exception as e:
                log.warning(f"Redis set failed: {e}")

    def clear(self):
        self.local.clear()
        self._redis_delete(f"{KEY_NAMESPACE}:*")
        log.info("Cache cleared")

    def invalidate(
        self,
        source_language: str = "",
        source_version: str = "",
        target_language: str = "",
        target_version: str = "",
    ) -> int:
        """Evict one migration family — e.g. every migration into Java 21.

        The narrow alternative to ``clear``: when a language profile changes or
        a target's RAG corpus is re-indexed, only migrations touching it are
        stale. Returns the number of local entries dropped; Redis deletes by the
        same prefix but does not report a count.
        """
        prefix = key_prefix(
            source_language, source_version, target_language, target_version
        )
        removed = self.local.invalidate_prefix(prefix)
        self._redis_delete(f"{prefix}*")
        log.info("Invalidated %d local entries under %s", removed, prefix)
        return removed

    def _redis_delete(self, pattern: str) -> None:
        """Delete keys matching ``pattern`` without blocking the server.

        ``KEYS`` scans the entire keyspace in one uninterruptible pass, which
        stalls every other client on a shared Redis. ``scan_iter`` walks it in
        cursor-sized batches instead, and deleting in batches keeps the argument
        list bounded on a large keyspace.
        """
        r = self.redis
        if not r:
            return
        try:
            batch: list[str] = []
            for key in r.scan_iter(match=pattern, count=500):
                batch.append(key)
                if len(batch) >= 500:
                    r.delete(*batch)
                    batch.clear()
            if batch:
                r.delete(*batch)
        except Exception as e:
            log.warning("Redis delete for %s failed: %s", pattern, e)

    def stats(self) -> dict:
        return {
            "local_entries": len(self.local),
            "local_max_entries": self.settings.local_cache_max_entries,
            "redis_enabled": self.settings.redis_enabled and not self._redis_failed,
            "redis_ttl_seconds": self._ttl,
        }
