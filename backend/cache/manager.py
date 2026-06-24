import logging
from collections import OrderedDict
from typing import Optional

import redis
from config import get_settings
from models.state import MigrationState

log = logging.getLogger("CodeMigrateAI.Cache")


class LRUCache:
    def __init__(self, maxsize: int = 500):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[MigrationState]:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def set(self, key: str, value: MigrationState):
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key: str):
        return key in self._cache


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
        r = self.redis
        if r:
            try:
                keys = r.keys("migrate:*")
                if keys:
                    r.delete(*keys)
            except Exception:
                pass
        log.info("Cache cleared")

    def stats(self) -> dict:
        return {
            "local_entries": len(self.local),
            "local_max_entries": self.settings.local_cache_max_entries,
            "redis_enabled": self.settings.redis_enabled and not self._redis_failed,
            "redis_ttl_seconds": self._ttl,
        }
