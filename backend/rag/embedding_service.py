import hashlib
import logging
from collections import OrderedDict

from langchain_community.embeddings import OllamaEmbeddings

log = logging.getLogger("CodeMigrateAI.Embeddings")


class CachedEmbeddings:
    """LRU-cached wrapper around OllamaEmbeddings."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://host.docker.internal:11434", max_cache: int = 200):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_cache = max_cache

    def embed_query(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            vec = self._inner.embed_query(text)
        except Exception as e:
            log.warning("Ollama embedding failed, using fallback: %s", e)
            vec = self._fallback_embed(text)
        self._cache[key] = vec
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        uncached = []
        indices = []
        results: list[list[float] | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                results[i] = cached
            else:
                uncached.append(text)
                indices.append(i)

        if uncached:
            try:
                batch = self._inner.embed_documents(uncached)
            except Exception as e:
                log.warning("Batch embedding failed: %s", e)
                batch = [self._fallback_embed(t) for t in uncached]
            # `indices[pos]` is the position in the original `texts` list, while
            # `uncached[pos]`/`batch[pos]` line up positionally with each other.
            for pos, original_index in enumerate(indices):
                key = hashlib.sha256(uncached[pos].encode()).hexdigest()
                self._cache[key] = batch[pos]
                results[original_index] = batch[pos]
                if len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)

        return [r for r in results if r is not None]

    def _fallback_embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h]

    def clear_cache(self):
        self._cache.clear()
