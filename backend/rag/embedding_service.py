import logging
from collections import OrderedDict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

log = logging.getLogger("CodeMigrateAI.Embeddings")


class CachedEmbeddings(Embeddings):
    """LRU-cached, failure-tolerant wrapper around OllamaEmbeddings.

    Subclasses LangChain's :class:`Embeddings` so it can be handed directly to
    Chroma (and any component that ``isinstance``-checks the interface) while
    still routing every call through the LRU cache and the zero-vector fallback.

    The cache is keyed by the text itself rather than a digest of it. Digesting
    first was strictly wasted work: ``dict`` already hashes the key, CPython
    caches that hash on the string object, and equality confirms the hit — so
    the digest bought no speed and cost a collision surface where a collision
    silently returns *another document's* embedding. Holding the key strings
    costs a fraction of what the 768-float vectors beside them already do.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://host.docker.internal:11434", max_cache: int = 200):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_cache = max_cache

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        try:
            vec = self._inner.embed_query(text)
        except Exception as e:
            log.warning("Ollama embedding failed, using fallback: %s", e)
            vec = self._fallback_embed(text)
        self._cache[text] = vec
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        uncached = []
        indices = []
        results: list[list[float] | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                results[i] = cached
            else:
                uncached.append(text)
                indices.append(i)

        if uncached:
            try:
                batch = self._inner.embed_documents(uncached)
            except Exception as e:
                log.warning("Batch embedding failed: %s", e)
                batch = []
            # A short batch would otherwise leave holes in `results`, and the
            # return below would hand back fewer vectors than texts — silently
            # misaligning every embedding with the wrong document downstream.
            if len(batch) < len(uncached):
                log.warning(
                    "Embedding backend returned %d vectors for %d texts; "
                    "padding with fallback",
                    len(batch),
                    len(uncached),
                )
                batch = list(batch) + [
                    self._fallback_embed(t) for t in uncached[len(batch):]
                ]
            # `indices[pos]` is the position in the original `texts` list, while
            # `uncached[pos]`/`batch[pos]` line up positionally with each other.
            for pos, original_index in enumerate(indices):
                self._cache[uncached[pos]] = batch[pos]
                results[original_index] = batch[pos]
                if len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)

        return [r for r in results if r is not None]

    def _fallback_embed(self, text: str) -> list[float]:
        # Return a zero vector matching nomic-embed-text's dimension (768).
        # A zero vector is at cosine-distance 1.0 from everything, so it will
        # never pass the 0.7 similarity threshold and effectively be ignored.
        return [0.0] * 768

    def clear_cache(self):
        self._cache.clear()
