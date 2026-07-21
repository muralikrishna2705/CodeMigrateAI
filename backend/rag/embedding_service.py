import logging
from collections import OrderedDict

from langchain_core.embeddings import Embeddings

log = logging.getLogger("CodeMigrateAI.Embeddings")

#: Fallback vector width used until a real embedding reveals the backend's own.
#: 768 matches nomic-embed-text, the historical default.
DEFAULT_DIMENSIONS = 768


class CachedEmbeddings(Embeddings):
    """LRU-cached, failure-tolerant wrapper around any embedding backend.

    Subclasses LangChain's :class:`Embeddings` so it can be handed directly to
    Chroma (and any component that ``isinstance``-checks the interface) while
    still routing every call through the LRU cache and the zero-vector fallback.

    The wrapped backend is injected rather than constructed here, so the same
    cache serves Gemini, Ollama, or anything else :mod:`llm.providers` can build.

    The cache is keyed by the text itself rather than a digest of it. Digesting
    first was strictly wasted work: ``dict`` already hashes the key, CPython
    caches that hash on the string object, and equality confirms the hit — so
    the digest bought no speed and cost a collision surface where a collision
    silently returns *another document's* embedding. Holding the key strings
    costs a fraction of what the vectors beside them already do.
    """

    def __init__(
        self,
        inner: Embeddings | None = None,
        model: str = "",
        base_url: str = "",
        max_cache: int = 200,
        dimensions: int = DEFAULT_DIMENSIONS,
    ):
        if inner is None:
            inner = self._build_inner(model, base_url)
        self._inner = inner
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_cache = max_cache
        # The fallback must match the backend's width or Chroma rejects the
        # insert, and backends disagree (nomic 768, gemini-embedding-001 3072).
        # Seeded with a default and corrected the first time a real vector
        # arrives, so a wrong configured value cannot outlive one success.
        self._dimensions = dimensions

    @staticmethod
    def _build_inner(model: str, base_url: str) -> Embeddings:
        """Resolve the backing embedding service.

        An explicit ``model``/``base_url`` pair selects Ollama directly (the
        signature predates the provider layer and tests still use it); with
        neither, the configured provider decides.
        """
        if model or base_url:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=model or "nomic-embed-text",
                base_url=base_url or "http://host.docker.internal:11434",
            )
        from llm import providers

        return providers.get_embeddings()

    def _remember_dimensions(self, vec) -> None:
        """Adopt the backend's real vector width from a successful embedding."""
        if vec and len(vec) != self._dimensions:
            log.info(
                "Embedding width is %d (was assuming %d)", len(vec), self._dimensions
            )
            self._dimensions = len(vec)

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        try:
            vec = self._inner.embed_query(text)
            self._remember_dimensions(vec)
        except Exception as e:
            # Not cached. Embedding failures here are overwhelmingly transient —
            # a 503, a rate limit, a dropped connection — and caching the
            # fallback would make one bad second permanent for that query text
            # for the life of the process, long after the service recovered.
            log.warning("Embedding failed, using fallback: %s", e)
            return self._fallback_embed(text)
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
                if batch:
                    self._remember_dimensions(batch[0])
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
        # A zero vector is at cosine-distance 1.0 from everything, so it will
        # never pass the similarity threshold and is effectively ignored — which
        # is the point: a failed embedding must degrade to "matches nothing"
        # rather than to a wrong neighbour. Width tracks the live backend so the
        # vector store still accepts it.
        return [0.0] * self._dimensions

    def clear_cache(self):
        self._cache.clear()
