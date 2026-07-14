import asyncio
import hashlib
import logging
from collections import OrderedDict

from config import get_settings

log = logging.getLogger("CodeMigrateAI.RAGPipeline")


class RAGPipeline:
    def __init__(self, vector_store, embedding_service):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self._query_cache: OrderedDict[str, list] = OrderedDict()
        self._max_cache = 100

    async def enrich_prompt(
        self,
        source_language: str,
        target_language: str,
        source_code: str,
        base_prompt: str,
    ) -> str:
        settings = get_settings()
        if not settings.enable_rag:
            return base_prompt

        query = f"{source_language} to {target_language} migration patterns and examples"

        cache_key = hashlib.sha256(query.encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is None:
            try:
                results = await asyncio.to_thread(
                    self._vector_store.similarity_search,
                    query,
                    k=settings.rag_top_k,
                    score_threshold=settings.rag_min_score,
                )
                cached = results
                self._query_cache[cache_key] = results
                if len(self._query_cache) > self._max_cache:
                    self._query_cache.popitem(last=False)
            except Exception as e:
                log.warning("RAG retrieval failed: %s", e)
                cached = []

        if not cached:
            return base_prompt

        context_parts = ["## Reference Examples\nHere are relevant code patterns from the target language:\n"]
        for doc, score in cached:
            lang = doc.metadata.get("language", "unknown")
            context_parts.append(f"### {lang} (relevance: {score:.2f})")
            context_parts.append(f"```{lang}\n{doc.page_content}\n```")

        context = "\n\n".join(context_parts)
        return f"{context}\n\n---\n\n{base_prompt}"
