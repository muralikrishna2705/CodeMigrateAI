import asyncio
import hashlib
import logging
import re
from collections import OrderedDict

from config import get_settings

log = logging.getLogger("CodeMigrateAI.RAGPipeline")


# --- Query construction ----------------------------------------------------
#
# Retrieval quality is the main anti-hallucination lever: a query built only
# from the language pair ("python to java migration patterns") retrieves generic
# boilerplate, whereas a query built from THIS code's imports and APIs retrieves
# examples that actually show how those constructs map to the target language.

# Import / include / require statements identify the libraries in play — the
# strongest signal for "what does this code depend on".
_IMPORT_PATTERNS = (
    re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),          # python / java / js / go / kotlin
    re.compile(r"^\s*from\s+([\w.]+)\s+import\b", re.MULTILINE),  # python
    re.compile(r"^\s*using\s+([\w.]+)\s*;", re.MULTILINE),        # csharp
    re.compile(r'^\s*#include\s*[<"]([\w./]+)[>"]', re.MULTILINE),  # cpp
    re.compile(r'require\(\s*["\']([\w./-]+)["\']\s*\)', re.MULTILINE),  # node
    re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),             # rust
)

# Called identifiers (``foo(``) and Pascal/CamelCase type names carry the next
# strongest signal: framework/type/method names distinctive to the code.
_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_TYPE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")

# Language keywords / ubiquitous identifiers that add noise rather than signal.
_STOPWORDS = frozenset(
    {
        "if", "else", "elif", "for", "while", "switch", "case", "return",
        "import", "from", "class", "def", "public", "private", "protected",
        "static", "final", "void", "int", "long", "float", "double", "string",
        "str", "bool", "boolean", "char", "byte", "var", "let", "const",
        "function", "func", "fn", "fun", "sub", "new", "delete", "this", "self",
        "super", "true", "false", "null", "none", "nil", "and", "or", "not",
        "in", "is", "as", "with", "try", "catch", "except", "finally", "throw",
        "throws", "raise", "await", "async", "yield", "print", "println",
        "main", "using", "namespace", "package", "struct", "enum", "interface",
        "record", "impl", "trait", "type", "map", "list", "set", "dict",
    }
)


class RAGPipeline:
    def __init__(self, vector_store, embedding_service):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self._query_cache: OrderedDict[str, list] = OrderedDict()
        self._max_cache = 100

    def _extract_code_signals(self, source_code: str, max_symbols: int) -> list[str]:
        """Pull the salient, code-specific symbols to seed the retrieval query.

        Imports/includes come first (they name the dependencies), followed by
        distinctive call and type identifiers in source order. Stopwords and
        duplicates are dropped, and the result is capped at ``max_symbols`` so a
        few high-signal terms dominate the query embedding.
        """
        signals: list[str] = []
        seen: set[str] = set()

        def _add(token: str) -> None:
            token = token.strip()
            if not token or token.lower() in _STOPWORDS or token in seen:
                return
            seen.add(token)
            signals.append(token)

        for pattern in _IMPORT_PATTERNS:
            for match in pattern.findall(source_code):
                _add(match)

        for pattern in (_CALL_PATTERN, _TYPE_PATTERN):
            for match in pattern.findall(source_code):
                _add(match)

        return signals[:max_symbols]

    def _build_query(
        self,
        source_language: str,
        target_language: str,
        source_code: str,
        symbols: list[str],
    ) -> str:
        """Compose a code-aware retrieval query from the source code + languages."""
        settings = get_settings()
        parts = [f"{source_language} to {target_language} migration"]
        if symbols:
            parts.append("involving " + ", ".join(symbols))
        # A short excerpt anchors the query in the actual code shape; capped so it
        # complements rather than drowns the symbol signal.
        excerpt = source_code.strip()[: settings.rag_query_code_chars]
        if excerpt:
            parts.append(excerpt)
        return "\n".join(parts)

    async def _search(self, query: str, where: dict | None):
        settings = get_settings()
        return await asyncio.to_thread(
            self._vector_store.similarity_search,
            query,
            k=settings.rag_top_k,
            score_threshold=settings.rag_min_score,
            where=where,
        )

    async def _keyword_search(self, symbols: list[str], where: dict | None):
        """Exact-symbol keyword leg; empty if the store has no keyword support."""
        if not symbols or not hasattr(self._vector_store, "keyword_search"):
            return []
        settings = get_settings()
        try:
            return await asyncio.to_thread(
                self._vector_store.keyword_search,
                symbols,
                settings.rag_top_k,
                where,
            )
        except Exception as e:
            log.warning("Keyword retrieval failed: %s", e)
            return []

    @staticmethod
    def _rrf_merge(vector_hits, keyword_hits, k: int, rrf_k: int):
        """Reciprocal-rank-fusion merge of the vector and keyword legs.

        RRF ranks by 1/(rrf_k + rank) summed across legs, so a doc that ranks
        well in either leg surfaces. The display score keeps each doc's own leg
        signal — calibrated cosine when it came from the vector leg, else the
        keyword match ratio — rather than the raw (tiny) RRF value.
        """
        fused: dict[str, float] = {}
        best: dict[str, tuple] = {}

        def _key(doc):
            return hashlib.sha256(doc.page_content.encode()).hexdigest()

        for rank, (doc, score) in enumerate(vector_hits):
            key = _key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            best[key] = (doc, score)
        for rank, (doc, ratio) in enumerate(keyword_hits):
            key = _key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            best.setdefault(key, (doc, ratio))  # keep vector score if already seen

        ordered = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
        return [best[key] for key in ordered]

    async def _retrieve(self, query: str, symbols: list[str], where: dict | None):
        """One retrieval pass: vector leg, optionally fused with the keyword leg."""
        settings = get_settings()
        vector_hits = await self._search(query, where)
        if not settings.rag_hybrid_enabled:
            return vector_hits
        keyword_hits = await self._keyword_search(symbols, where)
        if not keyword_hits:
            return vector_hits
        return self._rrf_merge(
            vector_hits, keyword_hits, settings.rag_top_k, settings.rag_rrf_k
        )

    @staticmethod
    def _filter_phases(target_language, target_version, settings):
        """Ordered metadata filters, tightest first, for the retrieval ladder.

        Phase A — target language AND (exact target version OR wildcard-version
                  docs), so unversioned corpus content still qualifies.
        Phase B — target language, any version.
        Phase C — unfiltered, so we never regress to zero grounding.

        Phases collapse when a filter is disabled or would duplicate another.
        """
        phases: list[dict | None] = []
        if target_version and settings.rag_filter_by_target_version:
            phases.append(
                {
                    "$and": [
                        {"language": target_language},
                        {
                            "version": {
                                "$in": [target_version, settings.rag_version_wildcard]
                            }
                        },
                    ]
                }
            )
        if settings.rag_filter_by_target_language:
            phases.append({"language": target_language})
        phases.append(None)

        deduped: list[dict | None] = []
        for phase in phases:
            if phase not in deduped:
                deduped.append(phase)
        return deduped

    async def _retrieve_ladder(self, query, symbols, phases):
        """Return hits from the tightest filter phase that finds anything.

        The target-version filter is tried first; only if it comes back empty do
        we broaden to language-only, then unfiltered. Using one phase's results
        (rather than topping up across phases) keeps exact-version grounding from
        being diluted by off-target docs and bounds retrieval to one extra query
        per empty phase — the same fallback contract the old code had, with the
        version phase added in front.
        """
        for where in phases:
            hits = await self._retrieve(query, symbols, where)
            if hits:
                return hits
        return []

    @staticmethod
    def _rank(results, target_version, settings):
        """Re-order hits by base relevance plus small authority boosts.

        The displayed score stays the raw relevance; only the ordering shifts so
        exact-version, official, and migration docs win ties without masking a
        genuinely more relevant (higher-cosine) example.
        """
        migration_types = set(settings.rag_migration_doc_types)

        def _boost(doc) -> float:
            md = doc.metadata or {}
            boost = 0.0
            if target_version and md.get("version") == target_version:
                boost += settings.rag_rank_weight_version
            if md.get("is_official"):
                boost += settings.rag_rank_weight_official
            if md.get("doc_type") in migration_types:
                boost += settings.rag_rank_weight_migration
            return boost

        return sorted(
            results, key=lambda pair: pair[1] + _boost(pair[0]), reverse=True
        )

    async def enrich_prompt(
        self,
        source_language: str,
        target_language: str,
        source_code: str,
        base_prompt: str,
        target_version: str = "",
    ) -> str:
        settings = get_settings()
        if not settings.enable_rag:
            return base_prompt

        symbols = self._extract_code_signals(
            source_code, settings.rag_query_max_symbols
        )
        query = self._build_query(
            source_language, target_language, source_code, symbols
        )

        # Target version steers both the retrieval filter ladder and the ranking
        # boosts, so it must be part of the cache identity.
        cache_key = hashlib.sha256(
            f"{target_version}\n{query}".encode()
        ).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
        else:
            try:
                phases = self._filter_phases(
                    target_language, target_version, settings
                )
                results = await self._retrieve_ladder(query, symbols, phases)
                results = self._rank(results, target_version, settings)
                cached = results[: settings.rag_top_k]
                self._query_cache[cache_key] = cached
                if len(self._query_cache) > self._max_cache:
                    self._query_cache.popitem(last=False)
            except Exception as e:
                log.warning("RAG retrieval failed: %s", e)
                cached = []

        if not cached:
            return base_prompt

        context_parts = ["## Reference Examples\nHere are relevant code patterns from the target language:\n"]
        for doc, score in cached:
            md = doc.metadata or {}
            lang = md.get("language", "unknown")
            version = md.get("version", "")
            doc_type = md.get("doc_type", "")
            # Surface version/authority so the model treats an official
            # target-version migration guide as stronger than a generic example.
            label_bits = [lang]
            if version and version != settings.rag_version_wildcard:
                label_bits.append(version)
            if doc_type:
                label_bits.append(doc_type)
            label = " · ".join(label_bits)
            context_parts.append(f"### {label} (relevance: {score:.2f})")
            context_parts.append(f"```{lang}\n{doc.page_content}\n```")

        context = "\n\n".join(context_parts)
        return f"{context}\n\n---\n\n{base_prompt}"
