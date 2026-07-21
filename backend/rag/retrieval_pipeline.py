import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

from config import get_settings
from dsa import top_k

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


# --- Agentic retrieval strategies ------------------------------------------
#
# The default retrieval is one passive pass. These strategies turn retrieval
# into an active step an agent reasons through: reformulating the query (HyDE,
# multi-query), decomposing it (multi-hop), grading what came back (CRAG,
# Self-RAG), expanding chunks to their parents, or compressing them. Each is a
# thin layer over ``RAGPipeline.run_query`` — so it inherits hybrid retrieval,
# the version/language filter ladder, and metadata ranking for free — plus
# optional LLM reasoning that always degrades to plain single-hop retrieval when
# no model is wired or a call fails.


@dataclass
class RetrievalRequest:
    """The unit every retrieval strategy consumes.

    Carries the formulated query plus the surrounding migration context, so a
    strategy can reformulate, decompose, or grade before it ever touches the
    vector store. ``symbols`` still drives the hybrid keyword leg; the source
    fields let a strategy ground its LLM reasoning in the actual code.
    """

    query: str
    target_language: str = ""
    target_version: str = ""
    symbols: list[str] | None = None
    source_language: str = ""
    source_code: str = ""
    code_metrics: dict | None = None
    extra_terms: list[str] | None = None


def merge_hits(hitlists: list[list[tuple]], k: int) -> list[tuple]:
    """Fuse several ``(doc, score)`` lists: dedup by content, keep the best score.

    Used by the multi-pass strategies (multi-query, multi-hop, CRAG) to combine
    the results of several retrievals into one ranked, deduplicated list. Dedup
    is by page content — the same identity ``RAGPipeline._rrf_merge`` uses — so a
    document that surfaces for several sub-queries is counted once, at its
    highest observed score.

    Selection is a bounded heap: only ``k`` of the fused candidates are ever
    wanted, so ordering all of them costs O(N log N) to throw most of it away.
    """
    best: dict[str, tuple] = {}
    for hits in hitlists:
        for doc, score in hits:
            current = best.get(doc.page_content)
            if current is None or score > current[1]:
                best[doc.page_content] = (doc, score)
    return top_k(best.values(), k)


class RetrievalStrategy(ABC):
    """Base for an agentic retrieval strategy.

    A strategy turns a :class:`RetrievalRequest` into ranked ``(doc, score)``
    hits, reusing the pipeline's ``run_query`` for store access. Its LLM helpers
    are deliberately failure-tolerant: they return neutral values (empty string,
    empty dict) when no client is wired or a call raises, so a strategy collapses
    to plain retrieval rather than breaking the migration — the same contract the
    RetrieverAgent tool loop already honors.
    """

    name = "base"

    def __init__(self, pipeline: "RAGPipeline", llm=None) -> None:
        self.pipeline = pipeline
        self.llm = llm

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        """Return ranked ``(doc, score)`` hits for the request."""

    async def _run_query(self, query: str, request: RetrievalRequest) -> list[tuple]:
        """One retrieval pass through the shared pipeline core; ``[]`` on failure."""
        try:
            return await self.pipeline.run_query(
                query,
                request.target_language,
                request.target_version,
                request.symbols,
            )
        except Exception as exc:  # noqa: BLE001 — retrieval degrades, never breaks
            log.warning("[%s] run_query failed: %s", self.name, exc)
            return []

    async def _ask(self, prompt: str, system: str = "", fmt: str | None = None) -> str:
        """Fast-model LLM call; empty string when unavailable or on error."""
        call = getattr(self.llm, "call_llm", None)
        if call is None:
            return ""
        try:
            return (
                await call(
                    prompt,
                    system_prompt=system,
                    fmt=fmt,
                    model=getattr(self.llm, "fast_model", None),
                )
                or ""
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] LLM call failed: %s", self.name, exc)
            return ""

    async def _ask_json(self, prompt: str, system: str = "") -> dict:
        """Fast-model JSON call parsed to a dict; ``{}`` when unavailable/bad."""
        raw = await self._ask(prompt, system=system, fmt="json")
        if not raw:
            return {}
        extract = getattr(self.llm, "extract_json", None)
        try:
            return extract(raw) if extract else json.loads(raw)
        except Exception:  # noqa: BLE001
            return {}

    @property
    def has_llm(self) -> bool:
        return getattr(self.llm, "call_llm", None) is not None


class SingleHopStrategy(RetrievalStrategy):
    """The default: one retrieval pass, exactly the pre-strategy behavior."""

    name = "single_hop"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        return await self._run_query(request.query, request)


class RAGPipeline:
    def __init__(self, vector_store, embedding_service, llm_client=None):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        # Optional — only the LLM-driven strategies (HyDE, multi-query, CRAG,
        # Self-RAG, …) use it. Left None (the default, and what every existing
        # caller/test passes) the pipeline behaves exactly as before: single-hop
        # retrieval with no LLM reasoning.
        self._llm = llm_client
        self._query_cache: OrderedDict[str, list] = OrderedDict()
        self._max_cache = 100

    @property
    def vector_store(self):
        """The backing store, for strategies that need direct access.

        The Parent Document strategy uses it to pull sibling chunks by
        ``parent_id``; everything else goes through ``run_query``.
        """
        return self._vector_store

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

    @staticmethod
    def _metric_terms(code_metrics: dict | None) -> list[str]:
        """Short construct terms from the AnalyzerAgent / DeepAnalyzerAgent output.

        These name what the code actually *does* (generics, async, reflection,
        specific stdlib deps) rather than just its import lines, so they sharpen
        retrieval toward the constructs that need migrating.
        """
        if not code_metrics:
            return []
        terms: list[str] = []
        for key in ("key_constructs", "deprecated_patterns"):
            vals = code_metrics.get(key)
            if isinstance(vals, list):
                terms.extend(str(v) for v in vals if v)
        deep = code_metrics.get("deep_analysis") or {}
        for key in ("complex_constructs", "stdlib_dependencies"):
            vals = deep.get(key)
            if isinstance(vals, list):
                terms.extend(str(v) for v in vals if v)
        return terms

    @classmethod
    def _merge_query_terms(
        cls,
        extra_terms: list[str] | None,
        symbols: list[str],
        code_metrics: dict | None,
        cap: int,
    ) -> list[str]:
        """Combine targeted re-retrieval terms, code symbols, and constructs.

        Priority order — re-retrieval requests first, then the code's own
        symbols, then analyzer-detected constructs — so the highest-signal terms
        survive the cap. Deduplicated, stopword-free, and length-bounded so a
        stray sentence-length metric can't drown the query.
        """
        merged: list[str] = []
        seen: set[str] = set()

        def _add(token: str) -> None:
            token = (token or "").strip()
            if not token or len(token) > 40 or "\n" in token:
                return
            if token.lower() in _STOPWORDS or token in seen:
                return
            seen.add(token)
            merged.append(token)

        for token in extra_terms or []:
            _add(token)
        for token in symbols:
            _add(token)
        for token in cls._metric_terms(code_metrics):
            _add(token)
        return merged[:cap]

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

        # Page content is the document identity, used directly as the dict key.
        # Digesting it first bought nothing: dict lookup already hashes the
        # string (once, cached on the object) and then confirms by equality, so
        # the extra pass was pure work that also introduced a collision surface.
        for rank, (doc, score) in enumerate(vector_hits):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            best[key] = (doc, score)
        for rank, (doc, ratio) in enumerate(keyword_hits):
            key = doc.page_content
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            best.setdefault(key, (doc, ratio))  # keep vector score if already seen

        # Ties are routine in RRF, so the tie-break must not shift: nlargest
        # breaks toward the earlier element, matching the stable sort it replaces.
        return [best[key] for key in top_k(fused, k, key=fused.__getitem__)]

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
    def _rank(results, target_version, settings, limit: int | None = None):
        """Re-order hits by base relevance plus small authority boosts.

        The displayed score stays the raw relevance; only the ordering shifts so
        exact-version, official, and migration docs win ties without masking a
        genuinely more relevant (higher-cosine) example.

        ``limit`` selects the top hits through a bounded heap instead of ordering
        the whole ladder result; omit it to rank everything.
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

        def _ranked(pair):
            return pair[1] + _boost(pair[0])

        if limit is None:
            return sorted(results, key=_ranked, reverse=True)
        return top_k(results, limit, key=_ranked)

    async def run_query(
        self,
        query: str,
        target_language: str = "",
        target_version: str = "",
        symbols: list[str] | None = None,
    ) -> list[tuple]:
        """Execute one retrieval: phases → ladder → rank → top-k. No caching.

        The shared retrieval core behind ``search``, ``enrich_prompt``, and every
        strategy. Kept cache-free and exception-raising so callers own their own
        caching and degradation policy — ``search``/``enrich_prompt`` wrap it in
        their existing try/except-and-cache, strategies in ``_run_query``.
        """
        settings = get_settings()
        if not settings.enable_rag or not query.strip():
            return []
        if symbols is None:
            symbols = self._extract_code_signals(query, settings.rag_query_max_symbols)
        # An empty target language would build a `{"language": ""}` phase that
        # matches nothing, wasting a query before the ladder broadens — so with no
        # target we go straight to unfiltered (mirrors the old `search`).
        phases = (
            self._filter_phases(target_language, target_version, settings)
            if target_language
            else [None]
        )
        results = await self._retrieve_ladder(query, symbols, phases)
        return self._rank(results, target_version, settings, settings.rag_top_k)

    async def retrieve(
        self, request: RetrievalRequest, strategy: str | None = None
    ) -> list[tuple]:
        """Retrieve for ``request`` using a named strategy (default from settings).

        The single entry point for agentic retrieval: the VectorDBTool routes
        here by intent, and ``enrich_prompt`` routes here when a non-default
        ``rag_strategy`` is configured.
        """
        name = strategy or getattr(get_settings(), "rag_strategy", "single_hop")
        return await self._build_strategy(name).retrieve(request)

    def _build_strategy(self, name: str) -> RetrievalStrategy:
        """Resolve a strategy name to an instance (lazy imports avoid cycles).

        Unknown names fall back to single-hop rather than raising, so a bad
        config value degrades to the safe default instead of breaking retrieval.
        """
        key = (name or "single_hop").lower()
        if key in ("single_hop", "single", "none", ""):
            return SingleHopStrategy(self, self._llm)
        if key == "hyde":
            from rag.hyde import HyDEStrategy

            return HyDEStrategy(self, self._llm)
        if key in ("multi_query", "multiquery"):
            from rag.multi_query import MultiQueryStrategy

            return MultiQueryStrategy(self, self._llm)
        if key in ("multi_hop", "multihop"):
            from rag.multi_hop import MultiHopStrategy

            return MultiHopStrategy(self, self._llm)
        if key in ("contextual_compression", "compression"):
            from rag.contextual_compression import ContextualCompressionStrategy

            return ContextualCompressionStrategy(self, self._llm)
        if key in ("parent_document", "parent"):
            from rag.parent_retriever import ParentDocumentStrategy

            return ParentDocumentStrategy(self, self._llm)
        if key in ("corrective", "crag"):
            from rag.corrective_rag import CorrectiveRAGStrategy

            return CorrectiveRAGStrategy(self, self._llm)
        if key in ("self_rag", "selfrag"):
            from rag.self_rag import SelfRAGStrategy

            return SelfRAGStrategy(self, self._llm)
        log.warning("Unknown rag_strategy %r; using single_hop", name)
        return SingleHopStrategy(self, self._llm)

    @staticmethod
    def render_reference_context(hits: list[tuple]) -> str:
        """Render hits as the ``## Reference Examples`` block.

        The single source of truth for reference-context formatting — shared by
        ``enrich_prompt`` and ``VectorDBTool.format_context``. The MigratorAgent
        keys off the literal "Reference Examples" heading, so every retrieval path
        must render identically or downstream would silently drop the context.
        """
        settings = get_settings()
        parts = [
            "## Reference Examples\nHere are relevant code patterns from the "
            "target language:\n"
        ]
        for doc, score in hits:
            md = doc.metadata or {}
            lang = md.get("language", "unknown")
            version = md.get("version", "")
            doc_type = md.get("doc_type", "")
            label_bits = [lang]
            if version and version != settings.rag_version_wildcard:
                label_bits.append(version)
            if doc_type:
                label_bits.append(doc_type)
            parts.append(f"### {' · '.join(label_bits)} (relevance: {score:.2f})")
            parts.append(f"```{lang}\n{doc.page_content}\n```")
        return "\n\n".join(parts)

    async def search(
        self,
        query: str,
        target_language: str = "",
        target_version: str = "",
        symbols: list[str] | None = None,
    ) -> list[tuple]:
        """Retrieve against an explicitly supplied query. Returns ``(doc, score)``.

        ``enrich_prompt`` derives its query from the source code; this is the
        entry point for a caller that has *formulated its own* query — the
        VectorDBTool, which lets an agent ask a targeted question ("Java 21
        virtual threads replacement for ExecutorService") instead of taking
        whatever the code-signal heuristic produces.

        Shares the filter ladder, ranking, and cache with ``enrich_prompt`` so
        both paths retrieve identically once a query exists.
        """
        settings = get_settings()
        if not settings.enable_rag or not query.strip():
            return []

        # Symbols drive the keyword leg of hybrid retrieval; absent an explicit
        # list, mine them from the query itself so the leg still contributes.
        if symbols is None:
            symbols = self._extract_code_signals(query, settings.rag_query_max_symbols)

        cache_key = hashlib.sha256(
            f"search\n{target_language}\n{target_version}\n{query}".encode()
        ).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return cached

        try:
            results = await self.run_query(
                query, target_language, target_version, symbols
            )
        except Exception as e:
            log.warning("RAG search failed: %s", e)
            return []

        self._query_cache[cache_key] = results
        if len(self._query_cache) > self._max_cache:
            self._query_cache.popitem(last=False)
        return results

    async def enrich_prompt(
        self,
        source_language: str,
        target_language: str,
        source_code: str,
        base_prompt: str,
        target_version: str = "",
        code_metrics: dict | None = None,
        extra_terms: list[str] | None = None,
    ) -> str:
        settings = get_settings()
        if not settings.enable_rag:
            return base_prompt

        symbols = self._extract_code_signals(
            source_code, settings.rag_query_max_symbols
        )
        # Enrich raw code symbols with analyzer-detected constructs and any
        # targeted re-retrieval terms (feedback bus) so the query reflects what
        # the code actually does, not just its import lines.
        symbols = self._merge_query_terms(
            extra_terms, symbols, code_metrics, settings.rag_query_max_symbols
        )
        query = self._build_query(
            source_language, target_language, source_code, symbols
        )

        hits: list[tuple] | None = None
        # Agentic strategies are opt-in and LLM-driven: take that path only when a
        # non-default strategy is configured AND a client is wired. Any failure or
        # empty result falls through to the single-hop path below, so enrichment
        # is never worse than the default one-shot retrieval.
        strategy_name = getattr(settings, "rag_strategy", "single_hop")
        if strategy_name and strategy_name != "single_hop" and self._llm is not None:
            request = RetrievalRequest(
                query=query,
                target_language=target_language,
                target_version=target_version,
                symbols=symbols,
                source_language=source_language,
                source_code=source_code,
                code_metrics=code_metrics,
                extra_terms=extra_terms,
            )
            try:
                hits = await self.retrieve(request, strategy=strategy_name)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "RAG strategy %r failed, using single-hop: %s", strategy_name, e
                )
                hits = None

        if not hits:
            # Single-hop path (the default) — cached by version+query identity.
            # Target version steers both the filter ladder and the ranking boosts,
            # so it must be part of the cache identity. Re-retrieval terms and
            # metric constructs are already folded into `query`.
            cache_key = hashlib.sha256(
                f"{target_version}\n{query}".encode()
            ).hexdigest()
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                hits = cached
            else:
                try:
                    hits = await self.run_query(
                        query, target_language, target_version, symbols
                    )
                    self._query_cache[cache_key] = hits
                    if len(self._query_cache) > self._max_cache:
                        self._query_cache.popitem(last=False)
                except Exception as e:
                    log.warning("RAG retrieval failed: %s", e)
                    hits = []

        if not hits:
            return base_prompt

        context = self.render_reference_context(hits)
        return f"{context}\n\n---\n\n{base_prompt}"
