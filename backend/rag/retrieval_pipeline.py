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
        self, source_language: str, target_language: str, source_code: str
    ) -> str:
        """Compose a code-aware retrieval query from the source code + languages."""
        settings = get_settings()
        symbols = self._extract_code_signals(
            source_code, settings.rag_query_max_symbols
        )

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

        query = self._build_query(source_language, target_language, source_code)

        cache_key = hashlib.sha256(query.encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
        else:
            try:
                where = (
                    {"language": target_language}
                    if settings.rag_filter_by_target_language
                    else None
                )
                results = await self._search(query, where)
                # Prefer target-language examples, but never regress to zero
                # grounding: if that corpus is empty, retry unfiltered.
                if not results and where is not None:
                    results = await self._search(query, None)
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
