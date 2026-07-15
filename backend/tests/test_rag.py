"""Tests for the RAG pipeline components."""

import asyncio
import hashlib

from rag.embedding_service import CachedEmbeddings
from rag.retrieval_pipeline import RAGPipeline
from rag.splitter import LANGUAGE_SEPARATORS, DocSplitter


class TestDocSplitter:
    def test_language_separators_cover_all_languages(self):
        for lang in (
            "python",
            "java",
            "javascript",
            "typescript",
            "csharp",
            "go",
            "kotlin",
            "rust",
            "cpp",
        ):
            assert lang in LANGUAGE_SEPARATORS

    def test_splitter_defaults(self):
        splitter = DocSplitter()
        assert splitter.chunk_size == 2000
        assert splitter.chunk_overlap == 200

    def test_fallback_separator_for_unknown_language(self):
        splitter = DocSplitter()
        # Should not raise — falls back to DEFAULT_SEPARATORS.
        from langchain_core.documents import Document

        docs = [Document(page_content="hello\nworld\nfoo\nbar\nbaz")]
        chunks = splitter.split(docs, "unknown_lang")
        assert len(chunks) > 0
        assert all("chunk_size" in c.metadata for c in chunks)


class TestCachedEmbeddings:
    def test_fallback_embed_returns_768_dims(self):
        emb = CachedEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        vec = emb._fallback_embed("hello world")
        assert len(vec) == 768, f"Expected 768 dimensions, got {len(vec)}"
        assert all(v == 0.0 for v in vec), "Fallback should be all zeros"

    def test_fallback_embed_deterministic(self):
        emb = CachedEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        assert emb._fallback_embed("test") == emb._fallback_embed("test")

    def test_embed_query_returns_cached_value_without_calling_model(self):
        emb = CachedEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        key = hashlib.sha256("cache me".encode()).hexdigest()
        sentinel = [0.5] * 768
        emb._cache[key] = sentinel

        class _Boom:
            def embed_query(self, text):
                raise AssertionError("inner model must not be called on a cache hit")

        emb._inner = _Boom()
        assert emb.embed_query("cache me") == sentinel


class _FakeDoc:
    def __init__(self, content: str, language: str, **metadata):
        self.page_content = content
        self.metadata = {"language": language, **metadata}


def _parse_where(where):
    """Extract (language, versions_or_None) from either filter shape.

    Mirrors how a real Chroma filter is built: a flat ``{"language": x}`` for the
    language-only phase, or a compound ``{"$and": [{language}, {version $in}]}``
    for the version-scoped phase.
    """
    if where is None:
        return None, None
    if "$and" in where:
        lang = versions = None
        for clause in where["$and"]:
            if "language" in clause:
                lang = clause["language"]
            if "version" in clause:
                versions = clause["version"].get("$in")
        return lang, versions
    return where.get("language"), None


class _RecordingStore:
    """Fake VectorStore that records queries and returns canned results.

    ``results_by_version`` models the version-scoped phase (keyed by a version
    the ``$in`` clause asks for); ``results_by_language`` models the language
    phase; ``unfiltered`` models the final unfiltered phase.
    """

    def __init__(
        self, results_by_language=None, unfiltered=None, results_by_version=None
    ):
        self.calls = []
        self._by_lang = results_by_language or {}
        self._by_version = results_by_version or {}
        self._unfiltered = unfiltered or []

    def similarity_search(self, query, k=4, score_threshold=0.7, where=None):
        self.calls.append({"query": query, "k": k, "where": where})
        lang, versions = _parse_where(where)
        if versions is not None:
            for version in versions:
                if version in self._by_version:
                    return list(self._by_version[version])
            return []
        if lang is not None:
            return list(self._by_lang.get(lang, []))
        return list(self._unfiltered)


class _HybridStore:
    """Fake store with both a vector and a keyword leg for hybrid tests."""

    def __init__(self, vector_hits, keyword_hits):
        self._vector = vector_hits
        self._keyword = keyword_hits
        self.calls = {"vector": 0, "keyword": 0}

    def similarity_search(self, query, k=4, score_threshold=0.7, where=None):
        self.calls["vector"] += 1
        return list(self._vector)

    def keyword_search(self, symbols, k=4, where=None):
        self.calls["keyword"] += 1
        return list(self._keyword)


class _MockSettings:
    enable_rag = True
    rag_top_k = 4
    rag_min_score = 0.7
    rag_query_max_symbols = 12
    rag_query_code_chars = 600
    rag_filter_by_target_language = True
    rag_filter_by_target_version = True
    rag_version_wildcard = "any"
    rag_rank_weight_version = 0.15
    rag_rank_weight_official = 0.10
    rag_rank_weight_migration = 0.10
    rag_migration_doc_types = ["migration-guide", "release-notes", "deprecation"]
    rag_hybrid_enabled = True
    rag_rrf_k = 60


def _patch_settings(monkeypatch, **overrides):
    settings = _MockSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    monkeypatch.setattr("rag.retrieval_pipeline.get_settings", lambda: settings)
    return settings


class TestRAGPipeline:
    def test_returns_base_prompt_when_rag_disabled(self, monkeypatch):
        class MockSettings:
            enable_rag = False

        monkeypatch.setattr(
            "rag.retrieval_pipeline.get_settings", lambda: MockSettings()
        )
        pipeline = RAGPipeline(None, None)
        result = asyncio.run(
            pipeline.enrich_prompt("python", "java", "code", "base_prompt")
        )
        assert result == "base_prompt"

    def test_extract_code_signals_pulls_imports_and_skips_stopwords(self):
        pipeline = RAGPipeline(None, None)
        code = (
            "import requests\n"
            "from collections import OrderedDict\n"
            "if True:\n"
            "    resp = requests.get(url)\n"
            "    data = parse_response(resp)\n"
        )
        signals = pipeline._extract_code_signals(code, max_symbols=12)
        # Import targets and distinctive call/type names are captured.
        assert "requests" in signals
        assert "collections" in signals
        assert "parse_response" in signals
        assert "OrderedDict" in signals
        # Language keywords carry no retrieval signal and are dropped.
        assert "if" not in signals
        assert "True" not in signals
        assert "import" not in signals

    def test_build_query_is_code_aware(self, monkeypatch):
        _patch_settings(monkeypatch)
        pipeline = RAGPipeline(None, None)
        code = "import pandas as pd\n\nframe = DataFrame()\nstats = compute_stats(frame)\n"
        symbols = pipeline._extract_code_signals(code, max_symbols=12)
        query = pipeline._build_query("python", "java", code, symbols)
        assert "python to java migration" in query
        # The query now depends on the actual code, not just the language pair.
        assert "pandas" in query
        assert "DataFrame" in query
        assert "compute_stats" in query

    def test_enrich_prompt_filters_to_target_language(self, monkeypatch):
        _patch_settings(monkeypatch)
        doc = _FakeDoc("System.out.println();", "java")
        store = _RecordingStore(results_by_language={"java": [(doc, 0.91)]})
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt("python", "java", "print('x')", "BASE")
        )
        assert "Reference Examples" in out
        assert "BASE" in out
        assert store.calls[0]["where"] == {"language": "java"}

    def test_enrich_prompt_falls_back_when_target_corpus_empty(self, monkeypatch):
        _patch_settings(monkeypatch)
        doc = _FakeDoc("print('x')", "python")
        # No java docs -> filtered search is empty -> retry unfiltered.
        store = _RecordingStore(results_by_language={}, unfiltered=[(doc, 0.8)])
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt("python", "java", "print('x')", "BASE")
        )
        assert "Reference Examples" in out
        assert len(store.calls) == 2
        assert store.calls[0]["where"] == {"language": "java"}
        assert store.calls[1]["where"] is None

    def test_enrich_prompt_returns_base_when_no_results(self, monkeypatch):
        _patch_settings(monkeypatch)
        store = _RecordingStore(results_by_language={}, unfiltered=[])
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt("python", "java", "print('x')", "BASE")
        )
        assert out == "BASE"


class TestHybridRetrieval:
    def test_rrf_merge_orders_by_fused_rank(self):
        doc_a = _FakeDoc("A", "java")
        doc_b = _FakeDoc("B", "java")
        doc_c = _FakeDoc("C", "java")
        vector = [(doc_a, 0.9), (doc_b, 0.8)]
        keyword = [(doc_b, 1.0), (doc_c, 0.5)]
        merged = RAGPipeline._rrf_merge(vector, keyword, k=3, rrf_k=60)
        keys = [doc.page_content for doc, _ in merged]
        # B ranks in both legs -> highest fused score -> first.
        assert keys[0] == "B"
        assert set(keys) == {"A", "B", "C"}
        # A doc from the vector leg keeps its calibrated cosine score.
        by_key = {doc.page_content: score for doc, score in merged}
        assert by_key["A"] == 0.9

    def test_hybrid_surfaces_keyword_only_doc(self, monkeypatch):
        _patch_settings(monkeypatch)
        vec_doc = _FakeDoc("vector doc using Foo", "java")
        kw_doc = _FakeDoc("keyword doc using BarBaz", "java")
        store = _HybridStore(vector_hits=[(vec_doc, 0.82)], keyword_hits=[(kw_doc, 1.0)])
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt("python", "java", "BarBaz()", "BASE")
        )
        assert "vector doc using Foo" in out
        # The keyword leg surfaced a doc the vector leg missed.
        assert "keyword doc using BarBaz" in out
        assert store.calls["keyword"] == 1

    def test_hybrid_disabled_is_vector_only(self, monkeypatch):
        _patch_settings(monkeypatch, rag_hybrid_enabled=False)
        vec_doc = _FakeDoc("vector only", "java")
        kw_doc = _FakeDoc("keyword only", "java")
        store = _HybridStore(vector_hits=[(vec_doc, 0.82)], keyword_hits=[(kw_doc, 1.0)])
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(pipeline.enrich_prompt("python", "java", "x()", "BASE"))
        assert "vector only" in out
        assert "keyword only" not in out
        assert store.calls["keyword"] == 0


class TestDocClassification:
    """Metadata derived from the corpus layout during ingestion."""

    def test_flat_examples_get_wildcard_version_and_example_type(self):
        from rag.loaders import derive_doc_metadata

        meta = derive_doc_metadata([], "examples")
        assert meta == {
            "version": "any",
            "doc_type": "example",
            "is_official": False,
        }

    def test_version_and_doc_type_parsed_from_path(self):
        from rag.loaders import derive_doc_metadata

        meta = derive_doc_metadata(["3.12", "migration-guide"], "examples")
        assert meta["version"] == "3.12"
        assert meta["doc_type"] == "migration-guide"
        assert meta["is_official"] is False

    def test_fetched_docs_are_official_reference(self):
        from rag.loaders import derive_doc_metadata

        meta = derive_doc_metadata([], "_fetched")
        assert meta["doc_type"] == "reference"
        assert meta["is_official"] is True

    def test_non_version_non_doctype_segment_is_ignored(self):
        from rag.loaders import derive_doc_metadata

        meta = derive_doc_metadata(["collections"], "examples")
        assert meta["version"] == "any"
        assert meta["doc_type"] == "example"

    def test_es_style_version_is_recognised(self):
        from rag.loaders import derive_doc_metadata

        meta = derive_doc_metadata(["ES2020"], "user")
        assert meta["version"] == "ES2020"
        assert meta["doc_type"] == "user"

    def test_versioned_fetched_path_is_official_and_migration_typed(self):
        # The layout the versioned fetcher writes: _fetched/<version>/<doc_type>/.
        # It must yield a real version, an authoritative doc_type, and official
        # authority — the three signals that drive the version-aware ladder.
        from config import get_settings
        from rag.loaders import derive_doc_metadata
        from rag.url_index import VERSIONED_DOC_TYPE

        meta = derive_doc_metadata(["3.12", VERSIONED_DOC_TYPE], "_fetched")
        assert meta["version"] == "3.12"
        assert meta["doc_type"] == VERSIONED_DOC_TYPE
        assert meta["is_official"] is True
        assert meta["doc_type"] in get_settings().rag_migration_doc_types


class TestVersionAwareRetrieval:
    def test_version_phase_runs_first(self, monkeypatch):
        _patch_settings(monkeypatch)
        exact = _FakeDoc("exact v21 doc", "java", version="21")
        store = _RecordingStore(results_by_version={"21": [(exact, 0.88)]})
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt(
                "python", "java", "x()", "BASE", target_version="21"
            )
        )
        assert "exact v21 doc" in out
        lang, versions = _parse_where(store.calls[0]["where"])
        assert lang == "java"
        assert versions == ["21", "any"]

    def test_falls_back_to_language_when_version_empty(self, monkeypatch):
        _patch_settings(monkeypatch)
        langdoc = _FakeDoc("any-version java doc", "java", version="17")
        store = _RecordingStore(
            results_by_language={"java": [(langdoc, 0.8)]},
            results_by_version={},  # nothing satisfies the version filter
        )
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt(
                "python", "java", "x()", "BASE", target_version="21"
            )
        )
        assert "any-version java doc" in out
        # Version phase (empty) then language-only phase.
        assert len(store.calls) == 2
        assert _parse_where(store.calls[0]["where"])[1] == ["21", "any"]
        assert _parse_where(store.calls[1]["where"]) == ("java", None)

    def test_version_filter_disabled_skips_version_phase(self, monkeypatch):
        _patch_settings(monkeypatch, rag_filter_by_target_version=False)
        langdoc = _FakeDoc("java doc", "java")
        store = _RecordingStore(results_by_language={"java": [(langdoc, 0.8)]})
        pipeline = RAGPipeline(store, None)

        asyncio.run(
            pipeline.enrich_prompt(
                "python", "java", "x()", "BASE", target_version="21"
            )
        )
        # First (and only needed) phase is the flat language filter.
        assert store.calls[0]["where"] == {"language": "java"}

    def test_ranking_lifts_exact_version_and_official_docs(self, monkeypatch):
        _patch_settings(monkeypatch)
        generic = _FakeDoc(
            "generic example", "java", version="any", doc_type="example"
        )
        authoritative = _FakeDoc(
            "official v21 guide",
            "java",
            version="21",
            doc_type="migration-guide",
            is_official=True,
        )
        # Generic has the higher RAW cosine; authority boosts must still win.
        store = _RecordingStore(
            results_by_version={"21": [(generic, 0.90), (authoritative, 0.82)]}
        )
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt(
                "python", "java", "x()", "BASE", target_version="21"
            )
        )
        assert out.index("official v21 guide") < out.index("generic example")

    def test_displayed_score_stays_raw_relevance(self, monkeypatch):
        _patch_settings(monkeypatch)
        doc = _FakeDoc(
            "boosted", "java", version="21", doc_type="migration-guide",
            is_official=True,
        )
        store = _RecordingStore(results_by_version={"21": [(doc, 0.80)]})
        pipeline = RAGPipeline(store, None)

        out = asyncio.run(
            pipeline.enrich_prompt(
                "python", "java", "x()", "BASE", target_version="21"
            )
        )
        # Boosts change ordering only; the shown relevance is the raw 0.80.
        assert "relevance: 0.80" in out
        # And version/doc_type surface in the reference label for the model.
        assert "21" in out and "migration-guide" in out


class TestVersionedDocRegistry:
    """The per-version official-doc registry that feeds the retrieval ladder."""

    def test_version_keys_match_selectable_target_versions(self):
        # If a registry version key is not a version the UI can send as
        # target_version, the version-filtered leg can NEVER match its docs —
        # the whole ladder would silently collapse back to language-only.
        from config import get_settings
        from rag.url_index import VERSIONED_DOC_URLS

        supported = {
            lang["id"]: set(lang["versions"])
            for lang in get_settings().supported_languages
        }
        for lang, versions in VERSIONED_DOC_URLS.items():
            assert lang in supported, f"{lang} is not a supported language"
            for version in versions:
                assert version in supported[lang], (
                    f"{lang} {version} is not a selectable target version"
                )

    def test_doc_type_earns_the_migration_ranking_boost(self):
        from config import get_settings
        from rag.url_index import VERSIONED_DOC_TYPE

        assert VERSIONED_DOC_TYPE in get_settings().rag_migration_doc_types

    def test_every_entry_has_official_https_urls(self):
        from rag.url_index import VERSIONED_DOC_URLS

        for lang, versions in VERSIONED_DOC_URLS.items():
            for version, urls in versions.items():
                assert urls, f"{lang} {version} has no URLs"
                for url in urls:
                    assert url.startswith("https://"), f"{lang} {version}: {url}"


class TestVersionedDocFetch:
    """End-to-end (offline) proof that fetching activates version metadata."""

    def test_fetch_lands_in_version_dir_and_loader_tags_it(
        self, tmp_path, monkeypatch
    ):
        from rag import web_doc_fetcher
        from rag.loaders import DocLoader
        from rag.url_index import VERSIONED_DOC_TYPE

        # Redirect the corpus root and stub out network + rate-limit sleeps.
        monkeypatch.setattr(web_doc_fetcher, "REFERENCE_DIR", tmp_path)

        async def _no_sleep(*_a, **_k):
            return None

        monkeypatch.setattr(web_doc_fetcher.asyncio, "sleep", _no_sleep)

        async def _run_fetch():
            fetcher = web_doc_fetcher.WebDocFetcher()

            async def _fake_convert(url):
                return f"# What's New\nContent for {url}\n"

            fetcher._fetch_and_convert = _fake_convert
            versioned = {"3.12": ["https://docs.python.org/3/whatsnew/3.12.html"]}
            saved = await fetcher.fetch_versioned(
                "python", versioned, VERSIONED_DOC_TYPE
            )
            await fetcher.close()
            return saved

        saved = asyncio.run(_run_fetch())

        # 1. The doc landed under _fetched/<version>/<doc_type>/.
        expected_dir = tmp_path / "python" / "_fetched" / "3.12" / VERSIONED_DOC_TYPE
        md_files = list(expected_dir.glob("*.md"))
        assert len(md_files) == 1
        assert saved == md_files

        # 2. Loading that corpus stamps the real version + authoritative doc_type
        #    that the retrieval ladder and ranking boosts key on.
        loader = DocLoader()
        loader._built_in_dir = tmp_path
        docs = asyncio.run(loader.load_source("fetched", "python"))
        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["language"] == "python"
        assert meta["version"] == "3.12"
        assert meta["doc_type"] == VERSIONED_DOC_TYPE
        assert meta["is_official"] is True
