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
    def __init__(self, content: str, language: str):
        self.page_content = content
        self.metadata = {"language": language}


class _RecordingStore:
    """Fake VectorStore that records queries and returns canned results."""

    def __init__(self, results_by_language=None, unfiltered=None):
        self.calls = []
        self._by_lang = results_by_language or {}
        self._unfiltered = unfiltered or []

    def similarity_search(self, query, k=4, score_threshold=0.7, where=None):
        self.calls.append({"query": query, "k": k, "where": where})
        if where is not None:
            return self._by_lang.get(where.get("language"), [])
        return self._unfiltered


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
