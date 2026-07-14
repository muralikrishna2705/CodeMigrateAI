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
