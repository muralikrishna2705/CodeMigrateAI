"""Tests for the agentic-RAG layer: reranking, strategy routing, query hygiene.

These cover the three changes that decide *what reaches the prompt*: a
cross-encoder re-scoring candidates, the model choosing how to search, and
comments being kept out of the retrieval query.
"""

import asyncio

import pytest
from config import Settings
from langchain_core.documents import Document
from rag import reranker
from rag.retrieval_pipeline import RAGPipeline, RetrievalRequest, _strip_comments


class _FakeReranker:
    """Stands in for FlashrankRerank without loading a model.

    Scores by term overlap so ordering is deterministic and the tests assert on
    behaviour rather than on a particular cross-encoder's opinion.
    """

    def __init__(self, top_n=4):
        self.top_n = top_n
        self.calls: list[tuple[str, int]] = []

    def compress_documents(self, docs, query):
        self.calls.append((query, len(docs)))
        terms = set(query.lower().split())

        def score(doc):
            words = set(doc.page_content.lower().split())
            return len(terms & words) / (len(terms) or 1)

        ranked = sorted(docs, key=score, reverse=True)[: self.top_n]
        return [
            Document(
                page_content=d.page_content,
                metadata={**(d.metadata or {}), "relevance_score": score(d)},
            )
            for d in ranked
        ]


def _doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


@pytest.fixture(autouse=True)
def _reset_reranker():
    reranker.reset()
    yield
    reranker.reset()


class TestReranking:
    def test_disabled_leaves_order_untouched(self, monkeypatch):
        monkeypatch.setattr(
            "rag.reranker.get_settings", lambda: Settings(rag_rerank_enabled=False)
        )
        hits = [(_doc("a"), 0.9), (_doc("b"), 0.8)]
        assert asyncio.run(reranker.rerank(hits, "query")) == hits

    def test_single_hit_short_circuits(self, monkeypatch):
        # Nothing to reorder, so don't pay for the model.
        monkeypatch.setattr("rag.reranker.get_reranker", lambda: _FakeReranker())
        hits = [(_doc("only"), 0.5)]
        assert asyncio.run(reranker.rerank(hits, "q")) == hits

    def test_reorders_by_query_relevance(self, monkeypatch):
        fake = _FakeReranker()
        monkeypatch.setattr("rag.reranker.get_reranker", lambda: fake)
        # Floor at 0 so this isolates *ordering*; the floor has its own tests.
        monkeypatch.setattr(
            "rag.reranker.get_settings", lambda: Settings(rag_rerank_min_score=0.0)
        )

        # The irrelevant doc is *first* by incoming score: this is exactly the
        # case fusion gets wrong, since RRF combines positions and cannot tell
        # that a keyword hit on one incidental symbol was incidental.
        hits = [
            (_doc("css flexbox aligns items"), 0.95),
            (_doc("java executorservice thread pool"), 0.60),
        ]
        out = asyncio.run(reranker.rerank(hits, "java executorservice"))
        assert "java" in out[0][0].page_content
        assert out[0][1] > out[1][1]

    def test_hits_below_the_floor_are_dropped_not_just_demoted(self, monkeypatch):
        """The cross-encoder's verdict must filter, not only reorder.

        ``rag_min_score`` is a cosine floor applied inside the vector store,
        before reranking. Without a floor here, a chunk clears that, gets
        re-scored as irrelevant, and still reaches the prompt at the bottom of
        the list — and because retrieval fetches 20 candidates to keep 4, that
        tail is padding whenever the corpus holds fewer than 4 good answers.
        """
        monkeypatch.setattr("rag.reranker.get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(
            "rag.reranker.get_settings", lambda: Settings(rag_rerank_min_score=0.2)
        )
        hits = [
            (_doc("java executorservice thread pool"), 0.9),
            (_doc("unrelated css flexbox notes"), 0.8),
        ]
        out = asyncio.run(reranker.rerank(hits, "java executorservice"))
        assert len(out) == 1
        assert "java" in out[0][0].page_content

    def test_an_entirely_irrelevant_query_retrieves_nothing(self, monkeypatch):
        # Returning nothing is a real answer: the caller emits the ungrounded
        # notice rather than inventing grounding from off-topic text.
        monkeypatch.setattr("rag.reranker.get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(
            "rag.reranker.get_settings", lambda: Settings(rag_rerank_min_score=0.2)
        )
        hits = [(_doc("java executorservice"), 0.9), (_doc("java collections"), 0.8)]
        assert asyncio.run(reranker.rerank(hits, "chocolate cake recipe")) == []

    def test_a_zero_floor_keeps_every_reranked_hit(self, monkeypatch):
        monkeypatch.setattr("rag.reranker.get_reranker", lambda: _FakeReranker())
        monkeypatch.setattr(
            "rag.reranker.get_settings", lambda: Settings(rag_rerank_min_score=0.0)
        )
        hits = [(_doc("alpha"), 0.9), (_doc("beta"), 0.8)]
        assert len(asyncio.run(reranker.rerank(hits, "gamma"))) == 2

    def test_failure_degrades_to_the_incoming_order(self, monkeypatch):
        class _Boom:
            top_n = 4

            def compress_documents(self, docs, query):
                raise RuntimeError("model missing")

        monkeypatch.setattr("rag.reranker.get_reranker", lambda: _Boom())
        monkeypatch.setattr("rag.reranker.get_settings", lambda: Settings())
        hits = [(_doc("a"), 0.9), (_doc("b"), 0.8)]
        # Retrieval must never end up *worse* than it was without reranking.
        assert asyncio.run(reranker.rerank(hits, "q")) == hits

    def test_unavailable_reranker_is_not_retried_every_query(self, monkeypatch):
        attempts = []

        def _explode(*a, **k):
            attempts.append(1)
            raise ImportError("flashrank not installed")

        monkeypatch.setattr("rag.reranker.get_settings", lambda: Settings())
        monkeypatch.setitem(
            __import__("sys").modules,
            "langchain_community.document_compressors",
            type("m", (), {"FlashrankRerank": _explode}),
        )
        assert reranker.get_reranker() is None
        assert reranker.get_reranker() is None
        # Caching the failure matters: otherwise every query pays the load
        # attempt's latency to fail again.
        assert len(attempts) == 1


class TestCandidateWidth:
    def test_reranking_widens_the_candidate_pool(self):
        settings = Settings(rag_rerank_enabled=True, rag_top_k=4, rag_rerank_candidates=20)
        assert RAGPipeline._candidate_k(settings) == 20

    def test_without_reranking_the_store_order_is_final(self):
        # Fetching extra would only pad the prompt with worse matches.
        settings = Settings(rag_rerank_enabled=False, rag_top_k=4, rag_rerank_candidates=20)
        assert RAGPipeline._candidate_k(settings) == 4

    def test_top_k_is_never_reduced_by_a_small_candidate_setting(self):
        settings = Settings(rag_rerank_enabled=True, rag_top_k=8, rag_rerank_candidates=2)
        assert RAGPipeline._candidate_k(settings) == 8


class _RoutingLLM:
    """Client whose structured output returns a scripted routing decision."""

    def __init__(self, decision=None, raises=False):
        self.decision = decision
        self.raises = raises
        self.calls = 0

    def chat_model(self, role="main", *, json_mode=False):
        outer = self

        class _M:
            def with_structured_output(self, schema):
                return self

            async def ainvoke(self, prompt):
                outer.calls += 1
                if outer.raises:
                    raise RuntimeError("provider down")
                return outer.decision

        return _M()


def _request():
    return RetrievalRequest(
        query="how do I replace ExecutorService",
        target_language="java",
        target_version="21",
        source_language="java",
    )


class TestStrategyRouting:
    def _pipeline(self, llm):
        return RAGPipeline(vector_store=None, embedding_service=None, llm_client=llm)

    def test_model_choice_is_honoured(self):
        from models.schemas import RetrievalRouteDecision

        llm = _RoutingLLM(RetrievalRouteDecision(strategy="hyde", needs_retrieval=True))
        chosen = asyncio.run(self._pipeline(llm)._route_strategy(_request()))
        assert chosen == "hyde"

    def test_model_can_decline_retrieval_entirely(self):
        # The only choice that removes latency rather than adding it.
        from models.schemas import RetrievalRouteDecision

        llm = _RoutingLLM(
            RetrievalRouteDecision(strategy="single_hop", needs_retrieval=False)
        )
        assert asyncio.run(self._pipeline(llm)._route_strategy(_request())) == "skip"

    def test_routing_failure_falls_back_to_single_hop(self):
        llm = _RoutingLLM(raises=True)
        assert (
            asyncio.run(self._pipeline(llm)._route_strategy(_request())) == "single_hop"
        )

    def test_client_without_a_chat_model_does_not_route(self):
        llm = object()
        pipeline = self._pipeline(llm)
        assert asyncio.run(pipeline._route_strategy(_request())) == "single_hop"

    def test_skip_resolves_to_a_strategy_that_retrieves_nothing(self):
        pipeline = self._pipeline(None)
        strategy = pipeline._build_strategy("skip")
        assert asyncio.run(strategy.retrieve(_request())) == []

    def test_a_skip_is_not_undone_by_the_fallback(self, monkeypatch):
        # enrich_prompt falls back to single-hop when a strategy returns nothing.
        # A deliberate skip must be distinguishable from "searched, found
        # nothing", or the fallback helpfully undoes every skip.
        from models.schemas import RetrievalRouteDecision

        llm = _RoutingLLM(
            RetrievalRouteDecision(strategy="single_hop", needs_retrieval=False)
        )
        pipeline = self._pipeline(llm)
        monkeypatch.setattr(
            "rag.retrieval_pipeline.get_settings",
            lambda: Settings(rag_strategy="auto", rag_rerank_enabled=False),
        )

        async def _boom(*a, **k):
            raise AssertionError("skip must not fall through to retrieval")

        monkeypatch.setattr(pipeline, "run_query", _boom)
        result = asyncio.run(
            pipeline.enrich_prompt("java", "java", "class A {}", "BASE", "21")
        )
        assert result == "BASE"


class TestContextualChunkHeaders:
    """A chunk from the middle of a document must be able to stand alone."""

    def _split(self, text, **metadata):
        from rag.splitter import DocSplitter

        doc = Document(page_content=text, metadata=metadata)
        return DocSplitter(chunk_size=120, chunk_overlap=0).split([doc], "python")

    def test_later_chunks_carry_language_version_and_type(self):
        chunks = self._split(
            "alpha " * 60,
            language="python",
            version="3.12",
            doc_type="migration-guide",
            source="/corpus/python/java-to-python.md",
        )
        assert len(chunks) > 1
        header = chunks[1].page_content.splitlines()[0]
        # Without this the embedding is computed from text missing its own
        # subject, and the model reads a fragment with no idea what it is about.
        assert "python" in header and "3.12" in header
        assert "migration guide" in header
        assert "java-to-python.md" in header

    def test_first_chunk_is_not_prefixed(self):
        # It already opens with the document's own title.
        chunks = self._split("beta " * 60, language="python", version="3.12")
        assert not chunks[0].page_content.startswith("[")

    def test_wildcard_version_is_omitted_from_the_header(self):
        # "any" is the absence of a version, not a version worth stating.
        chunks = self._split("gamma " * 60, language="python", version="any")
        assert "any" not in chunks[1].page_content.splitlines()[0]

    def test_only_the_filename_appears_not_the_full_path(self):
        chunks = self._split(
            "delta " * 60, language="go", source="/very/long/corpus/path/guide.md"
        )
        header = chunks[1].page_content.splitlines()[0]
        assert "guide.md" in header and "/very/long" not in header


class TestQueryHygiene:
    def test_comments_are_stripped_before_symbol_extraction(self):
        code = (
            "/*\n * Copyright 2019 Apache Software Foundation.\n"
            " * Licensed under the Apache License, Version 2.0.\n */\n"
            "import java.util.concurrent.ExecutorService;\n"
            "class Worker { void run() { pool.submit(task); } }\n"
        )
        signals = RAGPipeline(None, None)._extract_code_signals(code, max_symbols=12)
        # A licence header is prose, and the CamelCase type pattern misfires on
        # it. Because the cap keeps symbols in source order, a header at the top
        # of the file could otherwise consume the entire budget.
        assert "Copyright" not in signals
        assert "Apache" not in signals
        assert "java.util.concurrent.ExecutorService" in signals

    def test_python_docstrings_do_not_become_symbols(self):
        code = '"""Migrate the Legacy Widget Factory to the new API."""\nimport asyncio\n'
        signals = RAGPipeline(None, None)._extract_code_signals(code, max_symbols=12)
        assert "Legacy" not in signals and "Widget" not in signals
        assert "asyncio" in signals

    def test_cpp_includes_survive_the_hash_comment_rule(self):
        # `#include` starts with '#' but is the highest-signal line in a C++ file.
        assert "#include <vector>" in _strip_comments("#include <vector>\n# a comment\n")
        assert "a comment" not in _strip_comments("#include <vector>\n# a comment\n")
