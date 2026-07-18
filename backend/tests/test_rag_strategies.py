"""Tests for the agentic RAG strategies (Dimension 2).

Each strategy is exercised for its happy path and for graceful degradation (no
LLM / explicit rejection / disabled), plus the dispatch factory and the
VectorDBTool intent routing. Strategies talk to a fake pipeline (canned
``run_query``) and a fake LLM (scripted responses), so nothing here needs Ollama
or Chroma.
"""

import asyncio
import json
from types import SimpleNamespace

from agents.tools.vector_db import VectorDBTool
from rag.contextual_compression import ContextualCompressionStrategy
from rag.corrective_rag import CorrectiveRAGStrategy
from rag.hyde import HyDEStrategy
from rag.multi_hop import MultiHopStrategy
from rag.multi_query import MultiQueryStrategy
from rag.parent_retriever import ParentDocumentStrategy
from rag.retrieval_pipeline import RAGPipeline, RetrievalRequest
from rag.self_rag import SelfRAGStrategy


# --- Test doubles ----------------------------------------------------------


class _FakeDoc:
    def __init__(self, content, language="java", **metadata):
        self.page_content = content
        self.metadata = {"language": language, **metadata}


class _FakePipeline:
    """Canned ``run_query``: returns hits by query-substring, else a default."""

    def __init__(self, results_by_query=None, default=None, vector_store=None):
        self.queries = []
        self._results_by_query = results_by_query or {}
        self._default = default if default is not None else []
        self.vector_store = vector_store

    async def run_query(self, query, target_language="", target_version="", symbols=None):
        self.queries.append(query)
        for needle, hits in self._results_by_query.items():
            if needle in query:
                return list(hits)
        return list(self._default)


class _ScriptedLLM:
    """Returns the first rule whose needle is in the prompt, else a default."""

    fast_model = "fake-fast"

    def __init__(self, rules=None, default=""):
        self.rules = rules or []
        self.default = default
        self.calls = []

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None):
        self.calls.append(prompt)
        for needle, response in self.rules:
            if needle in prompt:
                return response
        return self.default

    def extract_json(self, raw):
        return json.loads(raw)


class _CallableLLM:
    """LLM whose response is computed by a handler(prompt, fmt) callable."""

    fast_model = "fake-fast"

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None):
        self.calls.append((prompt, fmt))
        return self._handler(prompt, fmt)

    def extract_json(self, raw):
        return json.loads(raw)


class _FakeStore:
    def __init__(self, siblings_by_parent):
        self._siblings = siblings_by_parent

    def get_by_metadata(self, where, limit=50):
        return list(self._siblings.get(where.get("parent_id"), []))


class _FakeWebTool:
    def __init__(self, results):
        self._results = results
        self.called = False

    async def __call__(self, query="", language="", fetch_content=False):
        self.called = True
        return SimpleNamespace(success=True, data={"results": self._results})


class _Settings:
    """Only the knobs the strategies read; overridden per test."""

    rag_top_k = 4
    rag_hyde_enabled = True
    rag_multi_query_count = 5
    rag_multi_hop_max_subqueries = 4
    rag_multi_hop_max_depth = 2
    rag_compression_enabled = False
    rag_crag_relevance_threshold = 0.5
    rag_crag_web_fallback = False
    rag_self_rag_enabled = False


def _patch(monkeypatch, module, **overrides):
    settings = _Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    monkeypatch.setattr(f"rag.{module}.get_settings", lambda: settings)
    return settings


# --- HyDE ------------------------------------------------------------------


class TestHyDE:
    def test_embeds_hypothetical_answer_in_query(self, monkeypatch):
        _patch(monkeypatch, "hyde")
        doc = _FakeDoc("answer")
        pipe = _FakePipeline(default=[(doc, 0.9)])
        llm = _ScriptedLLM(
            default="var e = Executors.newVirtualThreadPerTaskExecutor();"
        )
        strat = HyDEStrategy(pipe, llm)
        req = RetrievalRequest(query="replace ExecutorService", target_language="java")

        hits = asyncio.run(strat.retrieve(req))

        assert hits == [(doc, 0.9)]
        assert len(pipe.queries) == 1
        # Both the original query and the hypothetical snippet steer retrieval.
        assert "replace ExecutorService" in pipe.queries[0]
        assert "VirtualThread" in pipe.queries[0]

    def test_no_llm_falls_back_to_plain_query(self, monkeypatch):
        _patch(monkeypatch, "hyde")
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.8)])
        strat = HyDEStrategy(pipe, None)

        asyncio.run(strat.retrieve(RetrievalRequest(query="q", target_language="java")))

        assert pipe.queries == ["q"]

    def test_disabled_skips_llm(self, monkeypatch):
        _patch(monkeypatch, "hyde", rag_hyde_enabled=False)
        pipe = _FakePipeline(default=[])
        llm = _ScriptedLLM(default="hypothetical")
        strat = HyDEStrategy(pipe, llm)

        asyncio.run(strat.retrieve(RetrievalRequest(query="q", target_language="java")))

        assert pipe.queries == ["q"]
        assert llm.calls == []


# --- Multi-Query -----------------------------------------------------------


class TestMultiQuery:
    def test_fans_out_and_merges_unique(self, monkeypatch):
        _patch(monkeypatch, "multi_query")
        orig, a, b = _FakeDoc("orig"), _FakeDoc("A"), _FakeDoc("B")
        pipe = _FakePipeline(
            results_by_query={"about A": [(a, 0.8)], "about B": [(b, 0.7)]},
            default=[(orig, 0.9)],
        )
        llm = _ScriptedLLM(default="query about A\nquery about B")
        strat = MultiQueryStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="original", target_language="java"))
        )

        assert {d.page_content for d, _ in hits} == {"orig", "A", "B"}
        # Highest score first (fusion keeps best score, ranked desc).
        assert hits[0][0].page_content == "orig"
        assert len(pipe.queries) == 3  # original + two variants

    def test_no_llm_single_query(self, monkeypatch):
        _patch(monkeypatch, "multi_query")
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.9)])
        strat = MultiQueryStrategy(pipe, None)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert len(pipe.queries) == 1
        assert [d.page_content for d, _ in hits] == ["x"]


# --- Multi-hop -------------------------------------------------------------


class TestMultiHop:
    def test_decomposes_and_visits_each_subquestion(self, monkeypatch):
        _patch(
            monkeypatch,
            "multi_hop",
            rag_top_k=10,
            rag_multi_hop_max_subqueries=4,
            rag_multi_hop_max_depth=1,
        )
        d1, d2 = _FakeDoc("d1"), _FakeDoc("d2")
        pipe = _FakePipeline(
            results_by_query={"sub one": [(d1, 0.8)], "sub two": [(d2, 0.7)]},
            default=[],
        )
        llm = _ScriptedLLM(rules=[("Break this", "sub one\nsub two")])
        strat = MultiHopStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="compound", target_language="java"))
        )

        assert {"d1", "d2"} <= {d.page_content for d, _ in hits}
        assert "sub one" in pipe.queries
        assert "sub two" in pipe.queries

    def test_node_budget_bounds_traversal(self, monkeypatch):
        _patch(
            monkeypatch,
            "multi_hop",
            rag_top_k=10,
            rag_multi_hop_max_subqueries=2,
            rag_multi_hop_max_depth=2,
        )
        pipe = _FakePipeline(default=[])
        llm = _ScriptedLLM(rules=[("Break this", "a\nb\nc")])
        strat = MultiHopStrategy(pipe, llm)

        asyncio.run(strat.retrieve(RetrievalRequest(query="q", target_language="java")))

        # Only 2 of the 3 sub-questions are expanded (budget), plus the original.
        assert "a" in pipe.queries and "b" in pipe.queries
        assert "c" not in pipe.queries
        assert "q" in pipe.queries

    def test_no_llm_is_single_hop(self, monkeypatch):
        _patch(monkeypatch, "multi_hop")
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.9)])
        strat = MultiHopStrategy(pipe, None)

        asyncio.run(strat.retrieve(RetrievalRequest(query="q", target_language="java")))

        assert pipe.queries == ["q"]


# --- Contextual compression ------------------------------------------------


class TestCompression:
    def test_extracts_relevant_lines(self, monkeypatch):
        _patch(monkeypatch, "contextual_compression", rag_compression_enabled=True)
        doc = _FakeDoc("line1 relevant\nline2 noise")
        pipe = _FakePipeline(default=[(doc, 0.9)])
        llm = _ScriptedLLM(default="line1 relevant")
        strat = ContextualCompressionStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert hits[0][0].page_content == "line1 relevant"

    def test_drops_doc_on_none_sentinel(self, monkeypatch):
        _patch(monkeypatch, "contextual_compression", rag_compression_enabled=True)
        pipe = _FakePipeline(default=[(_FakeDoc("all noise"), 0.9)])
        llm = _ScriptedLLM(default="NONE")
        strat = ContextualCompressionStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert hits == []

    def test_disabled_returns_raw_docs(self, monkeypatch):
        _patch(monkeypatch, "contextual_compression", rag_compression_enabled=False)
        doc = _FakeDoc("full content")
        pipe = _FakePipeline(default=[(doc, 0.9)])
        llm = _ScriptedLLM(default="short")
        strat = ContextualCompressionStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert hits[0][0].page_content == "full content"
        assert llm.calls == []


# --- Parent document -------------------------------------------------------


class TestParentDocument:
    def test_stitches_siblings_in_order(self, monkeypatch):
        _patch(monkeypatch, "parent_retriever")
        child = _FakeDoc("chunk B", parent_id="p1", chunk_index=1)
        sib0 = _FakeDoc("chunk A", parent_id="p1", chunk_index=0)
        sib1 = _FakeDoc("chunk B", parent_id="p1", chunk_index=1)
        store = _FakeStore({"p1": [sib1, sib0]})  # deliberately out of order
        pipe = _FakePipeline(default=[(child, 0.9)], vector_store=store)
        strat = ParentDocumentStrategy(pipe, None)

        hits = asyncio.run(strat.retrieve(RetrievalRequest(query="q")))

        assert hits[0][0].page_content == "chunk A\nchunk B"
        assert hits[0][1] == 0.9  # child's score is preserved

    def test_legacy_chunk_without_parent_id_unchanged(self, monkeypatch):
        _patch(monkeypatch, "parent_retriever")
        child = _FakeDoc("solo")  # no parent_id metadata
        pipe = _FakePipeline(default=[(child, 0.9)], vector_store=_FakeStore({}))
        strat = ParentDocumentStrategy(pipe, None)

        hits = asyncio.run(strat.retrieve(RetrievalRequest(query="q")))

        assert hits[0][0].page_content == "solo"

    def test_store_without_lookup_returns_children(self, monkeypatch):
        _patch(monkeypatch, "parent_retriever")
        child = _FakeDoc("c", parent_id="p1", chunk_index=0)
        pipe = _FakePipeline(default=[(child, 0.9)], vector_store=object())
        strat = ParentDocumentStrategy(pipe, None)

        hits = asyncio.run(strat.retrieve(RetrievalRequest(query="q")))

        assert hits[0][0].page_content == "c"


# --- Corrective RAG --------------------------------------------------------


class TestCorrectiveRAG:
    def test_correct_grade_keeps_hits_without_reretrieval(self, monkeypatch):
        _patch(monkeypatch, "corrective_rag")
        pipe = _FakePipeline(default=[(_FakeDoc("good"), 0.9)])
        llm = _ScriptedLLM(rules=[("JSON", '{"relevance": "correct"}')])
        strat = CorrectiveRAGStrategy(pipe, llm)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert [d.page_content for d, _ in hits] == ["good"]
        assert len(pipe.queries) == 1

    def test_incorrect_grade_refines_and_reretrieves(self, monkeypatch):
        _patch(monkeypatch, "corrective_rag")
        weak, better = _FakeDoc("weak"), _FakeDoc("better")
        pipe = _FakePipeline(
            results_by_query={"refined": [(better, 0.85)]}, default=[(weak, 0.72)]
        )

        def handler(prompt, fmt):
            if fmt == "json":
                return '{"relevance": "incorrect"}'
            if "Rewrite" in prompt:
                return "refined query"
            return ""

        strat = CorrectiveRAGStrategy(pipe, _CallableLLM(handler))

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert "better" in {d.page_content for d, _ in hits}
        assert any("refined" in q for q in pipe.queries)

    def test_web_fallback_augments_when_enabled(self, monkeypatch):
        _patch(monkeypatch, "corrective_rag", rag_crag_web_fallback=True)
        pipe = _FakePipeline(default=[(_FakeDoc("weak"), 0.72)])

        def handler(prompt, fmt):
            if fmt == "json":
                return '{"relevance": "incorrect"}'
            return "refined query"

        web = _FakeWebTool(
            results=[{"title": "Official", "snippet": "the answer", "url": "http://x"}]
        )
        strat = CorrectiveRAGStrategy(pipe, _CallableLLM(handler), web_tool=web)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert web.called
        assert any("Official" in d.page_content for d, _ in hits)

    def test_no_llm_heuristic_grade_reretrieves_when_sparse(self, monkeypatch):
        _patch(monkeypatch, "corrective_rag", rag_top_k=4)
        # One hit (< top_k) → heuristic grade "ambiguous" → a corrective pass runs.
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.9)])
        strat = CorrectiveRAGStrategy(pipe, None)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert [d.page_content for d, _ in hits] == ["x"]
        assert len(pipe.queries) == 2  # initial + one refine pass


# --- Self-RAG --------------------------------------------------------------


class TestSelfRAG:
    def test_reflection_filters_irrelevant_passages(self, monkeypatch):
        _patch(monkeypatch, "self_rag")  # rag_self_rag_enabled False → always retrieve
        good, bad = _FakeDoc("good"), _FakeDoc("bad")
        pipe = _FakePipeline(default=[(good, 0.9), (bad, 0.8)])

        def handler(prompt, fmt):
            if "good" in prompt:
                return '{"relevant": true}'
            if "bad" in prompt:
                return '{"relevant": false}'
            return "{}"

        strat = SelfRAGStrategy(pipe, _CallableLLM(handler))

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert [d.page_content for d, _ in hits] == ["good"]

    def test_skips_retrieval_when_model_declines(self, monkeypatch):
        _patch(monkeypatch, "self_rag", rag_self_rag_enabled=True)
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.9)])

        def handler(prompt, fmt):
            if "Would retrieving" in prompt:
                return '{"retrieve": false}'
            return "{}"

        strat = SelfRAGStrategy(pipe, _CallableLLM(handler))

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert hits == []
        assert pipe.queries == []

    def test_all_rejected_falls_back_to_raw_hits(self, monkeypatch):
        _patch(monkeypatch, "self_rag")
        doc = _FakeDoc("x")
        pipe = _FakePipeline(default=[(doc, 0.9)])
        strat = SelfRAGStrategy(pipe, _CallableLLM(lambda p, f: '{"relevant": false}'))

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert [d.page_content for d, _ in hits] == ["x"]

    def test_no_llm_returns_all(self, monkeypatch):
        _patch(monkeypatch, "self_rag")
        pipe = _FakePipeline(default=[(_FakeDoc("x"), 0.9)])
        strat = SelfRAGStrategy(pipe, None)

        hits = asyncio.run(
            strat.retrieve(RetrievalRequest(query="q", target_language="java"))
        )

        assert [d.page_content for d, _ in hits] == ["x"]


# --- Dispatch + tool routing ----------------------------------------------


class TestStrategyDispatch:
    def test_build_strategy_resolves_names(self):
        pipe = RAGPipeline(None, None, llm_client=object())
        assert isinstance(pipe._build_strategy("hyde"), HyDEStrategy)
        assert isinstance(pipe._build_strategy("multi_query"), MultiQueryStrategy)
        assert isinstance(pipe._build_strategy("multi_hop"), MultiHopStrategy)
        assert isinstance(
            pipe._build_strategy("contextual_compression"),
            ContextualCompressionStrategy,
        )
        assert isinstance(
            pipe._build_strategy("parent_document"), ParentDocumentStrategy
        )
        assert isinstance(pipe._build_strategy("corrective"), CorrectiveRAGStrategy)
        assert isinstance(pipe._build_strategy("self_rag"), SelfRAGStrategy)

    def test_unknown_strategy_falls_back_to_single_hop(self):
        pipe = RAGPipeline(None, None)
        assert pipe._build_strategy("nonsense").name == "single_hop"
        assert pipe._build_strategy("single_hop").name == "single_hop"


class _RecordingRAG:
    def __init__(self):
        self.retrieve_calls = []
        self.search_calls = []

    async def retrieve(self, request, strategy=None):
        self.retrieve_calls.append((request.query, strategy))
        return [(_FakeDoc("hit"), 0.9)]

    async def search(self, query="", target_language="", target_version=""):
        self.search_calls.append(query)
        return [(_FakeDoc("searchhit"), 0.9)]


class TestVectorDBIntentRouting:
    def test_precise_intent_uses_hyde(self):
        rag = _RecordingRAG()
        result = asyncio.run(
            VectorDBTool(rag)(query="q", target_language="java", intent="precise")
        )
        assert rag.retrieve_calls == [("q", "hyde")]
        assert rag.search_calls == []
        assert result.success

    def test_exploratory_intent_uses_multi_query(self):
        rag = _RecordingRAG()
        asyncio.run(VectorDBTool(rag)(query="q", intent="exploratory"))
        assert rag.retrieve_calls == [("q", "multi_query")]

    def test_no_intent_uses_plain_search(self):
        rag = _RecordingRAG()
        asyncio.run(VectorDBTool(rag)(query="q", target_language="java"))
        assert rag.search_calls == ["q"]
        assert rag.retrieve_calls == []
