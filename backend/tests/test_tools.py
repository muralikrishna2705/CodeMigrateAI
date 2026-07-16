"""Tests for the agent tool layer: the contract, the registry, and tool use."""

import asyncio
import json

import pytest

from agents.analyzer_agent import AnalyzerAgent
from agents.base import BaseAgent
from agents.retriever_agent import RetrieverAgent
from agents.tools import build_registry
from agents.tools.base import AgentTool, ToolRegistry, ToolResult
from agents.tools.code_metrics import CodeMetricsTool
from agents.tools.source_reader import SourceReaderTool
from agents.tools.syntax_checker import SyntaxCheckTool
from agents.tools.vector_db import VectorDBTool
from agents.tools.web_search import WebSearchTool, official_domains
from config import Settings
from models.state import MigrationState


def _state(**overrides) -> MigrationState:
    base = dict(
        source_code="x = 1",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
    )
    base.update(overrides)
    return MigrationState(**base)


class _EchoTool(AgentTool):
    name = "echo"
    description = "Echo the value back"
    parameters = {"value": "anything"}

    async def run(self, value: str = "", **_) -> ToolResult:
        return ToolResult(tool=self.name, success=True, data=value, summary="echoed")


class _BoomTool(AgentTool):
    name = "boom"
    description = "Always raises"

    async def run(self, **_) -> ToolResult:
        raise RuntimeError("kaboom")


class _SlowTool(AgentTool):
    name = "slow"
    description = "Never returns in time"
    timeout_sec = 0.05

    async def run(self, **_) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(tool=self.name, success=True)


class TestToolContract:
    @pytest.mark.asyncio
    async def test_successful_call_is_timed(self):
        result = await _EchoTool()(value="hi")
        assert result.success
        assert result.data == "hi"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_exception_becomes_failed_result_not_a_raise(self):
        # The whole point of the contract: a broken tool degrades the caller's
        # result, it never propagates out and fails the migration.
        result = await _BoomTool()()
        assert not result.success
        assert "kaboom" in result.error

    @pytest.mark.asyncio
    async def test_timeout_becomes_failed_result(self):
        result = await _SlowTool()()
        assert not result.success
        assert "timed out" in result.error


class TestToolRegistry:
    def test_lookup_and_dict_protocol(self):
        registry = ToolRegistry([_EchoTool()])
        assert "echo" in registry
        assert registry.get("echo").name == "echo"
        assert registry.get("nope") is None
        assert registry["echo"].name == "echo"
        assert len(registry) == 1
        assert registry.names() == ["echo"]

    def test_empty_registry_is_falsy(self):
        assert not ToolRegistry()
        assert ToolRegistry([_EchoTool()])

    def test_subset_keeps_only_present_names(self):
        registry = ToolRegistry([_EchoTool(), _BoomTool()])
        subset = registry.subset(["echo", "never_registered"])
        assert subset.names() == ["echo"]

    def test_describe_renders_catalog_for_the_prompt(self):
        described = ToolRegistry([_EchoTool()]).describe()
        assert "echo(value: anything)" in described
        assert "Echo the value back" in described


class TestBuildRegistry:
    def test_tools_disabled_yields_empty_registry(self):
        registry = build_registry(settings=Settings(tools_enabled=False))
        assert not registry

    def test_dependency_free_tools_build_without_rag(self):
        registry = build_registry(settings=Settings())
        assert "code_metrics" in registry
        assert "syntax_check" in registry
        assert "source_reader" in registry

    def test_vector_db_needs_a_pipeline(self):
        assert "vector_db" not in build_registry(settings=Settings())
        assert "vector_db" in build_registry(
            rag_pipeline=object(), settings=Settings()
        )

    def test_per_tool_toggle_excludes_a_tool(self):
        registry = build_registry(settings=Settings(tool_source_reader_enabled=False))
        assert "source_reader" not in registry
        assert "code_metrics" in registry


class TestCodeMetricsTool:
    @pytest.mark.asyncio
    async def test_counts_match_the_analyzers_original_rule(self):
        code = "class Foo:\n    def bar(self):\n        if x:\n            pass\n"
        result = await CodeMetricsTool()(code=code)
        assert result.success
        assert result.data["class_count"] == 1
        assert result.data["method_count"] == 1
        assert result.data["complexity"] == "low"

    @pytest.mark.asyncio
    async def test_high_branch_count_reports_high_complexity(self):
        # Drives DispatcherAgent's deep-analysis routing, so the threshold matters.
        result = await CodeMetricsTool()(code="if x:\n    pass\n" * 25)
        assert result.data["complexity"] == "high"


class TestSyntaxCheckTool:
    @pytest.mark.asyncio
    async def test_valid_python_passes(self):
        result = await SyntaxCheckTool()(code="x = 1\n", language="python")
        assert result.success
        assert result.data["valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_python_fails_with_diagnostics(self):
        result = await SyntaxCheckTool()(code="def broken(:\n", language="python")
        assert result.success  # the tool ran fine; the *code* is invalid
        assert result.data["valid"] is False
        assert result.data["errors"]

    @pytest.mark.asyncio
    async def test_empty_code_is_vacuously_valid(self):
        result = await SyntaxCheckTool()(code="   ", language="python")
        assert result.data["valid"] is True


class TestSourceReaderTool:
    SOURCE = "\n".join(
        ["import os", "", "class Alpha:", "    def one(self):", "        return 1"]
        + [f"# filler {i}" for i in range(20)]
        + ["def omega():", "    return 'end'"]
    )

    @pytest.mark.asyncio
    async def test_outline_finds_symbols_past_the_truncation_point(self):
        result = await SourceReaderTool()(source_code=self.SOURCE, mode="outline")
        names = [s["name"] for s in result.data["symbols"]]
        assert "Alpha" in names
        assert "omega" in names  # the tail, which a truncated prompt would lose

    @pytest.mark.asyncio
    async def test_read_returns_the_requested_range(self):
        result = await SourceReaderTool()(
            source_code=self.SOURCE, mode="read", start_line=3, end_line=4
        )
        assert result.data["snippet"] == "class Alpha:\n    def one(self):"

    @pytest.mark.asyncio
    async def test_read_clamps_past_the_end_instead_of_erroring(self):
        result = await SourceReaderTool()(
            source_code=self.SOURCE, mode="read", start_line=1, end_line=9999
        )
        assert result.success
        assert result.data["end_line"] == len(self.SOURCE.splitlines())

    @pytest.mark.asyncio
    async def test_start_past_end_is_an_error(self):
        result = await SourceReaderTool()(
            source_code=self.SOURCE, mode="read", start_line=9999
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_find_returns_match_with_context(self):
        result = await SourceReaderTool()(
            source_code=self.SOURCE, mode="find", pattern="omega", context_lines=1
        )
        assert result.data["matches"]
        assert "omega" in result.data["matches"][0]["snippet"]

    @pytest.mark.asyncio
    async def test_unknown_mode_is_an_error(self):
        result = await SourceReaderTool()(source_code=self.SOURCE, mode="teleport")
        assert not result.success
        assert "unknown mode" in result.error


class _FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


class _FakeRAG:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.queries: list[str] = []

    async def search(self, query, target_language="", target_version="", symbols=None):
        self.queries.append(query)
        return self.hits


class TestVectorDBTool:
    @pytest.mark.asyncio
    async def test_passes_the_agents_query_through_verbatim(self):
        # The point of the tool: the *agent* formulates the query, unlike
        # enrich_prompt which derives one from code signals.
        rag = _FakeRAG(hits=[(_FakeDoc("code", {"language": "java"}), 0.9)])
        result = await VectorDBTool(rag)(query="virtual threads", target_language="java")
        assert rag.queries == ["virtual threads"]
        assert result.data["hits"][0]["language"] == "java"

    @pytest.mark.asyncio
    async def test_context_carries_the_heading_migrator_keys_on(self):
        rag = _FakeRAG(hits=[(_FakeDoc("code", {"language": "java"}), 0.9)])
        result = await VectorDBTool(rag)(query="q")
        assert "Reference Examples" in result.data["context"]

    @pytest.mark.asyncio
    async def test_no_hits_is_success_with_empty_data(self):
        result = await VectorDBTool(_FakeRAG())(query="q")
        assert result.success
        assert result.data["hits"] == []

    @pytest.mark.asyncio
    async def test_missing_pipeline_fails_cleanly(self):
        result = await VectorDBTool(None)(query="q")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_query_is_rejected(self):
        result = await VectorDBTool(_FakeRAG())(query="  ")
        assert not result.success


class TestWebSearchTool:
    def test_official_domains_come_from_the_url_index(self):
        domains = official_domains("python")
        assert "docs.python.org" in domains
        assert official_domains("klingon") == []

    def test_query_is_restricted_to_official_domains(self):
        built = WebSearchTool()._build_query("asyncio", "python")
        assert "site:docs.python.org" in built
        assert "asyncio" in built

    def test_unknown_language_falls_back_to_a_plain_query(self):
        assert WebSearchTool()._build_query("foo", "klingon") == "klingon foo"

    def test_ddg_redirect_urls_are_unwrapped(self):
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&rut=x"
        assert WebSearchTool._unwrap_url(wrapped) == "https://docs.python.org/3/"

    def test_relative_and_empty_hrefs_are_dropped(self):
        assert WebSearchTool._unwrap_url("") == ""
        assert WebSearchTool._unwrap_url("/settings") == ""

    def test_results_parse_out_of_ddg_html(self):
        html = """
        <div class="result">
          <a class="result__a" href="https://docs.python.org/3/library/asyncio.html">asyncio</a>
          <div class="result__snippet">async IO library</div>
        </div>
        """
        results = WebSearchTool()._parse_results(html)
        assert results == [
            {
                "title": "asyncio",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "snippet": "async IO library",
            }
        ]


class TestBaseAgentToolCalls:
    class _ToolUser(BaseAgent):
        name = "_ToolUser"
        requires_llm = False

        async def run(self, state):  # pragma: no cover — not exercised
            raise NotImplementedError

    def test_agent_without_tools_gets_an_empty_registry_not_none(self):
        agent = self._ToolUser(None)
        assert isinstance(agent.tools, ToolRegistry)
        assert not agent.tools

    @pytest.mark.asyncio
    async def test_missing_tool_returns_failure_rather_than_raising(self):
        agent = self._ToolUser(None)
        result = await agent._call_tool("nope")
        assert not result.success
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_calls_are_logged_without_argument_values(self):
        agent = self._ToolUser(None, {"tools": ToolRegistry([_EchoTool()])})
        await agent._call_tool("echo", value="a-very-long-source-file")
        log = agent.tool_call_log()
        assert log[0]["tool"] == "echo"
        assert log[0]["success"] is True
        # Keys only: values would duplicate whole source files into every report.
        assert log[0]["arguments"] == ["value"]
        assert "a-very-long-source-file" not in json.dumps(log)


class _SelectingLLM:
    """LLM stub that returns a scripted tool-selection JSON."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "{}"

    def extract_json(self, raw_text):
        return json.loads(raw_text)


class TestToolSelection:
    class _Selector(BaseAgent):
        name = "_Selector"
        requires_llm = False

        async def run(self, state):  # pragma: no cover
            raise NotImplementedError

    @pytest.mark.asyncio
    async def test_valid_selection_is_parsed(self):
        llm = _SelectingLLM(json.dumps({"tool": "echo", "arguments": {"value": "x"}}))
        agent = self._Selector(llm, {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") == ("echo", {"value": "x"})

    @pytest.mark.asyncio
    async def test_hallucinated_tool_name_is_rejected(self):
        # The classic small-model failure: inventing a plausible tool.
        llm = _SelectingLLM(json.dumps({"tool": "grep_the_internet"}))
        agent = self._Selector(llm, {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") is None

    @pytest.mark.asyncio
    async def test_null_choice_returns_none(self):
        llm = _SelectingLLM(json.dumps({"tool": None}))
        agent = self._Selector(llm, {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") is None

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_none(self):
        llm = _SelectingLLM("I would suggest using the echo tool!")
        agent = self._Selector(llm, {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") is None

    @pytest.mark.asyncio
    async def test_non_dict_arguments_are_coerced_to_empty(self):
        llm = _SelectingLLM(json.dumps({"tool": "echo", "arguments": "value=x"}))
        agent = self._Selector(llm, {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") == ("echo", {})

    @pytest.mark.asyncio
    async def test_no_tools_short_circuits_without_an_llm_call(self):
        llm = _SelectingLLM(json.dumps({"tool": "echo"}))
        agent = self._Selector(llm, {"tools": ToolRegistry()})
        assert await agent._select_tool("goal") is None
        assert llm.prompts == []

    @pytest.mark.asyncio
    async def test_llm_without_call_llm_returns_none(self):
        agent = self._Selector(object(), {"tools": ToolRegistry([_EchoTool()])})
        assert await agent._select_tool("goal") is None


class TestAnalyzerToolUse:
    @pytest.mark.asyncio
    async def test_metrics_come_from_the_tool_when_registered(self):
        agent = AnalyzerAgent(
            None,
            {
                "tools": ToolRegistry([CodeMetricsTool()]),
                "enable_semantic_analysis": False,
            },
        )
        state = _state(source_code="class Foo:\n    def bar(self):\n        pass\n")
        result = await agent.run(state)
        assert result.details["class_count"] == 1
        assert [c["tool"] for c in result.details["tool_calls"]] == ["code_metrics"]

    @pytest.mark.asyncio
    async def test_metrics_still_produced_when_tools_are_disabled(self):
        # The wrap-don't-replace guarantee: code_metrics feeds Dispatcher routing,
        # the RAG query, and PromptComposer, so it must survive tools_enabled=False.
        agent = AnalyzerAgent(None, {"enable_semantic_analysis": False})
        result = await agent.run(_state(source_code="class Foo:\n    pass\n"))
        assert result.details["class_count"] == 1
        assert result.details["complexity"] == "low"

    @pytest.mark.asyncio
    async def test_semantic_analysis_defaults_to_the_setting_not_a_missing_config(self):
        # Regression: the graph builds agents from `needs`, which never carried
        # enable_semantic_analysis, so config.get(...) was always None and the
        # semantic pass was dead on the graph path regardless of the setting.
        agent = AnalyzerAgent(None)
        assert agent._semantic_enabled() is True

    @pytest.mark.asyncio
    async def test_explicit_config_overrides_the_setting(self):
        agent = AnalyzerAgent(None, {"enable_semantic_analysis": False})
        assert agent._semantic_enabled() is False

    @pytest.mark.asyncio
    async def test_partial_semantic_json_keeps_the_full_metric_shape(self):
        # RAGPipeline._metric_terms and PromptComposer read these keys directly.
        llm = _SelectingLLM(json.dumps({"key_constructs": ["classes"]}))
        agent = AnalyzerAgent(llm, {"enable_semantic_analysis": True})
        result = await agent.run(_state())
        assert result.details["key_constructs"] == ["classes"]
        assert result.details["deprecated_patterns"] == []
        assert result.details["migration_challenges"] == []


class _HeuristicRAG(_FakeRAG):
    """RAG stub whose single-pass enrich_prompt returns a marked context."""

    async def enrich_prompt(self, **kwargs):
        return "## Reference Examples\nheuristic-context\n\n---\n\n"


class TestRetrieverToolLoop:
    @pytest.mark.asyncio
    async def test_loop_builds_context_from_the_chosen_tool(self, settings_override):
        settings_override(retriever_tool_loop=True)

        rag = _FakeRAG(hits=[(_FakeDoc("System.out", {"language": "java"}), 0.91)])
        llm = _SelectingLLM(
            json.dumps({"tool": "vector_db", "arguments": {"query": "println"}})
        )
        agent = RetrieverAgent(
            llm,
            {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])},
        )
        state = _state(target_language="java", target_version="21")
        result = await agent.run(state)

        assert result.details["mode"] == "tool-loop"
        assert "Reference Examples" in state.rag_context
        assert "System.out" in state.rag_context
        # The agent's own query reached the pipeline, not a code-signal query.
        assert rag.queries == ["println"]

    @pytest.mark.asyncio
    async def test_context_ends_with_the_separator_migrator_expects(
        self, settings_override
    ):
        # MigratorAgent concatenates rag_context straight onto its prompt, so a
        # missing separator would run the reference block into the instructions.
        settings_override(retriever_tool_loop=True)
        rag = _FakeRAG(hits=[(_FakeDoc("code", {"language": "java"}), 0.9)])
        llm = _SelectingLLM(
            json.dumps({"tool": "vector_db", "arguments": {"query": "q"}})
        )
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        await agent.run(state)
        assert state.rag_context.endswith("\n\n---\n\n")

    @pytest.mark.asyncio
    async def test_loop_falls_back_to_heuristic_when_selection_fails(
        self, settings_override
    ):
        settings_override(retriever_tool_loop=True)

        rag = _HeuristicRAG()
        llm = _SelectingLLM("not json at all")
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        result = await agent.run(state)

        # A failed selection must land on the pre-tool behaviour, not on nothing.
        assert result.details["mode"] == "heuristic"
        assert "heuristic-context" in state.rag_context

    @pytest.mark.asyncio
    async def test_loop_falls_back_when_the_tool_finds_nothing(self, settings_override):
        settings_override(retriever_tool_loop=True, retriever_max_tool_calls=1)

        rag = _HeuristicRAG()  # search() returns no hits
        llm = _SelectingLLM(
            json.dumps({"tool": "vector_db", "arguments": {"query": "q"}})
        )
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        result = await agent.run(state)

        assert result.details["mode"] == "heuristic"
        assert "heuristic-context" in state.rag_context

    @pytest.mark.asyncio
    async def test_loop_stops_when_the_model_repeats_a_query(self, settings_override):
        # Without dedupe the loop would spend its whole budget re-asking the same
        # question, since the goal text barely changes between attempts.
        settings_override(retriever_tool_loop=True, retriever_max_tool_calls=3)

        rag = _HeuristicRAG()
        choice = json.dumps({"tool": "vector_db", "arguments": {"query": "same"}})
        llm = _SelectingLLM(choice, choice, choice)
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        await agent.run(_state())

        # Two selections (the second is the repeat that breaks the loop), one call.
        assert len(llm.prompts) == 2
        assert len(agent.tool_call_log()) == 1

    @pytest.mark.asyncio
    async def test_loop_without_retrieval_tools_uses_heuristic(self, settings_override):
        settings_override(retriever_tool_loop=True)
        llm = _SelectingLLM(json.dumps({"tool": "echo"}))
        agent = RetrieverAgent(
            llm,
            {"rag_pipeline": _HeuristicRAG(), "tools": ToolRegistry([_EchoTool()])},
        )
        result = await agent.run(_state())
        assert result.details["mode"] == "heuristic"
        assert llm.prompts == []  # no retrieval tools -> no selection call

    @pytest.mark.asyncio
    async def test_loop_is_off_by_default(self):
        llm = _SelectingLLM(json.dumps({"tool": "vector_db"}))
        agent = RetrieverAgent(llm, {"rag_pipeline": _HeuristicRAG()})
        result = await agent.run(_state())
        assert result.details["mode"] == "heuristic"
        assert llm.prompts == []  # no selection call was made

    def test_arguments_are_filled_from_state_not_left_to_the_model(self):
        state = _state(target_language="java", target_version="21")
        args = RetrieverAgent._normalize_arguments(
            "vector_db", {"query": "streams"}, state
        )
        assert args == {
            "query": "streams",
            "target_language": "java",
            "target_version": "21",
        }

    def test_web_search_arguments_use_the_language_key(self):
        args = RetrieverAgent._normalize_arguments("web_search", {"query": "q"}, _state())
        assert args["language"] == "python"
        assert args["fetch_content"] is False

    def test_blocks_render_vector_hits_as_fenced_code(self):
        result = ToolResult(
            tool="vector_db",
            success=True,
            data={"hits": [{"content": "code", "score": 0.9, "language": "java"}]},
        )
        blocks = RetrieverAgent._blocks_from("vector_db", result)
        assert "```java" in blocks[0]
        assert "relevance: 0.90" in blocks[0]
