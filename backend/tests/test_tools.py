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
from langchain_core.messages import AIMessage
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

    def test_bindable_tools_expose_their_schema_to_the_model(self):
        # The field descriptions are what the model reads when choosing
        # arguments, so they are prompt surface and must survive the conversion.
        [tool] = ToolRegistry([CodeMetricsTool()]).as_langchain_tools()
        schema = tool.args_schema.model_json_schema()
        assert schema["required"] == ["code"]
        assert "source code to measure" in schema["properties"]["code"]["description"]


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


class _JsonLLM:
    """LLM stub returning scripted JSON, for the structured-output salvage path."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "{}"


class _BoundModel:
    """Stands in for a chat model with tools bound to it."""

    def __init__(self, owner):
        self.owner = owner

    def bind_tools(self, tools):
        self.owner.bound_tools = [t.name for t in tools]
        return self

    async def ainvoke(self, messages):
        self.owner.conversations.append(list(messages))
        turn = self.owner.turns.pop(0) if self.owner.turns else []
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "id": f"call-{i}",
                    "type": "tool_call",
                }
                for i, call in enumerate(turn)
            ],
        )


class _ToolCallingLLM:
    """LLM stub whose model emits scripted *native* tool calls.

    Each constructor argument is one turn: a list of ``{"name", "args"}`` dicts,
    or an empty list for a turn where the model asks for no tools.
    """

    def __init__(self, *turns):
        self.turns = [list(t) for t in turns]
        self.conversations: list[list] = []
        self.bound_tools: list[str] = []

    def chat_model(self, role="main", *, json_mode=False):
        return _BoundModel(self)


class TestToolBinding:
    """`BaseAgent.bind_tools` — the replacement for prompt-driven selection.

    A hallucinated tool name and malformed arguments used to be the two routine
    failure modes, each needing its own rejection path. Neither is reachable
    now: the model emits a schema-validated call or nothing.
    """

    class _Binder(BaseAgent):
        name = "_Binder"
        requires_llm = False

        async def run(self, state):  # pragma: no cover
            raise NotImplementedError

    def test_binds_the_registered_tools(self):
        llm = _ToolCallingLLM()
        agent = self._Binder(llm, {"tools": ToolRegistry([VectorDBTool(None)])})
        assert agent.bind_tools() is not None
        assert llm.bound_tools == ["vector_db"]

    def test_subset_limits_what_the_model_sees(self):
        # A short catalog is a requirement, not a limitation: selection accuracy
        # degrades as the option list grows.
        llm = _ToolCallingLLM()
        registry = ToolRegistry([VectorDBTool(None), CodeMetricsTool()])
        agent = self._Binder(llm, {"tools": registry})
        agent.bind_tools(("vector_db",))
        assert llm.bound_tools == ["vector_db"]

    def test_empty_registry_returns_none(self):
        agent = self._Binder(_ToolCallingLLM(), {"tools": ToolRegistry()})
        assert agent.bind_tools() is None

    def test_client_without_a_chat_model_returns_none(self):
        # Unit stubs. Callers keep their non-LLM fallback.
        agent = self._Binder(object(), {"tools": ToolRegistry([VectorDBTool(None)])})
        assert agent.bind_tools() is None

    def test_tools_without_an_args_schema_are_not_bindable(self):
        # _EchoTool declares no args_schema, so it is usable directly but not
        # exposed to the model — and that must not make the registry unbindable.
        assert ToolRegistry([_EchoTool()]).as_langchain_tools() == []


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
        llm = _JsonLLM(json.dumps({"key_constructs": ["classes"]}))
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
    async def test_loop_builds_context_from_the_chosen_tool(self):
        rag = _FakeRAG(hits=[(_FakeDoc("System.out", {"language": "java"}), 0.91)])
        llm = _ToolCallingLLM([{"name": "vector_db", "args": {"query": "println"}}])
        agent = RetrieverAgent(
            llm,
            {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])},
        )
        state = _state(target_language="java", target_version="21")
        result = await agent.run(state)

        assert result.details["mode"] == "tool-loop"
        assert "Reference Examples" in state.rag_context
        assert "System.out" in state.rag_context
        # The model's own query reached the pipeline, not a code-signal query.
        assert rag.queries == ["println"]

    @pytest.mark.asyncio
    async def test_tool_results_are_fed_back_as_tool_messages(self):
        # This is what makes it a loop rather than a single dispatch: the model
        # sees what a search returned and can search again knowing that.
        rag = _FakeRAG(hits=[(_FakeDoc("x", {"language": "java"}), 0.9)])
        llm = _ToolCallingLLM([{"name": "vector_db", "args": {"query": "q"}}])
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        await agent.run(_state())

        # One turn happened; the tool result was appended for a would-be next one.
        assert len(llm.conversations) == 1
        assert llm.bound_tools == ["vector_db"]

    @pytest.mark.asyncio
    async def test_a_failing_tool_is_reported_to_the_model_not_raised(self):
        # A failure the model can read is a failure it can route around.
        result = await _BoomTool()(x=1)
        assert result.success is False
        assert result.for_model().startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_context_ends_with_the_separator_migrator_expects(self):
        # MigratorAgent concatenates rag_context straight onto its prompt, so a
        # missing separator would run the reference block into the instructions.
        rag = _FakeRAG(hits=[(_FakeDoc("code", {"language": "java"}), 0.9)])
        llm = _ToolCallingLLM([{"name": "vector_db", "args": {"query": "q"}}])
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        await agent.run(state)
        assert state.rag_context.endswith("\n\n---\n\n")

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_when_the_model_asks_for_no_tools(self):
        rag = _HeuristicRAG()
        llm = _ToolCallingLLM([])  # a turn with no tool calls
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        result = await agent.run(state)

        assert result.details["mode"] == "heuristic"
        assert "heuristic-context" in state.rag_context

    @pytest.mark.asyncio
    async def test_loop_falls_back_when_the_tool_finds_nothing(self, settings_override):
        settings_override(retriever_max_tool_calls=1)

        rag = _HeuristicRAG()  # search() returns no hits
        llm = _ToolCallingLLM([{"name": "vector_db", "args": {"query": "q"}}])
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        state = _state()
        result = await agent.run(state)

        assert result.details["mode"] == "heuristic"
        assert "heuristic-context" in state.rag_context

    @pytest.mark.asyncio
    async def test_budget_bounds_the_loop(self, settings_override):
        # An empty-result turn used to need explicit query-repeat detection to
        # avoid burning the budget. The bound is now simply the budget.
        settings_override(retriever_max_tool_calls=2)
        rag = _HeuristicRAG()
        call = {"name": "vector_db", "args": {"query": "same"}}
        llm = _ToolCallingLLM([call], [call], [call])
        agent = RetrieverAgent(
            llm, {"rag_pipeline": rag, "tools": ToolRegistry([VectorDBTool(rag)])}
        )
        await agent.run(_state())

        assert len(llm.conversations) == 2
        assert len(agent.tool_call_log()) == 2

    @pytest.mark.asyncio
    async def test_loop_without_retrieval_tools_uses_heuristic(self):
        llm = _ToolCallingLLM([{"name": "echo", "args": {}}])
        agent = RetrieverAgent(
            llm,
            {"rag_pipeline": _HeuristicRAG(), "tools": ToolRegistry([_EchoTool()])},
        )
        result = await agent.run(_state())
        assert result.details["mode"] == "heuristic"
        assert llm.conversations == []  # nothing bindable -> no model call

    @pytest.mark.asyncio
    async def test_loop_can_be_switched_off(self, settings_override):
        settings_override(retriever_tool_loop=False)
        llm = _ToolCallingLLM([{"name": "vector_db", "args": {"query": "q"}}])
        agent = RetrieverAgent(llm, {"rag_pipeline": _HeuristicRAG()})
        result = await agent.run(_state())
        assert result.details["mode"] == "heuristic"
        assert llm.conversations == []

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

    def test_the_models_own_arguments_are_passed_through(self):
        # Only the facts the agent holds get overwritten. Whether to fetch a
        # page's full text is a judgement call, so it stays the model's.
        args = RetrieverAgent._normalize_arguments(
            "web_search", {"query": "q", "fetch_content": True}, _state()
        )
        assert args["fetch_content"] is True
        assert args["language"] == "python"  # still not the model's to guess

    def test_blocks_render_vector_hits_as_fenced_code(self):
        result = ToolResult(
            tool="vector_db",
            success=True,
            data={"hits": [{"content": "code", "score": 0.9, "language": "java"}]},
        )
        blocks = RetrieverAgent._blocks_from("vector_db", result)
        assert "```java" in blocks[0]
        assert "relevance: 0.90" in blocks[0]
