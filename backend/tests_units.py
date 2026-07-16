"""
Unit tests for CodeMigrateAI backend.
Run: pytest tests_units.py -v

Named tests_units.py (not tests.py) so it does not collide with the tests/
package: a lone module named ``tests`` beside a ``tests/`` directory can only be
imported as one or the other, which silently drops this file from a full-tree
collection.
"""

import json
from unittest.mock import AsyncMock

import pytest

from agents.analyzer_agent import AnalyzerAgent
from agents.migrator_agent import MigratorAgent
from config import Settings
from llm.client import LLMClient
from llm.language_profiles import get_profile, get_supported_profiles
from llm.prompt_composer import PromptComposer
from models.state import MigrationState, MigrationType
from pipeline.registry import AgentRegistry


class MockLLM:
    def __init__(self, responses=None, side_effect=None):
        if side_effect is not None:
            self.call_llm = AsyncMock(side_effect=side_effect)
        else:
            values = responses if isinstance(responses, list) else [responses or "{}"]
            self.call_llm = AsyncMock(side_effect=values)

    def extract_json(self, raw_text: str) -> dict:
        for block in raw_text.split("```"):
            text = block.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
        return json.loads(raw_text)


class StreamingMockLLM:
    """LLM stub that streams a canned response token-by-token.

    Used to verify the migrator forwards *unwrapped* code (not raw JSON) to the
    SSE callback. ``chunk`` deliberately splits the payload into tiny pieces so
    the incremental unwrap is exercised across escape/quote boundaries.
    """

    def __init__(self, response: str, chunk: int = 3):
        self._response = response
        self._chunk = chunk
        self.call_llm = AsyncMock(return_value=response)

    async def stream_llm(self, prompt, system_prompt, fmt=None):
        for i in range(0, len(self._response), self._chunk):
            yield self._response[i : i + self._chunk]


def _settings_with(**overrides) -> Settings:
    # Force fast_llm_model empty by default so the fallback test is not affected
    # by any FAST_LLM_MODEL in the environment/.env.
    base = {"fast_llm_model": ""}
    base.update(overrides)
    return Settings(**base)


def make_state(**overrides) -> MigrationState:
    data = {
        "source_code": "public class Foo {}",
        "source_language": "java",
        "source_version": "7",
        "target_language": "java",
        "target_version": "17",
    }
    data.update(overrides)
    return MigrationState(**data)


def test_extract_json_from_fenced_block():
    client = LLMClient()
    text = '```json\n{"key": "value"}\n```'
    assert client.extract_json(text) == {"key": "value"}


def test_extract_json_from_raw_object():
    client = LLMClient()
    text = 'Some preamble {"complexity": "high"} trailing'
    assert client.extract_json(text) == {"complexity": "high"}


def test_extract_json_raises_on_no_json():
    client = LLMClient()
    with pytest.raises(ValueError):
        client.extract_json("no json here at all")


def test_strip_fences_removes_code_block():
    agent = MigratorAgent(None)
    raw = "```python\ndef hello(): pass\n```"
    assert agent._strip_fences(raw) == "def hello(): pass"


def test_strip_fences_passthrough_clean_code():
    agent = MigratorAgent(None)
    code = "public class Foo {}"
    assert agent._strip_fences(code) == "public class Foo {}"


def test_strip_fences_removes_preamble():
    agent = MigratorAgent(None)
    raw = "Here is the migrated code:\ndef hello(): pass"
    result = agent._strip_fences(raw)
    assert "Here is" not in result


def test_state_record_success():
    state = make_state()
    state.record_success("TestAgent", "done", {"lines": 10})
    assert "TestAgent" in state.agents_done
    assert state.reports[0].status == "success"


def test_state_record_error():
    state = make_state(target_language="python", target_version="3.12")
    state.record_error("TestAgent", "something broke")
    assert state.errors == ["[TestAgent] something broke"]
    assert state.reports[0].status == "error"


@pytest.mark.asyncio
async def test_analyzer_agent_static_success():
    state = make_state(source_code="public class Foo {\n  public void bar() {}\n}")
    agent = AnalyzerAgent(MockLLM())
    result = await agent(state)

    assert result.code_metrics is not None
    assert result.code_metrics["total_lines"] == 3
    assert result.code_metrics["deprecated_patterns"] == []
    assert "AnalyzerAgent" in result.agents_done
    assert result.reports[0].status == "success"


@pytest.mark.asyncio
async def test_analyzer_agent_optional_llm_failure_graceful():
    state = make_state(
        source_code="x = 1\ny = 2\n",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
    )
    agent = AnalyzerAgent(
        MockLLM(side_effect=Exception("LLM unavailable")),
        {"enable_semantic_analysis": True},
    )
    result = await agent(state)

    assert result.code_metrics is not None
    assert result.code_metrics["total_lines"] == 2
    assert "AnalyzerAgent" in result.agents_done


def test_language_profiles_load_all_configured_languages():
    profiles = get_supported_profiles()
    expected = {
        "java",
        "python",
        "javascript",
        "typescript",
        "csharp",
        "go",
        "kotlin",
        "rust",
        "cpp",
    }
    assert expected.issubset(profiles.keys())
    assert get_profile("js").language_id == "javascript"
    assert get_profile("c++").language_id == "cpp"


def test_prompt_composer_composes_and_caches():
    composer = PromptComposer()
    java = get_profile("java")
    python = get_profile("python")

    prompt1 = composer.compose(
        source_profile=java,
        target_profile=python,
        source_version="8",
        target_version="3.12",
        source_code="System.out.println(name);",
        analyzer_context={"total_lines": 1, "complexity": "low"},
        migration_type="convert_language",
    )
    prompt2 = composer.compose(
        source_profile=java,
        target_profile=python,
        source_version="8",
        target_version="3.12",
        source_code="System.out.println(name);",
        analyzer_context={"total_lines": 1, "complexity": "low"},
        migration_type="convert_language",
    )

    assert "Java to Python migration engineer" in prompt1
    assert '"migrated_code"' in prompt1
    assert "System.out.println -> print" in prompt1
    assert prompt1 == prompt2
    assert composer.cache_size() == 1


@pytest.mark.asyncio
async def test_migrator_single_call_sets_plan_and_code():
    response = json.dumps(
        {
            "plan_summary": "Upgrade Java 7 to Java 17.",
            "migrated_code": "public class Foo {}",
        }
    )
    llm = MockLLM(response)
    agent = MigratorAgent(llm)
    state = make_state(code_metrics={"total_lines": 1, "complexity": "low"})

    result = await agent(state)

    assert result.migration_type == MigrationType.UPGRADE_VERSION
    assert result.inline_plan == "Upgrade Java 7 to Java 17."
    assert result.migrated_code == "public class Foo {}"
    llm.call_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_migrator_detects_language_conversion():
    response = json.dumps(
        {
            "plan_summary": "Convert Java collections to Python containers.",
            "migrated_code": "class Foo:\n    pass",
        }
    )
    agent = MigratorAgent(MockLLM(response))
    state = make_state(target_language="python", target_version="3.12")

    result = await agent(state)

    assert result.migration_type == MigrationType.CONVERT_LANGUAGE
    assert result.inline_plan.startswith("Convert Java")
    assert "class Foo" in result.migrated_code


@pytest.mark.asyncio
async def test_migrator_retries_invalid_json():
    retry_response = json.dumps(
        {
            "plan_summary": "Retry produced valid JSON.",
            "migrated_code": "print('ok')",
        }
    )
    llm = MockLLM(responses=["not json", retry_response])
    agent = MigratorAgent(llm)
    state = make_state(
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
    )

    result = await agent(state)

    assert result.inline_plan == "Retry produced valid JSON."
    assert result.migrated_code == "print('ok')"
    assert llm.call_llm.await_count == 2


def test_registry_discovers_domain_and_runtime_agents():
    registry = AgentRegistry(MockLLM())
    names = set(registry._agents)
    # Every agent the graph runs is discovered, including the dynamic router
    # (DispatcherAgent), the terminal observer, and the external-service validator.
    for expected in (
        "AnalyzerAgent",
        "DeepAnalyzerAgent",
        "RetrieverAgent",
        "PlannerAgent",
        "MigratorAgent",
        "ValidatorAgent",
        "FixerAgent",
        "DispatcherAgent",
        "ObserverAgent",
        "RuntimeValidatorAgent",
    ):
        assert expected in names
    # Provider (DI container) and Runtime (executor) are infrastructure classes,
    # not registered agents; the old RecoveryAgent became module-level functions.
    for gone in ("ProviderAgent", "RuntimeAgent", "RecoveryAgent"):
        assert gone not in names


@pytest.mark.asyncio
async def test_fixer_prompt_includes_offending_lines_and_rag_context():
    from agents.fixer_agent import FixerAgent

    llm = MockLLM("fixed_code_here")
    agent = FixerAgent(llm)
    state = make_state(target_language="python", target_version="3.12")
    state.migrated_code = "line1\nline2 BAD\nline3\n"
    state.rag_context = "## Reference Examples\nprint('grounded')"
    state.code_metrics = {"summary": "3 lines of python"}
    state.validation_result = {
        "valid": False,
        "errors": [{"line": 2, "column": 1, "message": "bad syntax"}],
        "warnings": [],
    }

    await agent(state)

    prompt = llm.call_llm.await_args.args[0]
    assert "OFFENDING LINES:" in prompt
    assert "line2 BAD" in prompt  # the exact offending source line, not just "line 2"
    assert "Reference Examples" in prompt  # retrieved context reused for grounding
    assert "3 lines of python" in prompt  # analyzer summary threaded in
    assert state.migrated_code == "fixed_code_here"


# --- Anti-hallucination: confidence gating + grounding report ---------------


def _migrator_report(state):
    return next(r for r in state.reports if r.agent == "MigratorAgent")


@pytest.mark.asyncio
async def test_migrator_injects_ungrounded_notice_without_rag():
    response = json.dumps({"plan_summary": "ok", "migrated_code": "print('x')"})
    llm = MockLLM(response)
    agent = MigratorAgent(llm)
    state = make_state(
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
        code_metrics={"total_lines": 1},
    )

    await agent(state)

    prompt = llm.call_llm.call_args.args[0]
    assert "GROUNDING NOTICE" in prompt


@pytest.mark.asyncio
async def test_migrator_uses_rag_context_and_skips_notice():
    response = json.dumps({"plan_summary": "ok", "migrated_code": "print('x')"})
    llm = MockLLM(response)
    agent = MigratorAgent(llm)
    state = make_state(
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
        code_metrics={"total_lines": 1},
    )
    state.rag_context = "## Reference Examples\nprint('grounded')\n\n---\n\n"

    await agent(state)

    prompt = llm.call_llm.call_args.args[0]
    assert "Reference Examples" in prompt
    assert "GROUNDING NOTICE" not in prompt


@pytest.mark.asyncio
async def test_migrator_flags_ungrounded_imports_in_report():
    response = json.dumps(
        {
            "plan_summary": "ok",
            "migrated_code": "import superfastjson\nprint('x')\n",
        }
    )
    agent = MigratorAgent(MockLLM(response))
    state = make_state(
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
    )

    await agent(state)

    grounding = _migrator_report(state).details["grounding"]
    assert "superfastjson" in grounding["unverified_imports"]


# --- Model routing ----------------------------------------------------------


def test_fast_model_falls_back_to_main_when_unset():
    client = LLMClient(_settings_with())
    assert client.fast_model == client.settings.llm_model


def test_fast_model_used_when_configured():
    client = LLMClient(_settings_with(fast_llm_model="fast:1b"))
    assert client.fast_model == "fast:1b"


def test_build_payload_honors_model_override():
    client = LLMClient(_settings_with())
    payload = client._build_payload("p", "s", stream=False, model="override:7b")
    assert payload["model"] == "override:7b"
    default_payload = client._build_payload("p", "s", stream=False)
    assert default_payload["model"] == client.settings.llm_model


def test_fast_model_kwargs_empty_for_stub_llm():
    # A stub LLM exposes no fast_model, so routing is a no-op (kwargs stay empty).
    assert MigratorAgent(MockLLM())._fast_model_kwargs() == {}


def test_fast_model_kwargs_present_for_real_client():
    agent = MigratorAgent(LLMClient(_settings_with(fast_llm_model="fast:1b")))
    assert agent._fast_model_kwargs() == {"model": "fast:1b"}


# --- Version grounding + streamed-code unwrap -------------------------------


def test_prompt_composer_grounds_target_version():
    composer = PromptComposer()
    python = get_profile("python")
    prompt = composer.compose(
        source_profile=python,
        target_profile=python,
        source_version="3.8",
        target_version="3.12",
        source_code="print('hi')",
        analyzer_context={"total_lines": 1},
        migration_type="upgrade_version",
    )
    assert "VERSION CONSTRAINTS — HARD REQUIREMENT" in prompt
    # The negative constraint (no features newer than the target) is the part
    # that actually prevents version hallucination.
    assert "introduced AFTER" in prompt
    assert "3.12" in prompt


@pytest.mark.asyncio
async def test_migrator_streams_unwrapped_code_not_json():
    code = "def greet(name):\n    print(f'hi {name}')\n"
    response = json.dumps({"plan_summary": "Converted.", "migrated_code": code})

    streamed: list[str] = []

    async def capture(token: str):
        streamed.append(token)

    agent = MigratorAgent(StreamingMockLLM(response), {"stream_callback": capture})
    state = make_state(
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
        source_code="print('hi')",
        code_metrics={"total_lines": 1},
    )

    result = await agent(state)

    streamed_text = "".join(streamed)
    # The client sees the decoded code, never the JSON envelope or plan text.
    # (The code itself contains f-string braces, so we check for the JSON
    # envelope specifically rather than any brace.)
    assert streamed_text == code
    assert "plan_summary" not in streamed_text
    assert '"migrated_code"' not in streamed_text
    assert not streamed_text.lstrip().startswith("{")
    # The authoritative parsed result still matches.
    assert result.migrated_code == code.strip()
