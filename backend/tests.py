"""
Unit tests for CodeMigrateAI backend.
Run: pytest tests.py -v
"""

import json
import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from agents.analyzer import AnalyzerAgent
from agents.migrator import MigratorAgent
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


def test_registry_order_is_two_agents():
    registry = AgentRegistry(MockLLM())
    assert [agent.name for agent in registry.get_order()] == [
        "AnalyzerAgent",
        "MigratorAgent",
    ]
