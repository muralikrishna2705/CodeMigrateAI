"""
Unit tests for CodeMigrateAI backend
Run: pytest tests.py -v
"""

import json
import os

# We import only the pure functions — no need to start the server
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from agents.analyzer import AnalyzerAgent
from agents.migrator import MigratorAgent
from agents.planner import MIGRATION_RECIPES, PlannerAgent
from llm.client import LLMClient
from models.state import MigrationState, MigrationType

# ── extract_json ──────────────────────────────────────────────────────────────

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


# ── _strip_fences ─────────────────────────────────────────────────────────────

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


# ── MigrationState ────────────────────────────────────────────────────────────

def test_state_record_success():
    state = MigrationState(
        source_code="x",
        source_language="java",
        source_version="7",
        target_language="java",
        target_version="17",
    )
    state.record_success("TestAgent", "done", {"lines": 10})
    assert "TestAgent" in state.agents_done
    assert state.reports[0].status == "success"

def test_state_record_error():
    state = MigrationState(
        source_code="x",
        source_language="java",
        source_version="7",
        target_language="python",
        target_version="3.12",
    )
    state.record_error("TestAgent", "something broke")
    assert state.errors == ["[TestAgent] something broke"]
    assert state.reports[0].status == "error"


# ── AnalyzerAgent (mocked LLM) ────────────────────────────────────────────────

def make_mock_llm(response=None, side_effect=None):
    """Create a mock LLM client for testing."""
    mock_llm = AsyncMock()
    if side_effect:
        mock_llm.call_llm.side_effect = side_effect
    else:
        mock_llm.call_llm.return_value = response
    mock_llm.extract_json = lambda text: json.loads(response) if response else {}
    return mock_llm


@pytest.mark.asyncio
async def test_analyzer_agent_success():
    state = MigrationState(
        source_code="public class Foo {\n  public void bar() {}\n}",
        source_language="java", source_version="7",
        target_language="java", target_version="17",
    )
    llm_response = json.dumps({
        "deprecated_patterns": ["raw types"],
        "migration_challenges": ["switch statements"],
        "key_constructs": ["class", "method"],
        "summary": "Simple Java class with one method.",
    })
    agent = AnalyzerAgent(make_mock_llm(llm_response))
    result = await agent(state)

    assert result.code_metrics is not None
    assert result.code_metrics["total_lines"] == 3
    assert "AnalyzerAgent" in result.agents_done
    assert result.reports[0].status == "success"


@pytest.mark.asyncio
async def test_analyzer_agent_llm_failure_graceful():
    """If LLM fails, AnalyzerAgent should still succeed using static metrics."""
    state = MigrationState(
        source_code="x = 1\ny = 2\n",
        source_language="python", source_version="2.7",
        target_language="python", target_version="3.12",
    )
    agent = AnalyzerAgent(make_mock_llm(side_effect=Exception("LLM unavailable")))
    result = await agent(state)

    # Should still succeed — graceful degradation
    assert result.code_metrics is not None
    assert result.code_metrics["total_lines"] == 2
    assert "AnalyzerAgent" in result.agents_done


# ── PlannerAgent ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_detects_upgrade():
    state = MigrationState(
        source_code="code",
        source_language="java",
        source_version="7",
        target_language="java",
        target_version="17",
        code_metrics={
            "deprecated_patterns": [],
            "migration_challenges": [],
            "complexity": "low",
        },
        pipeline_mode="deep",
    )
    llm_response = json.dumps({
        "strategy": "Upgrade Java 7 to 17",
        "syntax_changes": ["use var"],
        "api_changes": [],
        "risk_areas": [],
        "estimated_effort": "low",
    })
    agent = PlannerAgent(make_mock_llm(llm_response))
    result = await agent(state)

    assert result.migration_type == MigrationType.UPGRADE_VERSION
    assert result.migration_plan is not None


@pytest.mark.asyncio
async def test_planner_detects_conversion():
    state = MigrationState(
        source_code="code",
        source_language="java",
        source_version="8",
        target_language="python",
        target_version="3.12",
        code_metrics={
            "deprecated_patterns": [],
            "migration_challenges": [],
            "complexity": "medium",
        },
        pipeline_mode="deep",
    )
    agent = PlannerAgent(make_mock_llm(side_effect=Exception("LLM down")))
    result = await agent(state)

    # Should fall back to recipe-based plan
    assert result.migration_type == MigrationType.CONVERT_LANGUAGE
    assert result.migration_plan is not None
    assert len(result.migration_plan.get("syntax_changes", [])) > 0


# ── MIGRATION_RECIPES ─────────────────────────────────────────────────────────

def test_java_java_recipe_exists():
    recipe = MIGRATION_RECIPES.get(("java", "java"))
    assert recipe is not None
    assert len(recipe["syntax_changes"]) > 0

def test_java_python_recipe_exists():
    recipe = MIGRATION_RECIPES.get(("java", "python"))
    assert recipe is not None
    assert any("ArrayList" in s for s in recipe["syntax_changes"])
