"""
Unit tests for CodeMigrateAI backend
Run: pytest tests.py -v
"""

import pytest
import json
import re
from unittest.mock import AsyncMock, patch

# We import only the pure functions — no need to start the server
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    MigrationState,
    MigrationType,
    extract_json,
    _strip_fences,
    analyzer_agent,
    planner_agent,
    MIGRATION_RECIPES,
)


# ── extract_json ──────────────────────────────────────────────────────────────

def test_extract_json_from_fenced_block():
    text = '```json\n{"key": "value"}\n```'
    assert extract_json(text) == {"key": "value"}

def test_extract_json_from_raw_object():
    text = 'Some preamble {"complexity": "high"} trailing'
    assert extract_json(text) == {"complexity": "high"}

def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


# ── _strip_fences ─────────────────────────────────────────────────────────────

def test_strip_fences_removes_code_block():
    raw = "```python\ndef hello(): pass\n```"
    assert _strip_fences(raw) == "def hello(): pass"

def test_strip_fences_passthrough_clean_code():
    code = "public class Foo {}"
    assert _strip_fences(code) == "public class Foo {}"

def test_strip_fences_removes_preamble():
    raw = "Here is the migrated code:\ndef hello(): pass"
    result = _strip_fences(raw)
    assert "Here is" not in result


# ── MigrationState ────────────────────────────────────────────────────────────

def test_state_record_success():
    state = MigrationState(source_code="x", source_language="java",
                           source_version="7", target_language="java", target_version="17")
    state.record_success("TestAgent", "done", {"lines": 10})
    assert "TestAgent" in state.agents_done
    assert state.reports[0].status == "success"

def test_state_record_error():
    state = MigrationState(source_code="x", source_language="java",
                           source_version="7", target_language="python", target_version="3.12")
    state.record_error("TestAgent", "something broke")
    assert state.errors == ["[TestAgent] something broke"]
    assert state.reports[0].status == "error"


# ── AnalyzerAgent (mocked LLM) ────────────────────────────────────────────────

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
    with patch("main.call_llm", new=AsyncMock(return_value=llm_response)):
        result = await analyzer_agent(state)

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
    with patch("main.call_llm", side_effect=Exception("LLM unavailable")):
        result = await analyzer_agent(state)

    # Should still succeed — graceful degradation
    assert result.code_metrics is not None
    assert result.code_metrics["total_lines"] == 2
    assert "AnalyzerAgent" in result.agents_done


# ── PlannerAgent ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_detects_upgrade():
    state = MigrationState(
        source_code="code", source_language="java", source_version="7",
        target_language="java", target_version="17",
        code_metrics={"deprecated_patterns": [], "migration_challenges": [], "complexity": "low"},
    )
    llm_response = json.dumps({
        "strategy": "Upgrade Java 7 to 17",
        "syntax_changes": ["use var"], "api_changes": [], "risk_areas": [],
        "estimated_effort": "low",
    })
    with patch("main.call_llm", new=AsyncMock(return_value=llm_response)):
        result = await planner_agent(state)

    assert result.migration_type == MigrationType.UPGRADE_VERSION
    assert result.migration_plan is not None


@pytest.mark.asyncio
async def test_planner_detects_conversion():
    state = MigrationState(
        source_code="code", source_language="java", source_version="8",
        target_language="python", target_version="3.12",
        code_metrics={"deprecated_patterns": [], "migration_challenges": [], "complexity": "medium"},
    )
    with patch("main.call_llm", side_effect=Exception("LLM down")):
        result = await planner_agent(state)

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