"""
CodeMigrateAI — Backend
MTech Final Year Project

Three-agent pipeline:
  AnalyzerAgent  →  PlannerAgent  →  MigratorAgent

LLM  : deepseek-coder:1.3b via Ollama (~800 MB, free, open-source)
       Ollama runs on Windows host — Docker reaches it via host.docker.internal
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CodeMigrateAI")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Ollama is installed on Windows and already running.
# host.docker.internal lets the Docker container reach the Windows host.
# To change model: edit LLM_MODEL (must match what you have pulled in Ollama).
#   deepseek-coder:1.3b  →  800 MB  (current)
#   deepseek-coder:6.7b  →  4 GB
#   qwen2.5-coder:7b     →  5 GB
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL      = "http://host.docker.internal:11434"
LLM_MODEL       = "deepseek-coder:1.3b"
LLM_TIMEOUT_SEC = 600.0   # 10 min — small models are slow on CPU
MAX_CODE_CHARS  = 50_000

# ─────────────────────────────────────────────────────────────────────────────
# STATE + SUPPORTED LANGUAGES
# ─────────────────────────────────────────────────────────────────────────────

# SUPPORTED LANGUAGES  (drives the frontend dropdowns)
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    {"id": "java",       "name": "Java",       "versions": ["7", "8", "11", "17", "21"]},
    {"id": "python",     "name": "Python",     "versions": ["2.7", "3.8", "3.10", "3.12"]},
    {"id": "javascript", "name": "JavaScript", "versions": ["ES5", "ES6", "ES2020", "ES2022"]},
    {"id": "typescript", "name": "TypeScript", "versions": ["3.x", "4.x", "5.x"]},
    {"id": "csharp",     "name": "C#",         "versions": ["6", "8", "10", "12"]},
    {"id": "go",         "name": "Go",         "versions": ["1.18", "1.20", "1.22"]},
    {"id": "kotlin",     "name": "Kotlin",     "versions": ["1.7", "1.9", "2.0"]},
    {"id": "rust",       "name": "Rust",       "versions": ["1.70", "1.80"]},
    {"id": "cpp",        "name": "C++",        "versions": ["14", "17", "20", "23"]},
]

# ──────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (passed through every agent, mutated in-place)
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SHARED AGENT STATE  (Pydantic model passed through every agent)
# ─────────────────────────────────────────────────────────────────────────────

class MigrationType(str, Enum):
    UPGRADE_VERSION  = "upgrade_version"   # same language, newer version
    CONVERT_LANGUAGE = "convert_language"  # different language entirely


class AgentReport(BaseModel):
    """Structured report produced by each agent after it finishes."""
    agent:   str
    status:  str            # "success" | "error"
    summary: str
    details: Optional[dict] = None


class MigrationState(BaseModel):
    """
    Central state object shared across the entire agent pipeline.
    Each agent reads from it and writes its results back into it.
    """
    # ── Inputs (set at request time) ──
    source_code:     str = ""
    source_language: str = ""
    source_version:  str = ""
    target_language: str = ""
    target_version:  str = ""

    # ── Inter-agent data (written by each agent, read by the next) ──
    migration_type:  MigrationType = MigrationType.UPGRADE_VERSION
    code_metrics:    Optional[dict] = None   # written by AnalyzerAgent
    migration_plan:  Optional[dict] = None   # written by PlannerAgent
    migrated_code:   str = ""                # written by MigratorAgent

    # ── Pipeline tracking ──
    reports:         list[AgentReport] = Field(default_factory=list)
    errors:          list[str]         = Field(default_factory=list)
    agents_done:     list[str]         = Field(default_factory=list)

    # ── Helpers ──
    def record_success(self, agent: str, summary: str, details: dict | None = None):
        self.reports.append(AgentReport(agent=agent, status="success",
                                        summary=summary, details=details))
        self.agents_done.append(agent)

    def record_error(self, agent: str, message: str):
        self.reports.append(AgentReport(agent=agent, status="error", summary=message))
        self.errors.append(f"[{agent}] {message}")
        self.agents_done.append(agent)

# ──────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  (calls DeepSeek-Coder-V2 via local Ollama)
# ──────────────────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to the local Ollama server and return the response text.
    Uses DeepSeek-Coder-V2 — the best free open-source coding model (2025).
    """
    payload: dict[str, Any] = {
        "model":  LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,   # near-deterministic for code generation
            "num_predict": 2048,
            "top_p": 0.9,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT_SEC, write=30.0, pool=5.0)
    ) as client:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()


def extract_json(raw_text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.
    Handles markdown fences (```json ... ```) and raw JSON objects.
    """
    # Try fenced JSON blocks first
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            pass

    # Try finding a raw JSON object
    raw_objects = re.findall(r"\{[\s\S]*\}", raw_text)
    for obj in sorted(raw_objects, key=len, reverse=True):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


async def ollama_health() -> bool:
    """Check if local Ollama server is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — ANALYZER
# Responsibility: understand the source code before migration begins
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  — Ollama /api/generate
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to Ollama and return the generated text.
    Uses /api/generate — Ollama's native endpoint.
    Ollama and the model start automatically via docker-compose.
    """
    payload: dict[str, Any] = {
        "model":  LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 2048,
            "top_p": 0.9,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    log.info(f"Ollama  model={LLM_MODEL}  prompt={len(prompt)} chars")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT_SEC, write=30.0, pool=5.0)
    ) as client:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()


def extract_json(raw_text: str) -> dict:
    """Extract first valid JSON object from LLM output (handles ```json fences)."""
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            pass
    for obj in sorted(re.findall(r"\{[\s\S]*\}", raw_text), key=len, reverse=True):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON in LLM response")


async def ollama_health() -> bool:
    """Ping Ollama. Returns True when it is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# AGENTS + PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Call Ollama /api/generate and return the response text."""
    payload: dict[str, Any] = {
        "model":  LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 2048,
            "top_p": 0.9,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    log.info(f"Ollama  model={LLM_MODEL}  prompt={len(prompt)} chars")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=LLM_TIMEOUT_SEC, write=30.0, pool=5.0)
    ) as client:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()


def extract_json(raw_text: str) -> dict:
    """Extract first valid JSON object from LLM output."""
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            pass
    for obj in sorted(re.findall(r"\{[\s\S]*\}", raw_text), key=len, reverse=True):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON in LLM response")


async def ollama_health() -> bool:
    """Ping Ollama. Returns True when reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# AGENTS + PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def analyzer_agent(state: MigrationState) -> MigrationState:
    """
    AnalyzerAgent — Phase 1 of the pipeline.

    Does two things:
    1. Static analysis: counts lines, branches, classes, methods — no LLM needed,
       so this is always fast and never fails.
    2. Semantic analysis: asks DeepSeek-Coder-V2 to identify deprecated patterns,
       migration challenges, and code complexity. Falls back to static-only if LLM fails.
    """
    log.info("AnalyzerAgent: starting code analysis")
    code = state.source_code

    # ── Static metrics (regex-based, no LLM) ──────────────────────────────────
    lines    = code.splitlines()
    branches = len(re.findall(r'\b(if|else|elif|for|while|switch|case|catch|except|try)\b', code))
    classes  = len(re.findall(r'\bclass\s+\w+', code))
    methods  = len(re.findall(r'\b(def|void|func|fn|fun|sub|function)\s+\w+\s*\(', code))
    imports  = len(re.findall(r'\b(import|require|include|using|from)\b', code))

    complexity = (
        "low"    if branches < 6  else
        "medium" if branches < 25 else
        "high"
    )

    static_metrics = {
        "total_lines":    len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        "branch_count":   branches,
        "class_count":    classes,
        "method_count":   methods,
        "import_count":   imports,
        "complexity":     complexity,
    }

    # ── Semantic analysis via DeepSeek-Coder-V2 ───────────────────────────────
    try:
        llm_prompt = (
            f"Analyze this {state.source_language} {state.source_version} code.\n"
            f"Migration target: {state.target_language} {state.target_version}\n\n"
            f"```{state.source_language}\n{code[:3500]}\n```\n\n"
            "Respond ONLY with this JSON (no other text):\n"
            "{\n"
            '  "deprecated_patterns": ["list what is outdated or deprecated"],\n'
            '  "migration_challenges": ["list specific things that will be hard to migrate"],\n'
            '  "key_constructs": ["list main language constructs used"],\n'
            '  "summary": "2 sentence plain-english summary of what this code does"\n'
            "}"
        )
        raw = await call_llm(llm_prompt,
                             system_prompt="You are a code analysis expert. "
                                           "Respond ONLY with valid JSON. No explanation.")
        semantic = extract_json(raw)
        metrics  = {**static_metrics, **semantic}
        log.info(f"AnalyzerAgent: semantic analysis complete, complexity={complexity}")

    except Exception as exc:
        log.warning(f"AnalyzerAgent: LLM call failed ({exc}), using static metrics only")
        metrics = {
            **static_metrics,
            "deprecated_patterns":  [],
            "migration_challenges": [],
            "key_constructs":       [],
            "summary":              f"Static analysis: {len(lines)} lines of {state.source_language} code.",
        }

    state.code_metrics = metrics
    state.record_success(
        "AnalyzerAgent",
        f"{metrics['total_lines']} lines · {metrics['class_count']} classes · "
        f"{metrics['method_count']} methods · complexity={complexity}",
        details=metrics,
    )
    return state

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — PLANNER
# Responsibility: decide HOW to migrate, before any code is written
# ──────────────────────────────────────────────────────────────────────────────

# Built-in recipe knowledge so the LLM gets grounded, accurate context
MIGRATION_RECIPES: dict[tuple[str, str], dict] = {
    ("java", "java"): {
        "syntax_changes": [
            "Replace 'new ArrayList<String>()' with 'new ArrayList<>()'  (diamond inference)",
            "Replace 'Collections.sort(list)' with 'list.sort(null)'",
            "Replace 'new Date()' with 'LocalDateTime.now()'",
            "Replace raw for-loops with Stream API where applicable",
            "Use 'var' keyword for local variable type inference (Java 10+)",
            "Replace StringBuffer with StringBuilder",
            "Use switch expressions instead of switch statements (Java 14+)",
            "Replace anonymous Runnables with lambda expressions",
        ],
        "api_changes": [
            "java.util.Date → java.time.LocalDateTime / ZonedDateTime",
            "java.util.Calendar → java.time.LocalDate",
            "javax.* → jakarta.* (Jakarta EE 9+)",
            "Thread.stop() → cooperative interruption",
        ],
    },
    ("java", "python"): {
        "syntax_changes": [
            "Java class → Python class (remove 'public', add 'self' param to methods)",
            "ArrayList<T> → list",
            "HashMap<K,V> → dict",
            "System.out.println() → print()",
            "try-catch-finally → try-except-finally",
            "for (T x : list) → for x in list",
            "null → None",
            "Java streams → list comprehensions or generator expressions",
            "Getters/setters → Python @property",
            "final variables → convention (uppercase names)",
        ],
        "api_changes": [
            "java.util.Optional → typing.Optional or direct None check",
            "java.io.File → pathlib.Path",
            "java.util.Arrays.asList() → list()",
            "String.format() → f-strings",
        ],
    },
    ("python", "python"): {
        "syntax_changes": [
            "%-formatting / .format() → f-strings",
            "print statement (Py2) → print() function",
            "unicode / basestring → str",
            "dict.keys() used as list → list(dict.keys())",
            "Use walrus operator := where appropriate (Py 3.8+)",
            "Use match-case for pattern matching (Py 3.10+)",
            "Type hints on function signatures",
            "dataclasses instead of plain dicts for structured data",
        ],
        "api_changes": [
            "urllib2 → urllib.request",
            "ConfigParser → configparser",
            "cPickle → pickle",
            "reduce() → functools.reduce()",
        ],
    },
    ("javascript", "typescript"): {
        "syntax_changes": [
            "Add type annotations to all function parameters and return types",
            "Convert 'var' → 'let' or 'const' with correct types",
            "Create interfaces for object shapes",
            "Add generics to arrays: any[] → T[]",
            "Replace callback patterns with typed async/await",
            "Add strict null checks",
            "Use enum for string unions",
        ],
        "api_changes": [
            "require() → import (ES modules)",
            "module.exports → export default / named exports",
            "Add .d.ts files for untyped dependencies",
        ],
    },
}


async def planner_agent(state: MigrationState) -> MigrationState:
    """
    PlannerAgent — Phase 2 of the pipeline.

    Determines migration type (upgrade vs convert) and produces a structured
    migration plan. Injects built-in recipe knowledge so the LLM generates
    a grounded, accurate plan rather than hallucinating.
    """
    log.info("PlannerAgent: building migration plan")

    # ── Determine migration type ───────────────────────────────────────────────
    lang_aliases = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "c#": "csharp",  "c++": "cpp",      "golang": "go",
    }
    src_norm = lang_aliases.get(state.source_language.lower(), state.source_language.lower())
    tgt_norm = lang_aliases.get(state.target_language.lower(), state.target_language.lower())

    state.migration_type = (
        MigrationType.UPGRADE_VERSION  if src_norm == tgt_norm else
        MigrationType.CONVERT_LANGUAGE
    )

    # ── Pull recipe knowledge ──────────────────────────────────────────────────
    recipe = MIGRATION_RECIPES.get((src_norm, tgt_norm), {})
    recipe_context = ""
    if recipe:
        syntax_list = "\n".join(f"  - {s}" for s in recipe.get("syntax_changes", []))
        api_list    = "\n".join(f"  - {a}" for a in recipe.get("api_changes", []))
        recipe_context = f"\nKnown transformation patterns:\nSyntax:\n{syntax_list}\nAPI:\n{api_list}\n"

    # ── Pull analyzer findings ─────────────────────────────────────────────────
    metrics   = state.code_metrics or {}
    deprecated = metrics.get("deprecated_patterns", [])
    challenges = metrics.get("migration_challenges", [])

    analysis_context = ""
    if deprecated or challenges:
        analysis_context = (
            f"\nAnalyzer findings:\n"
            f"  Deprecated patterns: {deprecated}\n"
            f"  Migration challenges: {challenges}\n"
        )

    # ── Ask LLM to create a complete plan ─────────────────────────────────────
    try:
        llm_prompt = (
            f"Create a migration plan:\n"
            f"  Source: {state.source_language} {state.source_version}\n"
            f"  Target: {state.target_language} {state.target_version}\n"
            f"  Type: {state.migration_type.value}\n"
            f"{recipe_context}"
            f"{analysis_context}\n"
            "Respond ONLY with this JSON (no other text):\n"
            "{\n"
            '  "strategy": "one sentence describing the overall approach",\n'
            '  "syntax_changes": ["specific syntax transformation 1", "..."],\n'
            '  "api_changes": ["API mapping 1", "..."],\n'
            '  "risk_areas": ["thing that needs careful handling 1", "..."],\n'
            '  "estimated_effort": "low | medium | high"\n'
            "}"
        )
        raw  = await call_llm(llm_prompt,
                              system_prompt="You are a senior migration architect. "
                                            "Return ONLY valid JSON.")
        plan = extract_json(raw)

    except Exception as exc:
        log.warning(f"PlannerAgent: LLM failed ({exc}), using recipe-only plan")
        plan = {
            "strategy":       f"Migrate {state.source_language} {state.source_version} "
                               f"to {state.target_language} {state.target_version}",
            "syntax_changes": recipe.get("syntax_changes", []),
            "api_changes":    recipe.get("api_changes", []),
            "risk_areas":     challenges,
            "estimated_effort": metrics.get("complexity", "medium"),
        }

    plan["migration_type"] = state.migration_type.value
    state.migration_plan   = plan

    state.record_success(
        "PlannerAgent",
        f"type={state.migration_type.value} · "
        f"{len(plan.get('syntax_changes', []))} syntax transforms · "
        f"effort={plan.get('estimated_effort', '?')}",
        details=plan,
    )
    return state

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3 — MIGRATOR
# Responsibility: generate the actual migrated code using the plan as context
# ──────────────────────────────────────────────────────────────────────────────

async def migrator_agent(state: MigrationState) -> MigrationState:
    """
    MigratorAgent — Phase 3 of the pipeline.

    Uses DeepSeek-Coder-V2 with the full migration plan and analyzer findings
    as context to generate the migrated/converted code. The plan acts as a
    chain-of-thought scaffold, making the output significantly more accurate.
    """
    log.info("MigratorAgent: generating migrated code")

    plan     = state.migration_plan or {}
    metrics  = state.code_metrics   or {}
    is_upgrade = state.migration_type == MigrationType.UPGRADE_VERSION

    # ── System prompt: tell the model exactly what role it plays ──────────────
    if is_upgrade:
        system_prompt = (
            f"You are an expert {state.source_language} developer who specialises in "
            f"version migration. Your task is to upgrade {state.source_language} code "
            f"from version {state.source_version} to {state.target_version}.\n"
            "Rules:\n"
            "  1. Output ONLY the migrated code — no explanations, no markdown fences\n"
            "  2. Preserve ALL original logic and business functionality exactly\n"
            "  3. Apply every transformation from the migration plan below\n"
            "  4. Add a short comment only where a migration change is non-obvious\n"
            "  5. Keep the same class/method structure unless restructuring is required"
        )
    else:
        system_prompt = (
            f"You are an expert polyglot software engineer. Your task is to convert "
            f"{state.source_language} code to idiomatic {state.target_language} "
            f"{state.target_version}.\n"
            "Rules:\n"
            "  1. Output ONLY the converted code — no explanations, no markdown fences\n"
            "  2. Preserve ALL original logic and business functionality exactly\n"
            "  3. Use idiomatic {state.target_language} — do NOT do a word-for-word translation\n"
            "  4. Apply every transformation from the migration plan below\n"
            "  5. Use {state.target_language} standard library equivalents throughout"
        )

    # ── Build context block from plan ─────────────────────────────────────────
    context_lines = []
    if plan.get("strategy"):
        context_lines.append(f"Migration strategy: {plan['strategy']}")
    if plan.get("syntax_changes"):
        context_lines.append("Syntax transformations to apply:")
        context_lines.extend(f"  - {c}" for c in plan["syntax_changes"][:10])
    if plan.get("api_changes"):
        context_lines.append("API mappings to apply:")
        context_lines.extend(f"  - {a}" for a in plan["api_changes"][:10])
    if metrics.get("deprecated_patterns"):
        context_lines.append("Deprecated patterns to fix:")
        context_lines.extend(f"  - {d}" for d in metrics["deprecated_patterns"][:5])
    if plan.get("risk_areas"):
        context_lines.append("Handle these carefully:")
        context_lines.extend(f"  ⚠ {r}" for r in plan["risk_areas"][:5])

    context = "\n".join(context_lines)
    action  = (
        f"Upgrade from {state.source_version} to {state.target_version}"
        if is_upgrade else
        f"Convert to {state.target_language} {state.target_version}"
    )

    # Cap source code to keep prompt manageable for small local model
    code_for_prompt = state.source_code[:4000]

    user_prompt = (
        f"{action}\n\n"
        f"{context}\n\n"
        f"SOURCE CODE ({state.source_language} {state.source_version}):\n"
        f"{code_for_prompt}\n\n"
        f"OUTPUT ({state.target_language} {state.target_version}) — "
        "only the code, nothing else:"
    )

    # ── Call LLM and clean output ─────────────────────────────────────────────
    raw_output   = await call_llm(user_prompt, system_prompt)
    cleaned_code = _strip_fences(raw_output)

    if not cleaned_code.strip():
        raise ValueError("DeepSeek-Coder-V2 returned empty code output")

    state.migrated_code = cleaned_code
    state.record_success(
        "MigratorAgent",
        f"Generated {len(cleaned_code.splitlines())} lines of "
        f"{state.target_language} {state.target_version}",
        details={
            "input_lines":  metrics.get("total_lines"),
            "output_lines": len(cleaned_code.splitlines()),
            "migration_type": state.migration_type.value,
        },
    )
    return state


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that LLMs sometimes add despite instructions."""
    # Handle ```language\n...\n```
    fenced = re.match(r"^```[\w]*\n([\s\S]*?)```\s*$", raw.strip())
    if fenced:
        return fenced.group(1).strip()
    # Remove partial fences
    cleaned = re.sub(r"^```[\w]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    # Remove common preamble phrases
    cleaned = re.sub(
        r"(?i)^(here(?:'s| is) the (?:migrated|converted|upgraded) code[:\s]*\n)",
        "", cleaned
    )
    return cleaned.strip()

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE = [analyzer_agent, planner_agent, migrator_agent]


async def run_migration_pipeline(state: MigrationState) -> MigrationState:
    """
    Execute the three-agent pipeline sequentially.

    Design decisions:
    - AnalyzerAgent and PlannerAgent failures are non-fatal: the Migrator
      continues with reduced context (graceful degradation).
    - MigratorAgent failure is fatal: there is no migrated code to return.
    - Each agent gets the full shared state, so every agent can read what
      every previous agent wrote.
    """
    log.info("=" * 60)
    log.info(f"Pipeline START: {state.source_language} {state.source_version} → "
             f"{state.target_language} {state.target_version}")
    log.info(f"Source size: {len(state.source_code)} chars")
    log.info("=" * 60)

    for step, agent_fn in enumerate(PIPELINE, start=1):
        agent_name = agent_fn.__name__
        log.info(f"[{step}/{len(PIPELINE)}] Running {agent_name}...")
        try:
            state = await agent_fn(state)
        except Exception as exc:
            log.exception(f"{agent_name} raised unhandled exception")
            state.record_error(agent_name, str(exc))
            if agent_name == "migrator_agent":
                # Cannot proceed without output
                break

    log.info(f"Pipeline END. Agents done: {state.agents_done} | Errors: {len(state.errors)}")
    return state

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(f"OLLAMA_URL : {OLLAMA_URL}")
    log.info(f"LLM_MODEL  : {LLM_MODEL}")
    alive = await ollama_health()
    if alive:
        log.info("✅ Ollama is reachable and ready")
    else:
        log.warning("⚠  Ollama not reachable — check that Ollama is running on Windows")
    yield
    log.info("👋 Shutdown complete")


app = FastAPI(
    title="CodeMigrateAI",
    description="AI-driven code migration — MTech Final Year Project",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class MigrateRequest(BaseModel):
    source_code:     str = Field(..., min_length=1)
    source_language: str = Field(..., min_length=1)
    source_version:  str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    target_version:  str = Field(..., min_length=1)

    @field_validator("source_code")
    @classmethod
    def enforce_size_limit(cls, v: str) -> str:
        if len(v) > MAX_CODE_CHARS:
            raise ValueError(f"Code too large ({len(v)}/{MAX_CODE_CHARS} chars)")
        return v


class MigrateResponse(BaseModel):
    success:          bool
    migrated_code:    str
    migration_type:   str
    source_language:  str
    source_version:   str
    target_language:  str
    target_version:   str
    reports:          list[AgentReport]
    errors:           list[str]
    agents_completed: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by the frontend status pill."""
    alive = await ollama_health()
    return {
        "status": "ok",
        "model":  LLM_MODEL,
        "ollama": "connected" if alive else "unavailable",
    }


@app.get("/languages")
async def get_languages():
    """All languages shown in the source / target dropdowns."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/migrate", response_model=MigrateResponse)
async def migrate(request: MigrateRequest):
    """
    Core endpoint. Runs AnalyzerAgent → PlannerAgent → MigratorAgent
    and returns the migrated code. Called by the frontend Run Migration button.
    """
    if not await ollama_health():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {OLLAMA_URL}. "
                f"Make sure Ollama is running on Windows "
                f"and model '{LLM_MODEL}' is pulled."
            ),
        )

    state = MigrationState(
        source_code=request.source_code,
        source_language=request.source_language,
        source_version=request.source_version,
        target_language=request.target_language,
        target_version=request.target_version,
    )

    try:
        final_state = await run_migration_pipeline(state)
    except Exception as exc:
        log.exception("Pipeline crashed")
        raise HTTPException(status_code=500, detail=str(exc))

    return MigrateResponse(
        success=bool(final_state.migrated_code) and not final_state.errors,
        migrated_code=final_state.migrated_code,
        migration_type=final_state.migration_type.value,
        source_language=final_state.source_language,
        source_version=final_state.source_version,
        target_language=final_state.target_language,
        target_version=final_state.target_version,
        reports=final_state.reports,
        errors=final_state.errors,
        agents_completed=final_state.agents_done,
    )