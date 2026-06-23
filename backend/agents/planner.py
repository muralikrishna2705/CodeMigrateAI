import logging

from agents.base import AgentResult, BaseAgent
from llm.prompts import PLANNER_PROMPT
from models.state import MigrationState, MigrationType

log = logging.getLogger("CodeMigrateAI.PlannerAgent")

MIGRATION_RECIPES: dict[tuple[str, str], dict] = {
    ("java", "java"): {
        "syntax_changes": [
            (
                "Replace 'new ArrayList<String>()' with 'new ArrayList<>()' "
                "(diamond inference)"
            ),
            "Replace 'Collections.sort(list)' with 'list.sort(null)'",
            "Replace 'new Date()' with 'LocalDateTime.now()'",
            "Replace raw for-loops with Stream API where applicable",
            (
                "Use 'var' keyword for local variable type inference "
                "(Java 10+)"
            ),
            "Replace StringBuffer with StringBuilder",
            (
                "Use switch expressions instead of switch statements "
                "(Java 14+)"
            ),
            "Replace anonymous Runnables with lambda expressions",
        ],
        "api_changes": [
            "java.util.Date to java.time.LocalDateTime / ZonedDateTime",
            "java.util.Calendar to java.time.LocalDate",
            "javax.* to jakarta.* (Jakarta EE 9+)",
            "Thread.stop() to cooperative interruption",
        ],
    },
    ("java", "python"): {
        "syntax_changes": [
            "Java class to Python class (remove 'public', add 'self' param to methods)",
            "ArrayList<T> to list",
            "HashMap<K,V> to dict",
            "System.out.println() to print()",
            "try-catch-finally to try-except-finally",
            "for (T x : list) to for x in list",
            "null to None",
            "Java streams to list comprehensions or generator expressions",
            "Getters/setters to Python @property",
            "final variables to convention (uppercase names)",
        ],
        "api_changes": [
            "java.util.Optional to typing.Optional or direct None check",
            "java.io.File to pathlib.Path",
            "java.util.Arrays.asList() to list()",
            "String.format() to f-strings",
        ],
    },
    ("python", "python"): {
        "syntax_changes": [
            "%-formatting / .format() to f-strings",
            "print statement (Py2) to print() function",
            "unicode / basestring to str",
            "dict.keys() used as list to list(dict.keys())",
            "Use walrus operator := where appropriate (Py 3.8+)",
            "Use match-case for pattern matching (Py 3.10+)",
            "Type hints on function signatures",
            "dataclasses instead of plain dicts for structured data",
        ],
        "api_changes": [
            "urllib2 to urllib.request",
            "ConfigParser to configparser",
            "cPickle to pickle",
            "reduce() to functools.reduce()",
        ],
    },
    ("javascript", "typescript"): {
        "syntax_changes": [
            "Add type annotations to all function parameters and return types",
            "Convert 'var' to 'let' or 'const' with correct types",
            "Create interfaces for object shapes",
            "Add generics to arrays: any[] to T[]",
            "Replace callback patterns with typed async/await",
            "Add strict null checks",
            "Use enum for string unions",
        ],
        "api_changes": [
            "require() to import (ES modules)",
            "module.exports to export default / named exports",
            "Add .d.ts files for untyped dependencies",
        ],
    },
}


class PlannerAgent(BaseAgent):
    name = "PlannerAgent"
    requires_llm = True
    fast_mode_skip = True

    async def run(self, state: MigrationState) -> AgentResult:
        lang_aliases = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "c#": "csharp",
            "c++": "cpp",
            "golang": "go",
        }
        src_norm = lang_aliases.get(
            state.source_language.lower(), state.source_language.lower()
        )
        tgt_norm = lang_aliases.get(
            state.target_language.lower(), state.target_language.lower()
        )

        state.migration_type = (
            MigrationType.UPGRADE_VERSION
            if src_norm == tgt_norm
            else MigrationType.CONVERT_LANGUAGE
        )

        recipe = MIGRATION_RECIPES.get((src_norm, tgt_norm), {})

        if state.pipeline_mode.value == "fast" and recipe:
            plan = self._build_recipe_plan(recipe, state)
        else:
            try:
                plan = await self._llm_plan(state, recipe)
            except Exception as e:
                log.warning("LLM plan failed, falling back to recipe: %s", e)
                plan = self._build_recipe_plan(recipe, state)

        plan["migration_type"] = state.migration_type.value
        state.migration_plan = plan

        return AgentResult(
            success=True,
            summary=(
                f"type={state.migration_type.value} · "
                f"{len(plan.get('syntax_changes', []))} syntax transforms · "
                f"effort={plan.get('estimated_effort', '?')}"
            ),
            details=plan,
        )

    def _build_recipe_plan(self, recipe: dict, state: MigrationState) -> dict:
        metrics = state.code_metrics or {}
        return {
            "strategy": (
                f"Migrate {state.source_language} {state.source_version} "
                f"to {state.target_language} {state.target_version} "
                "using built-in recipe"
            ),
            "syntax_changes": recipe.get("syntax_changes", []),
            "api_changes": recipe.get("api_changes", []),
            "risk_areas": metrics.get("migration_challenges", []),
            "estimated_effort": metrics.get("complexity", "medium"),
        }

    async def _llm_plan(self, state: MigrationState, recipe: dict) -> dict:
        recipe_context = ""
        if recipe:
            syntax_list = "\n".join(
                f"  - {s}" for s in recipe.get("syntax_changes", [])
            )
            api_list = "\n".join(
                f"  - {a}" for a in recipe.get("api_changes", [])
            )
            recipe_context = (
                f"\nKnown transformation patterns:\nSyntax:\n{syntax_list}"
                f"\nAPI:\n{api_list}\n"
            )

        metrics = state.code_metrics or {}
        deprecated = metrics.get("deprecated_patterns", [])
        challenges = metrics.get("migration_challenges", [])

        analysis_context = ""
        if deprecated or challenges:
            analysis_context = (
                f"\nAnalyzer findings:\n  Deprecated patterns: {deprecated}\n"
                f"  Migration challenges: {challenges}\n"
            )

        prompt = PLANNER_PROMPT.format(
            source_language=state.source_language,
            source_version=state.source_version,
            target_language=state.target_language,
            target_version=state.target_version,
            migration_type=state.migration_type.value,
            recipe_context=recipe_context,
            analysis_context=analysis_context,
        )

        raw = await self.llm.call_llm(
            prompt,
            system_prompt=(
                "You are a senior migration architect. "
                "Return ONLY valid JSON."
            ),
        )
        return self.llm.extract_json(raw)
