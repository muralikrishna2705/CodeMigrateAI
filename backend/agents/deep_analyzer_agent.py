import logging

from config import get_settings
from llm.language_profiles import get_profile
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.DeepAnalyzerAgent")


class DeepAnalyzerAgent(BaseAgent):
    name = "DeepAnalyzerAgent"
    requires_llm = True

    async def run(self, state: MigrationState) -> AgentResult:
        settings = get_settings()
        profile = get_profile(state.source_language)

        prompt = (
            f"You are an expert code analyst. Perform a deep analysis of this "
            f"{state.source_language} {state.source_version} code for migration to "
            f"{state.target_language} {state.target_version}.\n\n"
            "Identify:\n"
            "1. Complex language-specific constructs (generics, reflection, async/await, macros)\n"
            "2. Standard library dependencies that need migration\n"
            "3. Inheritance hierarchies and design patterns\n"
            "4. Potential breaking changes\n"
            "5. Recommended migration strategy\n\n"
            f"CODE:\n```{profile.language_id}\n"
            f"{state.source_code[:settings.max_llm_code_chars]}\n```\n\n"
            "OUTPUT (JSON only):\n"
            "{\n"
            '  "complex_constructs": [...],\n'
            '  "stdlib_dependencies": [...],\n'
            '  "inheritance_depth": int,\n'
            '  "design_patterns": [...],\n'
            '  "breaking_changes": [...],\n'
            '  "recommended_strategy": "..."\n'
            "}"
        )

        raw = await self.llm.call_llm(
            prompt,
            system_prompt=(
                "You are a senior compiler engineer specializing in "
                "cross-language analysis. Output only valid JSON."
            ),
            fmt="json",
        )

        parsed = {}
        if hasattr(self.llm, "extract_json"):
            try:
                parsed = self.llm.extract_json(raw)
            except Exception as exc:
                log.warning("Deep analysis JSON parse failed: %s", exc)

        # Enrich the shared code metrics so the plan/migrate steps can use the
        # deep-analysis findings (complexity stays as computed by AnalyzerAgent).
        if parsed:
            metrics = dict(state.code_metrics or {})
            metrics["deep_analysis"] = parsed
            state.code_metrics = metrics

        return AgentResult(
            success=True,
            summary=(
                f"Deep analysis: {len(parsed.get('complex_constructs', []))} "
                "complex constructs found"
            ),
            details=parsed,
        )
