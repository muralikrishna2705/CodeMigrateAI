import json
import logging
import re
from typing import Any

from config import get_settings
from llm.language_profiles import ProfileRegistry, get_profile
from llm.prompt_composer import PromptComposer
from models.state import MigrationState, MigrationType

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.MigratorAgent")


class MigratorAgent(BaseAgent):
    name = "MigratorAgent"
    requires_llm = True

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self.stream_callback = config.get("stream_callback") if config else None
        self.prompt_composer = PromptComposer()

    async def run(self, state: MigrationState) -> AgentResult:
        source_profile = get_profile(state.source_language)
        target_profile = get_profile(state.target_language)
        state.migration_type = self._detect_migration_type(state)

        prompt = self.prompt_composer.compose(
            source_profile=source_profile,
            target_profile=target_profile,
            source_version=state.source_version,
            target_version=state.target_version,
            source_code=state.source_code[:get_settings().max_llm_code_chars],
            analyzer_context=state.code_metrics or {},
            migration_type=state.migration_type.value,
        )
        system_prompt = self._build_system_prompt(state)

        # Inject planner context if available (PlannerAgent runs upstream)
        if state.inline_plan:
            prompt = f"MIGRATION PLAN:\n{state.inline_plan}\n\n---\n\n{prompt}"

        # Inject RAG reference context if the RetrieverAgent found any (Phase 2).
        # rag_context already carries its own trailing separator. When retrieval
        # found nothing, an *empty* context prompt is indistinguishable from a
        # grounded one, so we instead inject an explicit low-confidence notice
        # that tells the model to stay conservative rather than invent APIs.
        if state.rag_context:
            prompt = f"{state.rag_context}{prompt}"
        else:
            prompt = f"{self._ungrounded_notice(state)}\n\n---\n\n{prompt}"

        raw_output = await self._call_llm(prompt, system_prompt, fmt="json")
        try:
            parsed = self._parse_llm_output(raw_output)
        except ValueError as first_error:
            log.warning("LLM JSON parse failed: %s", first_error)
            retry_prompt = self._build_retry_prompt(raw_output)
            try:
                retry_output = await self.llm.call_llm(
                    retry_prompt, system_prompt, fmt="json"
                )
                parsed = self._parse_llm_output(retry_output)
            except Exception as retry_error:
                log.warning("LLM JSON retry failed: %s", retry_error)
                parsed = self._fallback_from_raw(raw_output)

        state.inline_plan = parsed["plan_summary"].strip()
        state.migrated_code = parsed["migrated_code"].strip()

        if not state.migrated_code:
            raise ValueError("LLM returned empty migrated_code")

        output_lines = len(state.migrated_code.splitlines())
        details = {
            "input_lines": (state.code_metrics or {}).get("total_lines"),
            "output_lines": output_lines,
            "migration_type": state.migration_type.value,
            "plan_summary": state.inline_plan,
        }

        summary = (
            f"Generated {output_lines} lines of "
            f"{state.target_language} {state.target_version}"
        )

        # Advisory grounding check: surface imports that look invented so they
        # are visible in the report without failing an otherwise-valid migration.
        if get_settings().enable_grounding_check:
            from agents.grounding import check_import_grounding

            grounding = check_import_grounding(
                migrated_code=state.migrated_code,
                target_language=state.target_language,
                source_code=state.source_code,
                rag_context=state.rag_context or "",
            )
            details["grounding"] = grounding
            unverified = grounding.get("unverified_imports") or []
            if unverified:
                log.warning(
                    "Ungrounded imports in migrated code: %s", ", ".join(unverified)
                )
                summary += f" · ⚠ {len(unverified)} ungrounded import(s)"

        return AgentResult(success=True, summary=summary, details=details)

    async def _call_llm(
        self, prompt: str, system_prompt: str, fmt: str | None = None
    ) -> str:
        if not self.stream_callback:
            return await self.llm.call_llm(prompt, system_prompt, fmt=fmt)

        raw_output = ""
        async for token in self.llm.stream_llm(prompt, system_prompt, fmt=fmt):
            raw_output += token
            await self.stream_callback(token)
        return raw_output

    def _detect_migration_type(self, state: MigrationState) -> MigrationType:
        source = ProfileRegistry.normalize(state.source_language)
        target = ProfileRegistry.normalize(state.target_language)
        if source == target:
            return MigrationType.UPGRADE_VERSION
        return MigrationType.CONVERT_LANGUAGE

    def _ungrounded_notice(self, state: MigrationState) -> str:
        """Anti-hallucination guard used when RAG retrieved no reference examples.

        With no verified examples to ground against, an LLM is most likely to
        invent plausible-but-wrong third-party APIs. This notice constrains it to
        the standard library and flags anything uncertain instead of guessing.
        """
        return (
            "GROUNDING NOTICE — NO REFERENCE EXAMPLES\n"
            f"No verified {state.target_language} reference examples were "
            "retrieved for this migration. Work conservatively:\n"
            f"- Use only well-established {state.target_language} "
            f"{state.target_version} standard-library and language features.\n"
            "- Do NOT invent third-party libraries, package names, or APIs you "
            "are not certain exist.\n"
            "- Preserve the source behavior exactly; prefer a faithful, minimal "
            "translation over idioms you are unsure about.\n"
            "- If a construct has no clear standard equivalent, keep the closest "
            "faithful translation rather than guessing an unfamiliar API."
        )

    def _build_system_prompt(self, state: MigrationState) -> str:
        if state.migration_type == MigrationType.UPGRADE_VERSION:
            return (
                f"You are a senior {state.target_language} modernization engineer. "
                "Preserve behavior exactly, apply safe target-version idioms, "
                "and return only the requested JSON object."
            )
        return (
            "You are a senior polyglot migration engineer. Preserve behavior "
            f"while converting {state.source_language} to idiomatic "
            f"{state.target_language}. Return only the requested JSON object."
        )

    def _parse_llm_output(self, raw: str) -> dict[str, str]:
        data = self._extract_json(raw)
        plan_summary = str(data.get("plan_summary", "")).strip()
        migrated_code = str(
            data.get("migrated_code")
            or data.get("code")
            or data.get("output")
            or data.get("result")
            or data.get("source_code")
            or ""
        ).strip()
        if not migrated_code:
            raise ValueError("JSON response did not contain migrated_code")
        if not plan_summary:
            plan_summary = "Migration plan generated by the LLM."
        return {"plan_summary": plan_summary, "migrated_code": migrated_code}

    def _extract_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try stripping a conversational preamble and reparsing.
        preamble_stripped = re.sub(
            r"(?i)^(?:here(?:'s| is) (?:the |my |your )?"
            r"(?:migrated|converted|upgraded) code[:\s]*|output[:\s]*|"
            r"result[:\s]*|sure[^.]*\.)",
            "",
            text,
        ).strip()
        if preamble_stripped != text:
            try:
                data = json.loads(preamble_stripped)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        if hasattr(self.llm, "extract_json"):
            try:
                data = self.llm.extract_json(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw):
            try:
                data = json.loads(block.strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        for candidate in sorted(
            re.findall(r"\{[\s\S]*\}", raw),
            key=len,
            reverse=True,
        ):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON object in LLM response")

    def _build_retry_prompt(self, raw_output: str) -> str:
        return (
            "Your previous response was not valid JSON for CodeMigrateAI.\n"
            "Rewrite it as a single valid JSON object with exactly these keys:\n"
            '  "plan_summary": string\n'
            '  "migrated_code": string\n'
            "Do not add markdown, comments outside JSON, or extra keys.\n\n"
            f"Previous response:\n{raw_output[:5000]}"
        )

    def _fallback_from_raw(self, raw: str) -> dict[str, str]:
        code = self._strip_fences(raw)
        return {
            "plan_summary": (
                "The LLM returned unstructured output; CodeMigrateAI extracted "
                "the migrated code directly."
            ),
            "migrated_code": code,
        }

    def _strip_fences(self, raw: str) -> str:
        fenced = re.match(r"^```[\w+-]*\n([\s\S]*?)```\s*$", raw.strip())
        if fenced:
            return fenced.group(1).strip()
        cleaned = re.sub(r"^```[\w+-]*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        cleaned = re.sub(
            r"(?i)^(here(?:'s| is) the (?:migrated|converted|upgraded) code[:\s]*\n)",
            "",
            cleaned,
        )
        return cleaned.strip()
