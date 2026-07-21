import logging
import re

from config import get_settings
from llm.code_stream import MigratedCodeStreamer
from llm.language_profiles import ProfileRegistry, get_profile
from llm.prompt_composer import PromptComposer
from llm.structured import coerce_or_none
from models.schemas import MigrationOutput
from models.state import MigrationState, MigrationType

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.MigratorAgent")


class MigratorAgent(BaseAgent):
    name = "MigratorAgent"
    requires_llm = True
    needs = ("stream_callback", "tools")

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

        # Reflection feedback (Dimension 3): on a reflect-driven re-migration the
        # graph carries the reviewer's critique here. Prepend it so the model
        # regenerates *addressing* the flaws instead of reproducing them. Gated on
        # an active (non-pass) verdict — the same signal migrate_node uses to detect
        # the re-entry — so a passing reflection's notes never leak into an
        # unrelated fix-loop re-migration.
        if state.reflection_feedback and state.reflection_recommendation != "pass":
            prompt = (
                "REVIEWER FEEDBACK on your previous attempt — address every point:\n"
                f"{state.reflection_feedback}\n\n---\n\n{prompt}"
            )

        parsed = await self._generate(prompt, system_prompt)

        state.inline_plan = parsed.plan_summary.strip()
        state.migrated_code = parsed.migrated_code.strip()

        if not state.migrated_code:
            raise ValueError("LLM returned empty migrated_code")

        # Self-reflection (Dimension 3): critique the generated code for
        # correctness/completeness and regenerate once if the model isn't
        # confident. Opt-in, best-effort, and independent of the graph's outer
        # reflect node — off by default so a run is unchanged unless enabled.
        reflection_note = None
        if get_settings().enable_reflection:
            reflection_note = await self._reflect_and_regenerate(
                state, prompt, system_prompt
            )

        output_lines = len(state.migrated_code.splitlines())
        details = {
            "input_lines": (state.code_metrics or {}).get("total_lines"),
            "output_lines": output_lines,
            "migration_type": state.migration_type.value,
            "plan_summary": state.inline_plan,
        }
        if reflection_note:
            details["reflection"] = reflection_note

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

            # On-demand verification: the grounding check is corpus-relative, so
            # it flags any import the retrieved context happens not to mention —
            # including perfectly real APIs the corpus simply lacks. Before
            # spending a re-retrieval loop on them, ask the official docs whether
            # they exist. Confirmed imports are dropped from the flag list; only
            # the ones no authoritative source knows about stay suspect.
            if unverified:
                confirmed = await self._verify_imports(unverified, state)
                if confirmed:
                    details["confirmed_imports"] = confirmed
                    unverified = [i for i in unverified if i not in confirmed]
                    grounding["unverified_imports"] = unverified

            if unverified:
                log.warning(
                    "Ungrounded imports in migrated code: %s", ", ".join(unverified)
                )
                summary += f" · ⚠ {len(unverified)} ungrounded import(s)"

                # Adaptive RAG (feedback bus): when enough imports look invented
                # and the re-retrieval budget remains, enqueue them as targeted
                # retrieval queries. migrate_condition then loops the graph back
                # through retrieve -> plan -> migrate with grounded examples for
                # exactly these imports, instead of silently accepting them.
                settings = get_settings()
                if (
                    len(unverified) >= settings.rag_grounding_reretrieval_threshold
                    and state.reretrieval_count < settings.max_reretrievals
                ):
                    state.retrieval_requests = list(unverified)
                    details["reretrieval_requested"] = True
                    summary += " · requesting re-retrieval"

        if self._tool_calls:
            details["tool_calls"] = self.tool_call_log()

        return AgentResult(success=True, summary=summary, details=details)

    async def _reflect_and_regenerate(
        self, state: MigrationState, prompt: str, system_prompt: str
    ) -> dict:
        """Critique the migrated code; regenerate once when confidence is low.

        Delegates the critique to the code-specialized :class:`CriticAgent` (which
        also scans for stub markers), then, if the verdict is a low-confidence
        non-pass with actionable feedback, regenerates the code once with that
        feedback folded into the prompt. Mutates ``state.migrated_code`` in place
        and returns a note for the agent's report. Never raises.
        """
        from agents.critic_agent import CriticAgent

        critic = CriticAgent(self.llm, self.config)
        reflection = await critic.critique(state.migrated_code, state=state)
        note = {
            "confidence": reflection.confidence,
            "recommendation": reflection.recommendation,
            "feedback": reflection.feedback,
        }

        settings = get_settings()
        should_regen = (
            not reflection.passed
            and reflection.confidence < settings.reflection_min_confidence
            and bool(reflection.feedback)
        )
        if not should_regen:
            return note

        refine_prompt = (
            f"{prompt}\n\n---\n\nYOUR PREVIOUS ATTEMPT was reviewed and had these "
            f"problems:\n{reflection.feedback}\n\nRegenerate the migrated code so it "
            "addresses every point above. Return the same JSON object."
        )
        try:
            parsed = await self._generate(refine_prompt, system_prompt)
        except Exception as exc:  # noqa: BLE001 — regeneration is optional
            log.warning("Reflection-driven regeneration failed: %s", exc)
            return note

        if parsed.migrated_code.strip():
            state.migrated_code = parsed.migrated_code.strip()
            if parsed.plan_summary.strip():
                state.inline_plan = parsed.plan_summary.strip()
            note["regenerated"] = True
        return note

    async def _verify_imports(
        self, unverified: list[str], state: MigrationState
    ) -> list[str]:
        """Check flagged imports against official docs; return the ones that exist.

        Note this runs *after* generation, not during it. A single
        ``/api/generate`` completion is one indivisible call — there is no point
        mid-stream at which the model can pause, consult a tool, and resume — so
        "verify APIs the model just wrote" is the achievable form of the idea,
        and it catches the same hallucinated imports a mid-generation check would.

        Bounded and best-effort: each import costs a network round trip, so only
        the first few are checked, and an unavailable search leaves the flags
        exactly as the grounding check set them.
        """
        if "web_search" not in self.tools:
            return []

        settings = get_settings()
        confirmed: list[str] = []
        for name in unverified[: settings.migrator_max_import_checks]:
            result = await self._call_tool(
                "web_search",
                query=f"{name} module documentation",
                language=state.target_language,
            )
            if not result.success:
                # Search is down or rate-limited: stop rather than retry per
                # import, and leave the remaining flags untouched.
                log.info("Import verification unavailable: %s", result.error)
                break
            results = (result.data or {}).get("results") or []
            # The search is already restricted to official documentation domains,
            # so any hit mentioning the import is authoritative evidence it exists.
            if any(
                name.lower() in (hit.get("title", "") + hit.get("snippet", "")).lower()
                for hit in results
            ):
                confirmed.append(name)
                log.info("Import %r confirmed against official docs", name)
        return confirmed

    async def _generate(self, prompt: str, system_prompt: str) -> MigrationOutput:
        """Produce the migration, streaming tokens when a client is listening.

        Two paths, because **structured output and token streaming are mutually
        exclusive**: ``with_structured_output`` only resolves once the whole
        response has been decoded and validated, so there is nothing to emit
        along the way.

        Not streaming
            Native structured output. The provider constrains decoding to
            ``MigrationOutput``, so malformed JSON is not a reachable state.
        Streaming
            Raw JSON-mode streaming, with :class:`MigratedCodeStreamer`
            unwrapping ``migrated_code`` on the fly so the browser sees clean
            code rather than an escaped JSON string, then one validation pass at
            the end. This path can still see imperfect output, which is why the
            salvage in ``llm.structured`` still exists.
        """
        if self.stream_callback:
            return await self._generate_streaming(prompt, system_prompt)

        result = await self._call_structured(
            MigrationOutput, prompt, system_prompt, role="main"
        )
        if result is not None and result.migrated_code.strip():
            return result

        # Nothing usable came back under the schema. Ask once in plain mode and
        # salvage: a model that ignores the schema and simply emits the code is
        # still giving us a working migration, and throwing that away to report
        # a format error would be the wrong trade.
        log.warning("Structured migration produced nothing; retrying unstructured")
        raw = await self.llm.call_llm(prompt, system_prompt)
        return self._salvage(raw)

    async def _generate_streaming(
        self, prompt: str, system_prompt: str
    ) -> MigrationOutput:
        """Stream JSON, emitting decoded code deltas as they arrive."""
        raw_output = ""
        streamer = MigratedCodeStreamer()
        async for token in self.llm.stream_llm(prompt, system_prompt, fmt="json"):
            raw_output += token
            code_delta = streamer.feed(token)
            if code_delta:
                await self.stream_callback(code_delta)
        # The streamer's incremental view survives a response truncated before
        # its closing brace — which is precisely when parsing fails — so it is
        # the better salvage source here than the raw text.
        return self._salvage(raw_output, streamed=streamer.value())

    def _salvage(self, raw: str, streamed: str = "") -> MigrationOutput:
        """Turn a response that isn't schema-shaped into a usable result."""
        parsed = coerce_or_none(MigrationOutput, raw)
        if parsed is not None and parsed.migrated_code.strip():
            return parsed

        code = (streamed or self._strip_fences(raw)).strip()
        if not code:
            raise ValueError("LLM returned empty migrated_code")
        return MigrationOutput(
            plan_summary=(
                "The LLM returned unstructured output; CodeMigrateAI extracted "
                "the migrated code directly."
            ),
            migrated_code=code,
        )

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
