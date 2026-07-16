"""RetrieverAgent — grounds the migration in reference material.

Two modes, and the difference is who formulates the query.

**Heuristic (default).** One pass: ``RAGPipeline.enrich_prompt`` mines the source
for imports/calls/types, builds a query, retrieves, and injects whatever comes
back. Deterministic, offline-safe, one embedding call — but the query is a bag of
symbols, and if it retrieves nothing useful the agent has no recourse.

**Tool loop** (``settings.retriever_tool_loop``). The agent asks the LLM which
retrieval tool to call and what to ask it, inspects what comes back, and — if the
answer is thin — asks again with a different query. That is the actual value of
tool use here: a second attempt informed by the first, which the single-pass
design cannot express.

The loop is off by default and deliberately so. Selection is prompt-driven
because Ollama's ``/api/generate`` has no native tool calling, and
``deepseek-coder:1.3b`` is not tool-call trained, so a failed or nonsense
selection is routine. Every exit from the loop — no selection, no tools, empty
results, an exception — falls back to the heuristic pass, so the worst case is
the old behaviour plus one wasted LLM call.
"""

import logging

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RetrieverAgent")

# Tools the loop may choose from. A short catalog is not a limitation but a
# requirement: selection accuracy on a small model collapses as the option list
# grows, so the agent is offered only the tools that retrieve reference material.
_RETRIEVAL_TOOLS = ("vector_db", "semantic_search", "web_search")

# MigratorAgent tests for this literal heading before using the context, and
# concatenates rag_context directly onto its prompt, so tool-built context must
# carry the same heading and the same trailing separator enrich_prompt produces.
_CONTEXT_HEADING = (
    "## Reference Examples\nHere are relevant code patterns from the "
    "target language:\n"
)
_CONTEXT_SEPARATOR = "\n\n---\n\n"


class RetrieverAgent(BaseAgent):
    name = "RetrieverAgent"
    requires_llm = False
    needs = ("rag_pipeline", "tools")

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self._rag_pipeline = config.get("rag_pipeline") if config else None

    async def run(self, state: MigrationState) -> AgentResult:
        settings = get_settings()

        context = ""
        mode = "heuristic"
        if settings.retriever_tool_loop:
            context = await self._tool_loop(state)
            if context:
                mode = "tool-loop"

        if not context:
            if not self._rag_pipeline:
                return AgentResult(
                    success=True, summary="RAG pipeline not available, skipping"
                )
            context = await self._heuristic_pass(state, settings)

        state.rag_context = context

        summary = (
            f"Retrieved RAG context ({mode})"
            if state.rag_context
            else "No relevant context found"
        )
        if state.retrieval_requests:
            summary += f" · re-retrieval for {len(state.retrieval_requests)} import(s)"
        return AgentResult(
            success=True,
            summary=summary,
            details={
                "mode": mode,
                "context_length": len(state.rag_context),
                "reretrieval_terms": list(state.retrieval_requests),
                "tool_calls": self.tool_call_log(),
            },
        )

    # --- Heuristic mode ---------------------------------------------------

    async def _heuristic_pass(self, state: MigrationState, settings) -> str:
        """The original single-pass retrieval; also the tool loop's fallback."""
        # Feedback bus: targeted queries enqueued by an upstream agent (e.g. the
        # MigratorAgent's ungrounded imports). Optionally add LLM-expanded terms.
        extra_terms = list(state.retrieval_requests)
        if settings.rag_query_expansion:
            extra_terms += await self._expand_query_terms(state)

        enriched = await self._rag_pipeline.enrich_prompt(
            source_language=state.source_language,
            target_language=state.target_language,
            source_code=state.source_code,
            base_prompt="",  # Consumed by MigratorAgent via state.rag_context
            target_version=state.target_version,
            code_metrics=state.code_metrics,
            extra_terms=extra_terms,
        )
        return enriched if "Reference Examples" in enriched else ""

    async def _expand_query_terms(self, state: MigrationState) -> list[str]:
        """Reformulate code signals into targeted query terms via the fast model.

        Best-effort: any failure (stub LLM, model down, bad output) returns an
        empty list so retrieval falls back to the concatenated-signal query.
        """
        call_llm = getattr(self.llm, "call_llm", None)
        if call_llm is None:
            return []
        try:
            prompt = (
                f"A developer is migrating {state.source_language} "
                f"{state.source_version} code to {state.target_language} "
                f"{state.target_version}. Based on the code below, list up to 6 "
                "short search keywords (comma-separated, no sentences) naming the "
                "libraries, APIs, and language constructs whose migration matters "
                "most.\n\nCODE:\n"
                f"{state.source_code[: get_settings().max_llm_code_chars]}"
            )
            raw = await call_llm(
                prompt,
                system_prompt="Output only a comma-separated keyword list.",
                **self._fast_model_kwargs(),
            )
            terms = [t.strip() for t in raw.replace("\n", ",").split(",")]
            return [t for t in terms if t and len(t) <= 40][:6]
        except Exception as exc:  # noqa: BLE001 — expansion is strictly optional
            log.warning("Query expansion failed: %s", exc)
            return []

    # --- Tool loop mode ---------------------------------------------------

    async def _tool_loop(self, state: MigrationState) -> str:
        """Iteratively pick a retrieval tool, call it, and refine. Returns context."""
        tools = self.tools.subset(_RETRIEVAL_TOOLS)
        if not tools:
            log.debug("No retrieval tools registered; using heuristic pass")
            return ""

        settings = get_settings()
        blocks: list[str] = []
        tried: set[tuple[str, str]] = set()
        goal = self._initial_goal(state)

        for attempt in range(settings.retriever_max_tool_calls):
            selection = await self._select_tool(goal, tools=tools)
            if not selection:
                log.info("No tool selected on attempt %d; stopping loop", attempt + 1)
                break

            name, arguments = selection
            arguments = self._normalize_arguments(name, arguments, state)
            signature = (name, str(arguments.get("query", "")).strip().lower())
            if signature in tried:
                # The model re-proposed a query we already ran. Without this the
                # loop would burn its whole budget re-asking the same question.
                log.info("Tool loop repeated %s(%r); stopping", *signature)
                break
            tried.add(signature)

            result = await self._call_tool(name, **arguments)
            if not result.success:
                goal = (
                    f"{self._initial_goal(state)}\nThe {name} tool failed "
                    f"({result.error}). Try a different tool."
                )
                continue

            new_blocks = self._blocks_from(name, result)
            blocks.extend(new_blocks)
            if new_blocks:
                # Retrieval found material; one good grounding pass is the goal,
                # not exhausting the call budget.
                break

            # Retrieved nothing: the query was wrong, so say so explicitly rather
            # than re-issuing the same goal and inviting the same query back.
            goal = (
                f"{self._initial_goal(state)}\nA {name} search for "
                f"{arguments.get('query')!r} returned no results. Formulate a "
                "different, broader query, or choose another tool."
            )

        if not blocks:
            return ""
        return _CONTEXT_HEADING + "\n\n" + "\n\n".join(blocks) + _CONTEXT_SEPARATOR

    def _initial_goal(self, state: MigrationState) -> str:
        """Describe what to retrieve, seeded with the sharpest signals we have."""
        parts = [
            f"Find reference examples for migrating {state.source_language} "
            f"{state.source_version} code to {state.target_language} "
            f"{state.target_version}."
        ]
        if state.retrieval_requests:
            # Re-retrieval: the migrator produced imports it could not ground.
            # These are the highest-value thing to look up, so lead with them.
            parts.append(
                "Focus on these specific imports the migration could not verify: "
                + ", ".join(state.retrieval_requests)
            )
        constructs = (state.code_metrics or {}).get("key_constructs") or []
        if constructs:
            parts.append("Key constructs in the code: " + ", ".join(map(str, constructs[:6])))
        excerpt = state.source_code.strip()[: get_settings().rag_query_code_chars]
        if excerpt:
            parts.append(f"CODE:\n{excerpt}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_arguments(name: str, arguments: dict, state: MigrationState) -> dict:
        """Fill in the arguments the model reliably omits.

        A small model typically returns just ``{"query": "..."}``. Rather than
        rejecting the selection, supply the target language/version from state —
        they are facts the agent already holds and the model has no business
        guessing. Filters get dropped if the model invents a different language,
        since an off-target filter retrieves nothing.
        """
        normalized = {"query": str(arguments.get("query") or "").strip()}
        if name in ("vector_db", "semantic_search"):
            normalized["target_language"] = state.target_language
            normalized["target_version"] = state.target_version
        elif name == "web_search":
            normalized["language"] = state.target_language
            normalized["fetch_content"] = False
        return normalized

    @staticmethod
    def _blocks_from(name: str, result) -> list[str]:
        """Render a tool's payload into prompt-ready reference blocks."""
        data = result.data or {}
        blocks: list[str] = []

        if name == "vector_db":
            for hit in data.get("hits", []):
                lang = hit.get("language", "unknown")
                label = " · ".join(
                    filter(None, [lang, hit.get("version", ""), hit.get("doc_type", "")])
                )
                blocks.append(
                    f"### {label} (relevance: {hit.get('score', 0):.2f})\n"
                    f"```{lang}\n{hit.get('content', '')}\n```"
                )
        elif name == "semantic_search":
            for precedent in data.get("precedents", []):
                lang = precedent.get("target_language", "")
                blocks.append(
                    f"### Past migration to {lang} {precedent.get('target_version', '')} "
                    f"(similarity: {precedent.get('score', 0):.2f})\n"
                    f"{precedent.get('plan_summary', '')}\n"
                    f"```{lang}\n{precedent.get('migrated_code', '')}\n```"
                )
        elif name == "web_search":
            for hit in data.get("results", []):
                blocks.append(
                    f"### {hit.get('title', 'Official documentation')}\n"
                    f"{hit.get('snippet', '')}\n\nSource: {hit.get('url', '')}"
                )
            if data.get("content"):
                blocks.append(f"### Documentation page\n{data['content']}")

        return blocks
