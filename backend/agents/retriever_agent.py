"""RetrieverAgent — grounds the migration in reference material.

Two modes, and the difference is who formulates the query.

**Heuristic (default).** One pass: ``RAGPipeline.enrich_prompt`` mines the source
for imports/calls/types, builds a query, retrieves, and injects whatever comes
back. Deterministic, offline-safe, one embedding call — but the query is a bag of
symbols, and if it retrieves nothing useful the agent has no recourse.

**Tool loop** (``settings.retriever_tool_loop``, on by default). The model sees
the retrieval tools, emits native tool calls against their schemas, reads the
results, and searches again with a better query when the first answer is thin.
That second attempt informed by the first is the actual value of tool use here,
and the single-pass design cannot express it.

This used to be off by default, because selection was prompt-driven — Ollama's
``/api/generate`` has no ``tools`` parameter, so the catalog went into a prompt
and a 1.3b model was asked to reply with JSON naming its choice. Nonsense
selections were routine. With a tool-calling model the model emits a real,
schema-validated call, so the loop is now the default path.

Every exit still falls back to the heuristic pass — no tools registered, a model
that cannot bind them, an exception, or empty results — so the worst case
remains the old behaviour.
"""

import logging

from config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
        """Let the model drive retrieval through native tool calls.

        A genuine ReAct loop: the model sees the retrieval tools, emits real tool
        calls against their schemas, reads the results back as ``ToolMessage``s,
        and decides whether to search again with a better query. The loop ends
        when it stops asking for tools, when material has been found, or when the
        call budget runs out.

        Every call still goes through ``_call_tool`` rather than a ``ToolNode``,
        for two reasons: it records the call in ``tool_call_log`` — the report is
        the evidence that the model, not a rule, chose this — and it lets the
        results be rendered into reference blocks as they arrive.
        """
        bound = self.bind_tools(_RETRIEVAL_TOOLS)
        if bound is None:
            # No retrieval tools registered, or a client that cannot bind them
            # (unit stubs). The heuristic pass is the fallback, as before.
            log.debug("Retrieval tools unavailable; using heuristic pass")
            return ""

        settings = get_settings()
        messages: list = [
            SystemMessage(
                content=(
                    "You ground code migrations in reference material. Search for "
                    "the specific APIs and constructs in the code — not the "
                    "languages in general. If a search returns nothing, try a "
                    "broader query or a different tool. Stop once you have useful "
                    "material."
                )
            ),
            HumanMessage(content=self._initial_goal(state)),
        ]
        blocks: list[str] = []

        for attempt in range(settings.retriever_max_tool_calls):
            try:
                from llm import providers

                reply = await providers.ainvoke_with_retry(
                    bound, messages, label="Retrieval tool call"
                )
            except Exception as exc:  # noqa: BLE001 — fall back to the heuristic pass
                log.warning("Retrieval tool call failed: %s", exc)
                break
            messages.append(reply)

            tool_calls = getattr(reply, "tool_calls", None) or []
            if not tool_calls:
                log.info("Model requested no tools on attempt %d", attempt + 1)
                break

            for call in tool_calls:
                name = call.get("name", "")
                args = self._normalize_arguments(name, call.get("args") or {}, state)
                result = await self._call_tool(name, **args)
                blocks.extend(self._blocks_from(name, result))
                # The result goes back as a ToolMessage whether it succeeded or
                # not — a failure the model can read is a failure it can route
                # around, which is the whole point of tools never raising.
                messages.append(
                    ToolMessage(
                        content=result.for_model(), tool_call_id=call.get("id", "")
                    )
                )

            if blocks:
                # One good grounding pass is the goal, not exhausting the budget.
                break

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
        """Keep the model's arguments, but own the ones that are facts.

        The model chooses the query and the intent — that is the judgement we
        want from it. The target language and version are not judgement calls:
        the agent already holds them, and a model that guesses a different one
        produces a filter that matches nothing. So those are overwritten rather
        than merged, and everything else the model supplied is passed through.
        """
        normalized = dict(arguments)
        normalized["query"] = str(arguments.get("query") or "").strip()
        if name in ("vector_db", "semantic_search"):
            normalized["target_language"] = state.target_language
            normalized["target_version"] = state.target_version
        elif name == "web_search":
            normalized["language"] = state.target_language
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
