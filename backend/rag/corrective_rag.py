"""Corrective RAG (CRAG).

Passive RAG injects whatever it retrieved, even when the retrieval was bad —
which is exactly how off-topic context leads a model to hallucinate a confident,
wrong migration. CRAG adds a correction loop: retrieve, then have the LLM *grade*
whether what came back actually answers the question, and act on that judgment:

  * **correct**   → use the hits as-is.
  * **ambiguous** → results are relevant but thin; refine the query and retrieve
                    again, then fuse both passes.
  * **incorrect** → results miss; refine + re-retrieve, and (when enabled) fall
                    back to live official-docs web search for material the local
                    corpus simply doesn't have.

Every step degrades safely. With no LLM the grade comes from a score/coverage
heuristic; the web leg is opt-in (``rag_crag_web_fallback``) and off by default,
matching the rest of the repo's network policy; and if correction yields nothing
the original hits are returned. So CRAG is never worse than single-hop.
"""

import logging

from langchain_core.documents import Document

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy, merge_hits

log = logging.getLogger("CodeMigrateAI.RAG.CRAG")

_GRADES = ("correct", "ambiguous", "incorrect")
# Snippets shown to the grader are truncated — the grader judges topicality, and a
# few hundred chars per hit is plenty while keeping the prompt inside the context.
_GRADE_SNIPPET_CHARS = 400
# Score assigned to a web result so it ranks alongside corpus hits during fusion:
# above rag_min_score (it's authoritative), below a strong exact corpus match.
_WEB_HIT_SCORE = 0.75


class CorrectiveRAGStrategy(RetrievalStrategy):
    name = "corrective"

    def __init__(self, pipeline, llm=None, web_tool=None) -> None:
        super().__init__(pipeline, llm)
        self._web_tool = web_tool

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        hits = await self._run_query(request.query, request)

        grade = await self._grade(request, hits, settings)
        log.info("CRAG graded initial retrieval: %s (%d hits)", grade, len(hits))
        if grade == "correct":
            return hits

        # ambiguous / incorrect → refine and retrieve again.
        refined = await self._refine_query(request, len(hits))
        extra = await self._run_query(refined, request) if refined else []
        combined = merge_hits([hits, extra], settings.rag_top_k)

        # Only reach the live web when the corpus genuinely missed AND the operator
        # opted into network access.
        if grade == "incorrect" and getattr(settings, "rag_crag_web_fallback", False):
            web_hits = await self._web_search(request)
            if web_hits:
                combined = merge_hits([combined, web_hits], settings.rag_top_k)

        return combined or hits

    # --- Grading ----------------------------------------------------------

    async def _grade(
        self, request: RetrievalRequest, hits: list[tuple], settings
    ) -> str:
        """Grade retrieval as correct / ambiguous / incorrect."""
        threshold = getattr(settings, "rag_crag_relevance_threshold", 0.5)
        if not self.has_llm or not hits:
            return self._heuristic_grade(hits, threshold, settings)

        snippets = "\n".join(
            f"{i + 1}. {doc.page_content[:_GRADE_SNIPPET_CHARS]}"
            for i, (doc, _) in enumerate(hits)
        )
        prompt = (
            "Judge whether the retrieved snippets are sufficient to answer the "
            "migration question. Respond as JSON: "
            '{"relevance": "correct" | "ambiguous" | "incorrect"}. '
            "correct = they directly answer it; ambiguous = partially relevant, "
            "more retrieval would help; incorrect = they do not answer it.\n\n"
            f"QUESTION: {request.query}\n\nSNIPPETS:\n{snippets}"
        )
        data = await self._ask_json(
            prompt, system="Output only the JSON verdict."
        )
        grade = str(data.get("relevance", "")).strip().lower()
        if grade in _GRADES:
            return grade
        return self._heuristic_grade(hits, threshold, settings)

    @staticmethod
    def _heuristic_grade(hits: list[tuple], threshold: float, settings) -> str:
        """Grade without an LLM: score + coverage.

        Hits already clear ``rag_min_score``, so the useful offline signal is
        *coverage* — a full set of relevant hits is correct; a relevant but sparse
        set is ambiguous (worth another pass); nothing is incorrect.
        """
        if not hits:
            return "incorrect"
        top = max(score for _, score in hits)
        if top < threshold:
            return "incorrect"
        if len(hits) >= getattr(settings, "rag_top_k", 4):
            return "correct"
        return "ambiguous"

    # --- Correction -------------------------------------------------------

    async def _refine_query(self, request: RetrievalRequest, n_hits: int) -> str:
        """Reformulate the query after weak retrieval; original query on failure."""
        if not self.has_llm:
            return request.query
        prompt = (
            f"The search '{request.query}' for migrating to "
            f"{request.target_language} returned {n_hits} weak result(s). Rewrite "
            "it into one improved search query that would retrieve better "
            "documentation. Output only the query."
        )
        refined = (await self._ask(prompt, system="Output only the query.")).strip()
        return refined or request.query

    async def _web_search(self, request: RetrievalRequest) -> list[tuple]:
        """Search official docs on the live web; results as pseudo-hits."""
        tool = self._get_web_tool()
        if tool is None:
            return []
        try:
            result = await tool(
                query=request.query,
                language=request.target_language,
                fetch_content=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("CRAG web search failed: %s", exc)
            return []
        if not getattr(result, "success", False):
            return []

        hits: list[tuple] = []
        for item in (result.data or {}).get("results", []):
            snippet = item.get("snippet", "")
            title = item.get("title", "")
            content = f"{title}\n{snippet}".strip()
            if not content:
                continue
            doc = Document(
                page_content=content,
                metadata={
                    "language": request.target_language or "unknown",
                    "version": request.target_version or "",
                    "doc_type": "reference",
                    "is_official": True,
                    "source": item.get("url", ""),
                },
            )
            hits.append((doc, _WEB_HIT_SCORE))
        return hits

    def _get_web_tool(self):
        """Return the web tool, lazily constructing the default one if needed."""
        if self._web_tool is not None:
            return self._web_tool
        try:
            from agents.tools.web_search import WebSearchTool

            self._web_tool = WebSearchTool()
        except Exception as exc:  # noqa: BLE001 — web leg is optional
            log.warning("CRAG could not construct web tool: %s", exc)
            self._web_tool = None
        return self._web_tool
