"""Multi-hop retrieval over a decomposed query graph.

Some migration questions are compound — "port this Flask app's auth and its async
DB layer to FastAPI" bundles several independent retrieval needs. One embedding of
the whole thing retrieves a muddled average of all of them. Multi-hop decomposes
the query into sub-questions and retrieves for each, so every facet gets its own
sharp probe.

The decomposition is walked as an explicit **directed graph** (BFS): sub-questions
are the first layer of nodes; when a node retrieves thin results and depth remains,
the model proposes follow-up questions that become its children. A ``visited`` set
(keyed by normalized question text) prevents re-retrieving the same node, and two
bounds — ``rag_multi_hop_max_subqueries`` (total nodes expanded) and
``rag_multi_hop_max_depth`` (graph depth) — keep the traversal finite. Results
from every visited node are fused with ``merge_hits``.

With no LLM, decomposition yields nothing and this collapses to a single-hop pass.
"""

import logging
from collections import deque
from dataclasses import dataclass

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy, merge_hits

log = logging.getLogger("CodeMigrateAI.RAG.MultiHop")


@dataclass
class QueryNode:
    """A node in the query graph: a sub-question and its BFS depth."""

    question: str
    depth: int


class MultiHopStrategy(RetrievalStrategy):
    name = "multi_hop"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        max_nodes = getattr(settings, "rag_multi_hop_max_subqueries", 4)
        max_depth = getattr(settings, "rag_multi_hop_max_depth", 2)

        sub_questions = await self._decompose(request, max_nodes) if self.has_llm else []
        # No decomposition (no LLM, or a simple query) → seed the graph with the
        # original query, which makes this a plain single-hop pass.
        seed = sub_questions or [request.query]

        visited: set[str] = set()
        queue: deque[QueryNode] = deque(QueryNode(q, 1) for q in seed)
        hitlists: list[list[tuple]] = []

        while queue and len(visited) < max_nodes:
            node = queue.popleft()
            key = node.question.strip().lower()
            if not key or key in visited:
                continue
            visited.add(key)

            hits = await self._run_query(node.question, request)
            hitlists.append(hits)

            # Expand only when this branch is underserved and depth remains — that
            # is what makes it a graph search rather than a flat fan-out, while the
            # thinness gate and node budget keep LLM calls bounded.
            if node.depth < max_depth and len(hits) < settings.rag_top_k:
                for follow_up in await self._follow_ups(node.question, request):
                    if follow_up.strip().lower() not in visited:
                        queue.append(QueryNode(follow_up, node.depth + 1))

        # Always fold in the original top-level query so a poor decomposition can
        # never lose the user's actual intent.
        if request.query.strip().lower() not in visited:
            hitlists.append(await self._run_query(request.query, request))

        return merge_hits(hitlists, settings.rag_top_k)

    async def _decompose(self, request: RetrievalRequest, limit: int) -> list[str]:
        """Break a compound query into independent sub-questions, one per line."""
        prompt = (
            f"Break this migration research question into up to {limit} independent, "
            "self-contained sub-questions that can each be looked up separately. "
            "If it is already a single question, return it unchanged. One question "
            "per line, no numbering.\n\n"
            f"QUESTION: {request.query}"
        )
        return self._parse_lines(
            await self._ask(prompt, system="Output one sub-question per line."),
            limit,
        )

    async def _follow_ups(self, question: str, request: RetrievalRequest) -> list[str]:
        """Propose narrower follow-ups when a sub-question retrieved little."""
        prompt = (
            f"The search '{question}' (migrating to {request.target_language}) "
            "returned little. Suggest up to 2 narrower or reworded follow-up "
            "questions that might retrieve relevant documentation. One per line, "
            "no numbering."
        )
        return self._parse_lines(
            await self._ask(prompt, system="Output one question per line."), 2
        )

    @staticmethod
    def _parse_lines(raw: str, limit: int) -> list[str]:
        out: list[str] = []
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-*0123456789.) \t").strip()
            if cleaned:
                out.append(cleaned)
        return out[:limit]
