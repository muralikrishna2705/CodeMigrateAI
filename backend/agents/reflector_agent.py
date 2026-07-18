"""ReflectorAgent — the Reflexion node the migration graph reflects through.

Runs as the ``reflect`` graph node between ``migrate`` and ``validate``. It
evaluates whatever the pipeline just produced and writes a verdict onto the state
— ``reflection_score`` (confidence), ``reflection_feedback`` (what to fix), and
``reflection_recommendation`` (the routing action) — which
``graph.conditions.reflect_condition`` consults to decide whether to send the
output back for regeneration.

This is the *outer* reflection loop. It is model-driven, not rule-based: the
recommendation comes from a critique of the actual output (delegated to the
:class:`~agents.critic_agent.CriticAgent` for code, which is the graph's usual
case), reflecting together with any validation errors already on the state — the
Reflexion idea of learning from the trajectory, not just the latest artifact.
"""

import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ReflectorAgent")


class ReflectorAgent(BaseAgent):
    name = "ReflectorAgent"
    requires_llm = True
    needs = ("tools",)

    async def run(self, state: MigrationState) -> AgentResult:
        if state.migrated_code:
            result = await self._reflect_on_code(state)
            target = "migrated code"
        elif state.inline_plan:
            result = await self._reflect_on_plan(state)
            target = "plan"
        else:
            # Nothing to reflect on — pass through so the graph proceeds.
            state.reflection_recommendation = "pass"
            return AgentResult(
                success=True, summary="Nothing to reflect on", details={"skipped": True}
            )

        state.reflection_score = result.confidence
        state.reflection_feedback = result.feedback
        state.reflection_recommendation = result.recommendation

        verdict = "✓ pass" if result.passed else f"↻ {result.recommendation}"
        return AgentResult(
            success=True,
            summary=(
                f"Reflected on {target}: {verdict} "
                f"(confidence {result.confidence:.2f})"
            ),
            details={
                "confidence": result.confidence,
                "recommendation": result.recommendation,
                "feedback": result.feedback,
                "target": target,
                **(result.details or {}),
            },
        )

    async def _reflect_on_code(self, state: MigrationState):
        """Delegate code critique to the specialized CriticAgent."""
        from agents.critic_agent import CriticAgent

        critic = CriticAgent(self.llm, self.config)
        return await critic.critique(state.migrated_code, state=state)

    async def _reflect_on_plan(self, state: MigrationState):
        """Generic reflection on the migration plan text."""
        from agents.tools.reflection import PLAN_CRITERIA

        return await self.reflect(
            state,
            state.inline_plan,
            criteria=PLAN_CRITERIA,
            stage="migration plan",
        )
