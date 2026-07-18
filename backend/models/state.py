from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MigrationType(str, Enum):
    UPGRADE_VERSION = "upgrade_version"
    CONVERT_LANGUAGE = "convert_language"


class AgentReport(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    agent: str
    status: str
    summary: str
    details: Optional[dict] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MigrationState(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    source_code: str
    source_language: str
    source_version: str
    target_language: str
    target_version: str

    migration_type: MigrationType = MigrationType.UPGRADE_VERSION
    code_metrics: Optional[dict] = None
    inline_plan: str = ""
    rag_context: str = ""
    migrated_code: str = ""
    validation_result: Optional[dict] = None

    # Adaptive-RAG feedback bus: agents enqueue targeted retrieval queries here
    # (e.g. ungrounded imports the MigratorAgent produced) and the retrieve node
    # consumes them to refine retrieval. reretrieval_count bounds the
    # migrate -> retrieve loop the way retry_count bounds the fix loop.
    retrieval_requests: list[str] = Field(default_factory=list)
    reretrieval_count: int = 0

    # Dynamic routing: the DispatcherAgent writes its decision here (e.g.
    # {"deep_analyze": bool, "sequence": [...]}) and the graph's edge conditions
    # consult it instead of recomputing the branch from complexity.
    route_plan: dict = Field(default_factory=dict)

    # Agent reflection (Dimension 3): the ReflectorAgent (the graph `reflect` node)
    # writes its self-critique here. reflection_score is the model's 0.0-1.0
    # confidence in the current output; reflection_feedback is the actionable
    # critique a regeneration should address; reflection_recommendation is the
    # routing verdict ("pass" | "re-generate" | "gather-more-info") the
    # reflect_condition consults. Defaults are a passing no-op, so a run that never
    # reflects (reflection disabled) routes straight through.
    reflection_score: float = 0.0
    reflection_feedback: str = ""
    reflection_recommendation: str = "pass"

    reports: list[AgentReport] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    agents_done: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    _stream_callback = None

    def record_success(
        self,
        agent: str,
        summary: str,
        details: dict | None = None,
        duration_ms: int = 0,
    ):
        self.reports.append(
            AgentReport(
                agent=agent,
                status="success",
                summary=summary,
                details=details,
                duration_ms=duration_ms,
            )
        )
        self.agents_done.append(agent)

    def record_error(self, agent: str, message: str, duration_ms: int = 0):
        self.reports.append(
            AgentReport(
                agent=agent,
                status="error",
                summary=message,
                duration_ms=duration_ms,
            )
        )
        self.errors.append(f"[{agent}] {message}")
        self.agents_done.append(agent)

    def record_skip(self, agent: str, reason: str):
        self.reports.append(
            AgentReport(agent=agent, status="skipped", summary=reason)
        )

    def snapshot(self) -> "MigrationState":
        return self.model_copy(deep=True)
