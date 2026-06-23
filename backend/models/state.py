from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MigrationType(str, Enum):
    UPGRADE_VERSION = "upgrade_version"
    CONVERT_LANGUAGE = "convert_language"


class PipelineMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"
    VALIDATED = "validated"


class AgentReport(BaseModel):
    agent: str
    status: str
    summary: str
    details: Optional[dict] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MigrationState(BaseModel):
    source_code: str
    source_language: str
    source_version: str
    target_language: str
    target_version: str
    pipeline_mode: PipelineMode = PipelineMode.FAST

    migration_type: MigrationType = MigrationType.UPGRADE_VERSION
    code_metrics: Optional[dict] = None
    migration_plan: Optional[dict] = None
    migrated_code: str = ""
    validation_result: Optional[dict] = None

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
