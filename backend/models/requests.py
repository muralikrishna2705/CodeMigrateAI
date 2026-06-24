from typing import Optional

from config import get_settings
from pydantic import BaseModel, Field, field_validator


class MigrateRequest(BaseModel):
    source_code: str = Field(..., min_length=1)
    source_language: str = Field(..., min_length=1)
    source_version: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    target_version: str = Field(..., min_length=1)

    @field_validator("source_code")
    @classmethod
    def enforce_size_limit(cls, v: str) -> str:
        settings = get_settings()
        if len(v) > settings.max_code_chars:
            raise ValueError(
                f"Code too large ({len(v)}/{settings.max_code_chars} chars)"
            )
        return v


class MigrateResponse(BaseModel):
    success: bool
    migrated_code: str
    inline_plan: str = ""
    migration_type: str
    source_language: str
    source_version: str
    target_language: str
    target_version: str
    reports: list[dict]
    errors: list[str]
    agents_completed: list[str]
    validation_result: Optional[dict] = None
