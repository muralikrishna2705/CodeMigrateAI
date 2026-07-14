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

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        settings = get_settings()
        valid_ids = {lang["id"] for lang in settings.supported_languages}
        if v.strip().lower() not in valid_ids:
            raise ValueError(
                f"Unsupported language '{v}'. Must be one of: {', '.join(sorted(valid_ids))}"
            )
        return v.strip().lower()

    @field_validator("source_version", "target_version")
    @classmethod
    def validate_version(cls, v: str, info) -> str:
        settings = get_settings()
        # Determine which language this version belongs to
        field_name = info.field_name
        lang_field = "source_language" if "source" in field_name else "target_language"
        lang_value = info.data.get(lang_field, "")
        for lang in settings.supported_languages:
            if lang["id"] == lang_value:
                if v not in lang["versions"]:
                    raise ValueError(
                        f"Invalid version '{v}' for {lang['name']}. "
                        f"Must be one of: {', '.join(lang['versions'])}"
                    )
                break
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
