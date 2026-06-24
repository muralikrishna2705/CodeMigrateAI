from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from validators import get_validator, supported_languages


class ValidateRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


app = FastAPI(
    title="CodeMigrateAI Validator",
    description="Syntax validation service for migrated code",
    version="1.0.0",
)


@app.get("/health")
async def health():
    languages = supported_languages()
    return {
        "status": "ok",
        "supported_languages": languages,
        "language_count": len(languages),
    }


@app.post("/validate")
async def validate(request: ValidateRequest):
    validator = get_validator(request.language)
    if validator is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported language: {request.language}",
                "supported_languages": supported_languages(),
            },
        )
    result = await validator.validate(request.code, request.version)
    return result.model_dump()
