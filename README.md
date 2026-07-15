# CodeMigrateAI

CodeMigrateAI is an AI-driven code migration platform — an MTech Final Year Project — built around a multi-agent LLM pipeline. It converts source code between languages (e.g. Java 8 → Python 3.12) or upgrades versions within the same language (e.g. Python 2.7 → Python 3.12).

## Pipeline Architecture

```text
Request → Cache → [Runtime Agents] → RetrieverAgent (RAG) → LangGraph Workflow → Optional Validator → Response

LangGraph Workflow:
  analyze → (if high complexity) deep_analyze → plan → migrate → validate → (on failure) fix → migrate (retry loop, max 2 retries)
```

The pipeline uses **LangGraph** (`StateGraph`) to orchestrate the migration workflow with retry logic, and **LangChain / ChromaDB** for Retrieval-Augmented Generation.

## Supported Languages

| Language | IDs / aliases | Sample Versions |
| --- | --- | --- |
| Java | `java` | 7, 8, 11, 17, 21 |
| Python | `python`, `py`, `python3` | 2.7, 3.8, 3.10, 3.12 |
| JavaScript | `javascript`, `js`, `node` | ES5, ES6, ES2020, ES2022 |
| TypeScript | `typescript`, `ts` | 3.x, 4.x, 5.x |
| C# | `csharp`, `c#`, `cs` | 6, 8, 10, 12 |
| Go | `go`, `golang` | 1.18, 1.20, 1.22 |
| Kotlin | `kotlin` | 1.7, 1.9, 2.0 |
| Rust | `rust` | 1.70, 1.80 |
| C++ | `cpp`, `c++` | 14, 17, 20, 23 |

## Key Features

### LangGraph Migration Workflow
A compiled `StateGraph` with nodes for analysis, deep analysis, planning, migration, validation, and fixing. On validation failure, a retry loop (`validate → fix → migrate`) executes up to `max_retries` (default 2).

### RAG Pipeline (Phase 2)
- **Ingestion**: Embeds reference documentation into a ChromaDB vector store at startup.
- **Code-aware query construction**: Extracts imports, API calls, and type names from source code.
- **Hybrid search**: Combines dense vector embeddings with keyword retrieval fused by Reciprocal Rank Fusion (RRF).
- **Version-aware filtering**: Phased retrieval ladder (exact version → language-only → unfiltered) with metadata-weighted ranking.
- **Code grounding check**: Post-hoc analysis flags library imports not grounded by source, RAG context, or target stdlib.

### Streaming via SSE
Server-Sent Events deliver token-by-token code output to the frontend. The `MigratedCodeStreamer` unwraps the JSON wrapper on the fly, showing only clean migrated code while maintaining JSON structure internally for parsing.

### Anti-Hallucination Measures
- `UngroundedNotice` prepended when RAG finds no reference examples.
- Version constraints in prompts fence off APIs newer than the target version.
- LLM output parsing with multiple fallback strategies (JSON extraction, preamble stripping, markdown fence extraction, truncated JSON repair).

### Fast Model Routing
Analysis and planning tasks use a lighter model (e.g. `llama3.2:1b`) while code generation uses the main model (`deepseek-coder:1.3b`). Missing models are auto-pulled on startup.

### Caching
Two-tier cache: **Redis** (primary, with TTL) and **local LRU cache** (fallback). Cache keys use SHA-256 hashes of source code, language/version IDs, migration type, and analyzer context.

### Validator Service
A separate FastAPI microservice (`validator_service/`) with per-language syntax validators using real toolchains when available, and graceful degradation otherwise.

### CI/CD Integration
- **GitHub Actions workflow** (`cicd/github-actions.yml`): test → build/push Docker images to GHCR → SSH deploy.
- **CodeMigrate Pipeline** (`.github/workflows/codemigrate-pipeline.yml`): On PRs, scans changed files, runs migration via a GitHub Action, commits to a `codemigrate/<stem>` branch, and opens a migration PR.

## Backend

Key modules:

- `backend/agents/analyzer_agent.py` — static code metrics and LLM-based semantic analysis.
- `backend/agents/migrator_agent.py` — single LLM call with strict JSON parsing.
- `backend/graph/migration_graph.py` — LangGraph workflow definition.
- `backend/rag/retriever_agent.py` — RAG retrieval and grounding checks.
- `backend/rag/ingestion.py` — ChromaDB ingestion pipeline.
- `backend/llm/prompt_composer.py` — cached migration prompt builder.
- `backend/llm/language_profiles/` — shared language guidance and few-shot examples.
- `backend/clients/llm_client.py` — Ollama HTTP client (sync and streaming).
- `backend/clients/validator_client.py` — optional validator service caller.
- `backend/streaming.py` — SSE streaming and `MigratedCodeStreamer`.

## Environment Variables

```text
OLLAMA_URL=http://host.docker.internal:11434
LLM_MODEL=deepseek-coder:1.3b
FAST_LLM_MODEL=llama3.2:1b
EMBEDDING_MODEL=nomic-embed-text
REDIS_URL=redis://redis:6379/0
CHROMA_URL=http://chromadb:8000
VALIDATOR_URL=http://validator:8000
ENABLE_VALIDATION=false
ENABLE_RAG=false
ENABLE_GROUNDING_CHECK=false
ENABLE_WEB_DOCS=false
OLLAMA_AUTO_PULL=true
```

`ENABLE_VALIDATION=false` keeps the hot migration path lowest-latency. Docker Compose enables validation because it starts the validator service.

## Validator Service

`validator_service/` is a separate FastAPI service with:

- `GET /health`
- `POST /validate`

Returns:

```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

Uses real syntax tools when available; returns a structured warning if a toolchain is missing.

## Run Locally

Backend:

```powershell
cd backend
uvicorn main:app --reload
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```

Docker:

```powershell
docker compose -f cicd/docker-compose.yml up --build
```

The frontend is served on `http://localhost:3000`; the backend API is on `http://localhost:8000`; the validator service is exposed on `http://localhost:8001`.

## Verification

```powershell
python -m pytest backend/tests/ -q
python -m compileall -q backend validator_service
npm.cmd run build
```
