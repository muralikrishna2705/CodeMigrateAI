FROM python:3.12-slim

LABEL maintainer="MTech Final Year Project — CodeMigrateAI"
LABEL description="FastAPI backend with two-agent LLM-first migration pipeline"

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Non-root user for security.
#
# The two volume mount points are created here, before the chown, on purpose:
# Docker seeds a named volume from the image path it covers — permissions and
# ownership included. If the directory does not exist in the image, Docker
# creates the mount point as root and the non-root process cannot write its
# index or its databases.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data /app/rag/chroma_db \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# SINGLE worker, deliberately. This process owns three single-writer resources,
# and a second worker corrupts or doubles all of them:
#
#   * the embedded Chroma index (rag/chroma_db) and the SQLite memory and
#     checkpoint databases — one writer each, not multi-process safe;
#   * the rate limiters in llm/providers.py, which are per-process, so N workers
#     burst to N times the configured rate against a quota that is per-project;
#   * startup RAG ingestion, which every worker would run in full — duplicating
#     every doc fetch and every (metered) embedding call.
#
# Concurrency here is I/O-bound on the LLM API, which one event loop already
# handles; scale with replicas behind a proxy, not with workers sharing a
# filesystem.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
