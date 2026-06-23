FROM python:3.12-slim

LABEL maintainer="MTech Final Year Project — CodeMigrateAI"
LABEL description="FastAPI backend with DeepSeek-Coder-V2 agent pipeline"

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uvicorn with 2 workers for modest concurrency
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]