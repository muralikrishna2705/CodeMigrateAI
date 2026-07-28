# CodeMigrateAI

CodeMigrateAI is an AI-driven code migration platform — an MTech Final Year Project — built around a multi-agent LLM pipeline. It converts source code between languages (e.g. Java 8 → Python 3.12) or upgrades versions within the same language (e.g. Python 2.7 → Python 3.12).

## Quick start

```bash
echo "GOOGLE_API_KEY=your-key-here" > .env   # repo root; get one at aistudio.google.com/apikey
pip install -r backend/requirements.txt
python backend/scripts/build_index.py        # build the RAG index once
cd backend && uvicorn main:app --reload
```

`.env` lives at the **repo root** and is git-ignored. `config.py` resolves it
from its own location rather than the working directory, so every entry point —
uvicorn, the index builder, pytest — reads the same file no matter where you
launch it from.

The default model provider is **Gemini**, because native tool calling is what
the agentic paths require. `LLM_PROVIDER=ollama` switches to local inference,
but Ollama's `/api/generate` has no `tools` parameter, so the tool loop and
structured routing fall back to their deterministic paths there.

### Measured latency

Real Java 8 → Python 3.12 migrations against `gemini-3.5-flash` +
`gemini-3.1-flash-lite`, measured end to end:

| Configuration | Wall clock | Model calls |
| --- | --- | --- |
| Shipped defaults, RAG on, 49-chunk index | **98 s** | 14 (13 fast, 1 main) |
| Decision flags off, RAG off | **38 s** | 5 (4 fast, 1 main) |

**The model is the ceiling, not the rate limiter** — which corrects the
assumption this project was originally tuned around. Individual calls take
5–10 s because the prompts are large, so at `LLM_REQUESTS_PER_SECOND=0.16` the
limiter accounted for 1.2 s of a 38 s run. Raising it does not speed up a single
migration; it matters for concurrent branches and concurrent users. The levers
that move this number are fewer calls and shorter prompts.

For a fast demo, set `DISPATCHER_LLM_ROUTING=false`,
`ORCHESTRATOR_LLM_PLANNING=false`, and `ENABLE_RAG=false` — that is the 38 s
row. It also makes the run fully deterministic, at the cost of the model no
longer choosing the route.

`thinking_budget=0` on the fast role is real but modest — about 15% on a short
routing call, less on a long one.

## Pipeline architecture

```text
Request → Cache → LangGraph workflow → Optional validator service → Response
```

A compiled `StateGraph` with 13 nodes and three budget-bounded cycles:

```text
analyze → dispatch → orchestrate ═(Send × n)═→ branch ═══════╗
                                  ├─(deep)─→ deep_analyze ──╮║
                                  └─────────→ retrieve ─────┴╩→ plan → migrate
                                                                            │
     ┌──────────────────────────────────────────────────────────────────────┘
     ▼
  reflect ──(low confidence)──→ migrate            [bounded by max_reflections]
     │
     ▼
  validate ──(fail)──→ fix ──→ migrate             [bounded by max_retries]
     │
     └──(pass)──→ service_validate → observe → END

  migrate ──(ungrounded imports)──→ retrieve       [bounded by max_reretrievals]
```

The double lines are the fan-out. When the `OrchestratorAgent` finds sub-tasks
with no data dependency on each other, `orchestrate_condition` returns one
`Send` per task instead of a branch name; LangGraph runs those invocations of
`branch` concurrently in a single superstep and folds their results back
through the reducers on `GraphState`. Today that overlaps deep analysis with
RAG retrieval — both read only the base metrics `AnalyzerAgent` wrote.

### What the model actually decides

The distinction that matters for an agentic system is whether the LLM controls
*flow*, not just content:

| Decision | Mechanism | Default |
| --- | --- | --- |
| Which route (deep analysis or not) | `RouteDecision` structured output | **on** |
| Which sub-tasks can run concurrently | `SubTaskPlan`, `Literal`-constrained | **on** |
| Which retrieval tools to call, and with what query | Native tool calling (`bind_tools`) | **on** |
| Which retrieval strategy — or none at all | `RetrievalRouteDecision`, `rag_strategy="auto"` | **on** |
| Whether its own output is good enough | `Critique` → re-migrate loop | off |

Each of the first four costs one extra fast-model call and falls back to a
deterministic rule on any failure, so the worst case is the previous behaviour
plus one call. Self-critique is off by default because it costs three calls to
second-guess output the validator has already accepted; enable it with
`ENABLE_REFLECTION=true`.

Verified live, not just wired: with `rag_strategy="auto"` the model picks
`multi_hop` for a generics-and-reflection port and `single_hop` for a rename.

### Noise defenses

| Noise | Defense |
| --- | --- |
| Malformed model output | `with_structured_output` against Pydantic schemas |
| Hallucinated tool names / arguments | Native tool calling with typed `args_schema` |
| Irrelevant retrieved chunks | FlashRank cross-encoder rerank, then a relevance floor |
| Chunks stripped of context | Contextual chunk headers at ingestion |
| Invented APIs | Grounding check + web verification + re-retrieval loop |
| Comments diluting the retrieval query | Comment stripping before symbol extraction |

The reranking row is two defenses, and the second is the one that does the work.
Reordering alone only demotes an irrelevant chunk — it still reaches the prompt.
`rag_rerank_min_score` drops it. Measured against the real index, an off-topic
query ("how do I bake a chocolate cake") returned four chunks scored 0.000 before
the floor existed and returns nothing after it, which is the correct answer: the
caller emits the ungrounded notice rather than inventing grounding.

Built on **LangGraph** for orchestration and **LangChain / ChromaDB** for retrieval.

## Architecture dimensions

The system is organized into six cross-cutting dimensions, each addressing a
distinct class of problem:

| Dimension | Concern | Key modules |
| --- | --- | --- |
| **D1** | Migration pipeline — the core LangGraph workflow | `graph/migration_graph.py`, `graph/nodes.py` |
| **D2** | Retrieval — RAG ingestion, embedding, hybrid search | `rag/` |
| **D3** | Reflection — agent self-critique and regeneration loop | `reflector_agent.py`, `graph/conditions.py` |
| **D4** | Orchestration — parallel sub-task decomposition | `orchestrator_agent.py`, `graph/subgraphs.py` |
| **D5** | Persistence — cross-session memory, graph checkpointing | `memory/`, `pipeline/orchestrator.py` |
| **D6** | Data structures — custom containers tuned for each call site | `dsa/` |

Below, D3–D6 are detailed alongside the D1/D2 features already shown.

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

### Subgraphs & Dynamic Orchestration (D4)
`backend/graph/subgraphs.py` provides five independently compiled subgraphs:
- **analysis** — Deep structural analysis (depends only on base metrics)
- **retrieval** — RAG context provisioning (parallel-safe with analysis)
- **migration** — Plan → generate migrated code
- **fix** — Validate → fix → re-migrate loop (bounded by retries)
- **reflection** — Self-critique of migrated output

The `OrchestratorAgent` decomposes a migration into sub-tasks, resolves them
against the `SUBGRAPH_TASKS` catalog, and fans out data-independent tasks
concurrently via `Send`. The `branch` node dispatches each task to its compiled
subgraph and accumulates results through `operator.add` reducers.

### Self-Critique & Reflection (D3)
The `ReflectorAgent` scores the migrated code on a 0.0–1.0 scale and issues a
recommendation: `"pass"`, `"re-generate"`, or `"gather-more-info"`. The
`reflect_condition` edge routes back to the `migrate` node while the reflection
budget allows (`max_reflections`). Defaults to a passing no-op so the graph
never stalls on an unavailable critique; enable with `ENABLE_REFLECTION=true`.

### RAG Pipeline
- **Corpus**: seven curated migration guides — java8→21, java→python, python2→3,
  python→java, js→ts, ES5→ES2022, java→C# — filed under
  `rag/reference/<target-lang>/examples/<version>/migration/`. That path is
  load-bearing: `derive_doc_metadata` reads the version and doc type from the
  segments, which is what earns each chunk its ranking boosts. **49 chunks**
  across five languages.
- **Ingestion**: Embeds the corpus into ChromaDB in a background task at startup,
  or ahead of time with `scripts/build_index.py`.
- **Code-aware query construction**: Extracts imports, API calls, and type names
  from source code — so queries look like symbols, not prose. Retrieval is much
  stronger on `TryGetValue KeyNotFoundException` than on a natural-language
  paraphrase, and the former is what the pipeline actually generates. Comments are
  stripped before extraction to avoid dilution.
- **Hybrid search**: Dense vector embeddings plus keyword retrieval, fused by
  Reciprocal Rank Fusion (RRF).
- **Version-aware filtering**: Phased retrieval ladder (exact version →
  language-only → unfiltered) with metadata-weighted ranking.
- **Two-stage precision**: fetch 20 candidates, rerank with a local FlashRank
  cross-encoder, then drop everything below `rag_rerank_min_score`. The cosine
  floor is disabled whenever reranking is on — applying it first would let the
  weaker signal overrule the stronger one, and at 0.3 it was rejecting good
  matches outright.
- **Adaptive RAG**: Self-RAG strategies (`single_hop`, `multi_hop`, `hyde`,
  `multi_query`) selected by the model at runtime via `rag_strategy="auto"`.
  Ungrounded imports detected by the migrator trigger a re-retrieval loop
  (bounded by `max_reretrievals`).
- **Code grounding check**: Post-hoc analysis flags library imports not grounded
  by source, RAG context, or target stdlib.
- **Degrades, never fails**: an embedding outage returns no hits instead of
  raising, and the caller emits the ungrounded notice.

### Streaming via SSE
Server-Sent Events deliver token-by-token code output to the frontend. The
`MigratedCodeStreamer` unwraps the JSON wrapper on the fly, showing only clean
migrated code while maintaining JSON structure internally for parsing. Five
event types: `agent_start`, `agent_complete`, `token`, `complete`, `error`.

### Anti-Hallucination Measures
- `UngroundedNotice` prepended when RAG finds no reference examples.
- Version constraints in prompts fence off APIs newer than the target version.
- Structured output means malformed responses are not a reachable state; the
  salvage path in `llm/structured.py` exists for the streaming path and test
  doubles only.

### Model role routing
Two roles, resolved by `llm/providers.py`. `main` generates code; `fast` handles
routing, grading, decomposition, and query reformulation on the cheaper model
with the reasoning budget disabled — a thinking budget spent on a yes/no routing
answer is pure latency. Model ids default per provider, so switching
`LLM_PROVIDER` carries the whole set with it.

### Language Profiles
`backend/llm/language_profiles/` — a `LanguageProfile` dataclass per language
carrying syntax rules, idioms, stdlib mappings, version features, few-shot
examples, and cross-language mappings. The `ProfileRegistry` normalizes aliases
(py→python, js→javascript, c#→csharp) and provides system role prompts tailored
to upgrade vs. conversion migrations.

### Caching
Two-tier cache: **Redis** (primary, with TTL) and **local PrefixLRU** (fallback).
Cache entries are evicted by migration-family prefix (e.g. all migrations into
Java 21) when a language profile or RAG corpus changes — instead of flushing the
whole cache. The local `PrefixLRU` in `dsa/lru.py` uses `OrderedDict` recency
(O(1)) with `str.startswith` prefix scan (outperformed a trie 27x at 500 entries).

### Persistent Memory (D5)
`backend/memory/` stores cross-session migration outcomes in SQLite:
- **MemoryStore**: four tables — `migrations` (successful runs), `patterns`
  (deduplicated conversions), `failures` (runs that errored), `corrections`
  (user edits). Failures are recorded to warn, never to ground.
- **MigrationMemory**: two-stage recall — a `LanguagePairTrie` narrows
  candidates to one language pair in O(k), then cosine similarity ranks
  survivors against the incoming source. `PatternStore` fronts SQLite with a
  bloom filter for O(1) deduplication.
- **Graph checkpointing**: `build_checkpointer` persists LangGraph state per
  `thread_id` via `aiosqlite`, so an interrupted migration can resume.
- The pipeline calls `_recall()` before the graph (seeds prior art into RAG
  context) and `_persist()` after (records outcome or failure).

### Runtime Layer
`backend/runtime/` centralizes agent execution inside the graph:
- **Circuit Breaker** (`agent_recovery.py`): per-agent, 3 consecutive failures
  → 60s cooldown. Consulted by every graph node before constructing an agent.
- **Provider** (`agent_providers.py`): a dependency-injection container with
  instance and factory registration. Agents declare `needs` as a tuple of names;
  the runtime resolves them via `Provider.resolve()` when constructing each
  agent instance.
- **Runtime** (`agent_runtime.py`): the single executor all graph nodes delegate
  to through `_make_node`. Centralizes circuit breaker check, agent resolution,
  DI hydration, execution, error recording, and delta writeback.

### Validator Service
A separate FastAPI microservice (`validator_service/`) with per-language syntax
validators using real toolchains when available, and graceful degradation
otherwise. Called from within the graph by `RuntimeValidatorAgent` via
`clients/validator_client.py`.

### CI/CD Integration
- **GitHub Actions** (`cicd/github-actions.yml`): test → build/push Docker
  images to GHCR → SSH deploy.
- **CodeMigrate Pipeline** (`.github/workflows/codemigrate-pipeline.yml`): On
  PRs, `cicd/scan_candidates.py` detects changed files by extension, runs
  migration via the `.github/actions/codemigrate` composite action, commits to
  a `codemigrate/<stem>` branch, and opens a migration PR.
- **CI/CD Graph** (`backend/graph/cicd_graph.py`): a separate LangGraph for the
  automated PR lifecycle — `check_pr` (parse GitHub event) → `create_pr` (branch
  + commit + PR via PyGithub) → `wait_ci` (poll combined status, 10 min timeout)
  → `auto_merge` (squash-merge on pass). Runnable locally for smoke testing:
  `python -m backend.graph.cicd_graph`.

### Data Structures (D6)
`backend/dsa/` — custom containers, each chosen by measured performance:
- **Trie** — prefix tree backing the `LanguagePairTrie` for open-ended queries
  ("every migration out of Java"). A trie was rejected at other call sites where
  `str.startswith` in C was 15–21x faster.
- **PrefixLRU** — `OrderedDict` recency with prefix-based invalidation for
  cache-family eviction.
- **top_k** — routes between `sorted` (N<128, timsort in C) and `heapq.nlargest`
  (N≥128, heap in interpreted Python) at the measured crossover. Changes the
  constant factor by 2.1x at the low end and 4.2x at the high end.

## Backend

Key modules:

- `backend/llm/providers.py` — provider-agnostic chat model + embeddings, and
  two-layer rate limiting: an open-loop token bucket that paces to the assumed
  quota, plus a reactive `_QuotaGate` that parks *every* caller in the process
  for the reset a 429 advertised. Retries live here rather than in the provider
  SDK, whose own retry ignores that advertised delay and fires below LangChain
  where its extra requests never take a rate-limiter token.
- `backend/llm/structured.py` — structured-output coercion and JSON salvage.
- `backend/models/schemas.py` — every structured LLM response shape.
- `backend/models/state.py` — `MigrationState` Pydantic model with agent reports,
  adaptive-RAG fields, routing plans, parallel task tracking, and memory hooks.
- `backend/models/requests.py` — `MigrateRequest`/`MigrateResponse` with language
  and version validation against `Settings.supported_languages`.
- `backend/graph/migration_graph.py` — LangGraph workflow definition.
- `backend/graph/cicd_graph.py` — CI/CD PR lifecycle graph.
- `backend/graph/subgraphs.py` — five independently compiled subgraphs for the
  orchestrator to dispatch.
- `backend/graph/conditions.py` — edge routing logic: dispatch, orchestrate,
  migrate (adaptive RAG), reflect, validate.
- `backend/graph/state.py` — `GraphState` TypedDict with reducers
  (`operator.add`, `_keep_highest`, `_merge_dicts`).
- `backend/graph/nodes.py` — agent-to-node adapters, DI provider wiring,
  stream token callback.
- `backend/agents/base.py` — `BaseAgent`, `_call_structured`, `bind_tools`,
  reflection hook, `ReflectionResult`.
- `backend/agents/tools/` — `AgentTool`, `StructuredTool` adapter, syntax
  checker, source reader, semantic search, web search, reflection tool.
- `backend/agents/orchestrator_agent.py` — LLM-powered sub-task decomposition.
- `backend/agents/retriever_agent.py` — the ReAct retrieval loop.
- `backend/pipeline/orchestrator.py` — `Pipeline` class: `run()`, `_recall()`,
  `_persist()`, `_run_graph()`. The `run_migration_pipeline()` convenience
  function wires LLM client, cache, and memory.
- `backend/pipeline/registry.py` — `AgentRegistry` with meta-based auto-discovery.
- `backend/runtime/agent_runtime.py` — `Runtime` executor class.
- `backend/runtime/agent_recovery.py` — circuit breaker.
- `backend/runtime/agent_providers.py` — DI `Provider` container.
- `backend/runtime/agent_dispatcher.py` — `DispatcherAgent` (LLM or rule routing).
- `backend/runtime/agent_observer.py` — `ObserverAgent` (metrics + memory persist).
- `backend/runtime/agent_validator.py` — `RuntimeValidatorAgent`.
- `backend/rag/retrieval_pipeline.py` — hybrid retrieval, filter ladder, strategy routing.
- `backend/rag/reranker.py` — FlashRank cross-encoder reranking.
- `backend/rag/ingestion.py` — ChromaDB ingestion pipeline.
- `backend/rag/self_rag.py` — Self-RAG strategies and adaptive retrieval.
- `backend/rag/web_doc_fetcher.py` — online documentation fetching.
- `backend/rag/url_index.py` — URL-based documentation indexing.
- `backend/memory/memory_store.py` — SQLite persistence layer.
- `backend/memory/migration_memory.py` — `LanguagePairTrie` + cosine recall.
- `backend/memory/pattern_store.py` — bloom-filter-deduplicated patterns.
- `backend/memory/checkpointer.py` — graph state checkpointing.
- `backend/dsa/trie.py` — prefix tree.
- `backend/dsa/lru.py` — `PrefixLRU` cache.
- `backend/dsa/ranking.py` — adaptive `top_k` selection.
- `backend/llm/language_profiles/base.py` — `LanguageProfile` dataclass + `ProfileRegistry`.
- `backend/llm/prompt_composer.py` — cached migration prompt builder.
- `backend/scripts/build_index.py` — offline index build (`--stats` to check).
- `backend/clients/validator_client.py` — HTTP client for the validator service.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Provider status, model ids, RAG state, language profiles, metrics |
| `GET` | `/languages` | Supported languages and versions |
| `POST` | `/migrate` | Synchronous migration — returns `MigrateResponse` |
| `POST` | `/migrate/stream` | SSE-streaming migration — events: agent_start, agent_complete, token, complete, error |
| `GET` | `/cache/stats` | Cache hit rates, size, Redis status |
| `POST` | `/cache/clear` | Clear cache, optionally scoped by language/version family |

## Environment variables

Every setting is a field on `Settings` in `backend/config.py`, annotated there
with why the default is what it is; that is the authoritative list. Any field
can be overridden by an upper-case entry in `.env` or a real environment
variable, which wins over the file. The essentials:

```text
LLM_PROVIDER=google_genai        # or "ollama"
GOOGLE_API_KEY=                  # required for google_genai
LLM_REQUESTS_PER_SECOND=0.16     # ~10 RPM free tier; raise on a paid plan
LLM_MAX_RETRIES=3                # retries are ours, not the provider SDK's
LLM_QUOTA_COOLDOWN_SEC=30        # hold when a 429 advertises no reset delay
EMBEDDING_PROVIDER=google_genai
REDIS_URL=redis://redis:6379/0
VALIDATOR_URL=http://validator:8000
ENABLE_VALIDATION=false
ENABLE_RAG=false
ENABLE_GROUNDING_CHECK=false
ENABLE_WEB_DOCS=false
OLLAMA_AUTO_PULL=true
ENABLE_REFLECTION=false           # D3 self-critique
MEMORY_ENABLED=true               # D5 cross-session persistence
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

Uses real syntax tools when available; returns a structured warning if a toolchain is missing. Validators for: Python, JavaScript, TypeScript, Java, C#, Go, Kotlin, Rust, C++, C++ (header).

## Frontend

A React single-page app (`frontend/`) with:

- **Sidebar** — source/target language and version selectors with swap button
- **Editor** — source code input with syntax highlighting
- **Output** — streaming migrated code display
- **AgentLog** — real-time agent pipeline progress (analyze → deep analyze → retrieve → plan → migrate → validate → fix → observe)
- **Header** — branding, example selectors, LLM status pill

Three pre-loaded examples: Java 7 → Java 17, Java 8 → Python 3.12, Python 2.7 → Python 3.12.

Keyboard shortcuts: `Ctrl+Enter` / `⌘↵` to run, `Esc` to cancel. Backend health polls every 15 s.

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
python -m pytest -q                     # ~500 offline tests, no key or network needed
python -m compileall -q backend validator_service
cd frontend; npm.cmd run build
```

Tests are configured in `pyproject.toml`: `--import-mode=importlib` for the
non-installed backend layout, `-m 'not live'` to skip tests needing API keys.
Key test files: `test_graph.py`, `test_rag.py`, `test_providers.py`,
`test_memory.py`, `test_orchestration.py`, `test_rag_strategies.py`,
`test_grounding.py`, `test_reflection.py`, `test_streaming.py`, `test_dsa.py`.
The `conftest.py` fixture isolates the circuit breaker, observer metrics, and
provider cache per test.

Before a demo or a submission, also run the opt-in live checks:

```powershell
python -m pytest backend/tests/test_live_provider.py -m live -v
```

Five small real API calls confirming the things no offline test can: that the
configured chat and embedding models are still *served*, that every tool schema
is accepted by the provider, and that the Chroma index was built with the
embedding model currently configured.

This matters more than it sounds. Model ids are strings, and a string keeps
passing every offline test long after the vendor stops serving it — Google
retired `gemini-2.5-flash` and `gemini-2.5-flash-lite` for new accounts while
they were this project's defaults, and the `/models` listing still advertised
them. The whole suite was green; every real migration would have 404'd on its
first call.
