"""Opt-in checks that the configured provider will actually serve us.

Everything else in this suite runs against stubs, which is right — a unit test
should not depend on a network or a quota. But it leaves one failure mode
completely uncovered: the model ids in ``providers.DEFAULT_MODELS`` are strings,
and a string keeps passing every offline test long after the vendor stops
serving it.

That is not hypothetical. Google retired ``gemini-2.5-flash`` and
``gemini-2.5-flash-lite`` for new accounts while this project had them as its
defaults; every offline test stayed green and every real migration would have
failed on its first call with a 404. Worse, the ``/models`` listing endpoint
still advertises them, so even checking the catalogue would have said they were
fine. The only thing that settles it is calling them.

Run before a demo, a submission, or a release::

    pytest backend/tests/test_live_provider.py -m live -v

Deselected by default (see ``addopts`` in pyproject.toml), so the normal suite
stays offline and free. Each test makes one small call.
"""

import pytest
from config import get_settings
from llm import providers

# Marked per test rather than module-wide: one check here inspects generated
# schemas offline and must run in the ordinary suite, where it is free.


def _skip_without_credentials():
    settings = get_settings()
    if providers.is_hosted(settings) and not settings.google_api_key:
        pytest.skip(f"no API key configured for {settings.llm_provider}")
    return settings


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["main", "fast"])
async def test_the_configured_model_for_each_role_is_served(role):
    """The check that would have caught the 2.5 retirement.

    Asserts nothing about the answer — only that the provider accepts the model
    id and returns something. A 404 here means the default is dead and every
    migration is about to fail; a 429 means the quota is exhausted, which is a
    different problem with the same symptom, so the assertion message
    distinguishes them.
    """
    settings = _skip_without_credentials()
    name = providers.resolve_model_name(role, settings)

    try:
        reply = await providers.get_chat_model(role, settings=settings).ainvoke(
            "Reply with the single word: OK"
        )
    except Exception as exc:  # noqa: BLE001 — the failure *is* the result here
        detail = str(exc)
        if "NOT_FOUND" in detail or "404" in detail:
            pytest.fail(
                f"Model {name!r} (role={role}) is no longer served. Update "
                f"providers.DEFAULT_MODELS[{settings.llm_provider!r}][{role!r}]."
            )
        pytest.fail(f"Model {name!r} (role={role}) unreachable: {detail[:300]}")

    assert reply.text.strip(), f"{name} returned an empty response"


@pytest.mark.live
@pytest.mark.asyncio
async def test_the_configured_embedding_model_is_served():
    """Embeddings retire independently of chat models, and fail more quietly.

    A dead chat model stops a migration outright. A dead embedding model just
    makes retrieval return nothing, which looks like a thin corpus rather than
    an outage — so this is the one worth checking explicitly.
    """
    settings = _skip_without_credentials()
    name = providers.resolve_embedding_model(settings)

    try:
        vector = await providers.get_embeddings(settings).aembed_query("migration")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Embedding model {name!r} unreachable: {str(exc)[:300]}")

    assert vector, f"{name} returned an empty vector"
    assert any(vector), f"{name} returned an all-zero vector"


def test_no_tool_schema_declares_an_empty_enum_value():
    """Offline guard for a schema the provider rejects but Pydantic accepts.

    ``Literal["", "precise", …]`` is valid Python and produces a JSON schema
    with an empty enum value, which Gemini refuses at bind time:
    ``properties[intent].enum[0]: cannot be empty`` (400). The retrieval tool
    shipped exactly that, so every tool call failed and the agent fell back to
    its no-tool path — the symptom was weaker retrieval, never an error, while
    all the offline tool tests passed against a stub that never saw the schema.

    Express "no value" as ``Optional[Literal[...]] = None`` instead.

    Not marked live: it inspects generated schemas and needs no API call. It
    lives here because it belongs with the other "will the provider actually
    accept us" checks.
    """
    from agents.tools import build_registry

    registry = build_registry(rag_pipeline=object(), migration_memory=object())
    offenders = []
    for tool in registry.as_langchain_tools():
        schema = tool.args_schema
        properties = (
            schema.model_json_schema().get("properties", {})
            if hasattr(schema, "model_json_schema")
            else {}
        )
        for field, spec in properties.items():
            candidates = [spec] + spec.get("anyOf", []) + spec.get("oneOf", [])
            for candidate in candidates:
                for value in candidate.get("enum") or []:
                    if value is not None and str(value).strip() == "":
                        offenders.append(f"{tool.name}.{field}")

    assert not offenders, (
        f"empty enum value in tool schema(s): {offenders}. Providers reject the "
        "whole tool declaration for this; use Optional[Literal[...]] = None."
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_every_tool_schema_is_accepted_by_the_provider():
    """Bind the real registry to the real model.

    The offline check above catches the one failure mode we have already seen.
    This catches the rest — anything else a provider dislikes about a generated
    schema — by doing the thing that actually settles it. One call.
    """
    settings = _skip_without_credentials()
    from agents.tools import build_registry

    tools = build_registry(
        rag_pipeline=object(), migration_memory=object()
    ).as_langchain_tools()
    if not tools:
        pytest.skip("no tools registered")

    model = providers.get_chat_model("fast", settings=settings).bind_tools(tools)
    try:
        await model.ainvoke("What is 2 + 2? Answer directly.")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"provider rejected the tool declarations ({len(tools)} tools): "
            f"{str(exc)[:400]}"
        )


@pytest.mark.live
@pytest.mark.asyncio
async def test_the_index_matches_the_embedding_model_in_use():
    """Guards the mismatch that takes retrieval down without an error message.

    Chroma pins a collection to the width of whatever embedded it. Change the
    embedding model — or build the index with one provider and query with
    another — and every search raises "expecting dimension N, got M" rather than
    returning poor results. Rebuild with scripts/build_index.py when this fails.
    """
    settings = _skip_without_credentials()
    from rag.embedding_service import CachedEmbeddings
    from rag.vector_store import VectorStore

    embeddings = CachedEmbeddings(providers.get_embeddings(settings), max_cache=8)
    store = VectorStore(embeddings)
    store.initialize()
    if not store.count():
        pytest.skip("index is empty; run scripts/build_index.py")

    hits = store.similarity_search("migration guide", k=1, score_threshold=0.0)
    assert hits, (
        "The index is populated but returns nothing for a generic query, which "
        "points at an embedding/collection dimension mismatch. Rebuild it with "
        f"scripts/build_index.py using {providers.resolve_embedding_model(settings)}."
    )
