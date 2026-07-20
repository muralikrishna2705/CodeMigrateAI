"""Fan-out execution for independent migration sub-tasks.

The orchestrator decomposes a migration into sub-tasks; the ones with no data
dependency on each other run here concurrently, each inside its own compiled
subgraph (see :mod:`graph.subgraphs`), bounded by a semaphore.

**Why ``asyncio.gather`` rather than LangGraph's native fan-out.** LangGraph runs
multiple edges out of one node in the same superstep, but every branch then emits
a partial update for the *same* ``GraphState`` keys, and a plain ``TypedDict``
without per-key reducers rejects that with ``InvalidUpdateError: Can receive only
one value per step``. Adding ``Annotated[..., operator.add]`` reducers is not a
drop-in here either: the nodes in this project return the *whole* accumulated
state dict (``nodes.writeback`` returns ``state``), so an additive reducer would
re-append every historical report on each update. Running the branches inside one
node over isolated state copies keeps that contract intact — the graph still sees
a single node returning a single state — and gives us explicit control over both
the concurrency ceiling and how conflicts are resolved (see :mod:`graph.merge`).

Each branch gets a deep copy of the inbound state, so concurrent branches cannot
observe or clobber each other's partial writes; :func:`graph.merge.merge_results`
folds them back together afterwards.
"""

import asyncio
import copy
import logging
import time

log = logging.getLogger("CodeMigrateAI.GraphParallel")


class BranchResult:
    """One parallel branch's outcome.

    ``state`` is the branch's own (isolated) resulting graph state on success and
    ``None`` on failure. A failed branch never propagates its exception: the
    parallel path is an execution *strategy*, so a branch that blows up must
    degrade to "this sub-task contributed nothing", exactly as a skipped agent
    would — never take down the migration.
    """

    __slots__ = ("task", "state", "error", "duration_ms")

    def __init__(
        self,
        task: str,
        state: dict | None = None,
        error: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        self.task = task
        self.state = state
        self.error = error
        self.duration_ms = duration_ms

    @property
    def ok(self) -> bool:
        return self.error is None and self.state is not None

    def to_dict(self, base_agents: list | None = None) -> dict:
        """The record written onto ``state["subgraph_results"]``.

        ``base_agents`` is the ``agents_completed`` list the branch started from.
        When given, ``agents`` reports only what *this branch* ran; without it
        the branch's whole list is used. The delta is what makes the record
        readable — a branch inherits the run's full history, so the raw list
        would name every agent that ran before the fan-out too.
        """
        agents = list((self.state or {}).get("agents_completed", []))
        if base_agents is not None:
            agents = agents[len(base_agents) :]
        record = {
            "task": self.task,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "agents": agents,
        }
        if self.error:
            record["error"] = self.error
        return record


async def run_parallel(
    tasks: list[tuple[str, object]],
    state: dict,
    *,
    max_concurrency: int = 4,
) -> list[BranchResult]:
    """Run ``tasks`` concurrently over isolated copies of ``state``.

    ``tasks`` is a list of ``(name, runnable)`` pairs, where ``runnable`` is any
    compiled LangGraph app (or plain async callable) accepting a state dict and
    returning one. Results come back in the same order as ``tasks`` so the merge
    step can apply a deterministic precedence regardless of completion order.

    ``max_concurrency`` is clamped to at least 1; a non-positive setting would
    otherwise deadlock the semaphore.
    """
    if not tasks:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(name: str, runnable) -> BranchResult:
        async with semaphore:
            start = time.perf_counter()
            # Deep copy per branch: branches run concurrently against the same
            # inbound state and each mutates it in place (nodes hydrate/writeback
            # onto the dict they are handed), so sharing it would interleave
            # writes non-deterministically.
            branch_state = copy.deepcopy(state)
            try:
                result = await runnable.ainvoke(branch_state)
                duration_ms = int((time.perf_counter() - start) * 1000)
                log.info("Parallel branch %r finished in %dms", name, duration_ms)
                return BranchResult(name, state=result, duration_ms=duration_ms)
            except Exception as exc:  # noqa: BLE001 — a branch must never abort the run
                duration_ms = int((time.perf_counter() - start) * 1000)
                log.exception("Parallel branch %r failed after %dms", name, duration_ms)
                return BranchResult(name, error=str(exc), duration_ms=duration_ms)

    started = time.perf_counter()
    log.info(
        "Fanning out %d task(s) with concurrency %d: %s",
        len(tasks),
        max(1, max_concurrency),
        ", ".join(name for name, _ in tasks),
    )
    # return_exceptions is belt-and-braces: run_one already catches everything,
    # but a cancellation or an error raised outside the try (e.g. the deepcopy)
    # would otherwise propagate and lose the sibling branches' work.
    results = await asyncio.gather(
        *(run_one(name, runnable) for name, runnable in tasks),
        return_exceptions=True,
    )

    branches: list[BranchResult] = []
    for (name, _), result in zip(tasks, results):
        if isinstance(result, BaseException):
            log.error("Parallel branch %r raised outside its guard: %s", name, result)
            branches.append(BranchResult(name, error=str(result)))
        else:
            branches.append(result)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    failed = [b.task for b in branches if not b.ok]
    log.info(
        "Fan-out complete in %dms (%d ok, %d failed%s)",
        elapsed_ms,
        len(branches) - len(failed),
        len(failed),
        f": {', '.join(failed)}" if failed else "",
    )
    return branches
