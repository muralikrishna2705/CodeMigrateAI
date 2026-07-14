"""CI/CD wrapper graph (Phase 4).

LangGraph nodes that drive the automated, PR-based migration workflow::

    check_pr ──▶ create_pr ──▶ wait_ci ──(passed)──▶ auto_merge ──▶ END
                                   └──(not passed)──────────────────▶ END

* ``check_pr``   reads the triggering GitHub PR payload (``GITHUB_EVENT_PATH``).
* ``create_pr``  opens a migration PR with the migrated code (PyGithub).
* ``wait_ci``    polls the migration PR's combined status until it settles.
* ``auto_merge`` squash-merges the PR only when CI passed.

Every GitHub call is lazy-imported and guarded on ``GITHUB_TOKEN`` /
``GITHUB_REPOSITORY``, so the module imports and the graph compiles/runs even
without network access or credentials — the nodes simply no-op in that case.
This keeps ``python -m graph.cicd_graph`` runnable locally for a smoke test.
"""

import datetime
import json
import logging
import os
import time

from langgraph.graph import END, StateGraph

from .state import GraphState

log = logging.getLogger("CodeMigrateAI.CICDGraph")


class CICDState(GraphState):
    """Migration graph state extended with CI/CD bookkeeping fields.

    Inherits every :class:`~graph.state.GraphState` field (``migrated_code``,
    ``source_language`` …) and adds the pull-request / CI tracking used by the
    nodes below. Like ``GraphState`` this is a ``total=False`` ``TypedDict`` —
    all keys are optional, so callers seed only what they have.
    """

    source_path: str
    pr_number: int
    pr_url: str
    ci_status: str
    auto_merge_attempted: bool


def _github_repo():
    """Return a ``(repo, pr_number)`` handle or ``None`` if unconfigured.

    Centralises the ``GITHUB_TOKEN`` / ``GITHUB_REPOSITORY`` guard and the lazy
    PyGithub import so the nodes stay short and the module imports without
    PyGithub installed.
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo_name:
        return None
    from github import Github

    return Github(token).get_repo(repo_name)


async def check_pr_node(state: dict) -> dict:
    """Parse the triggering PR payload and record its number/URL."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path, encoding="utf-8") as f:
            event = json.load(f)
        pr = event.get("pull_request")
        if pr:
            state["pr_number"] = pr.get("number", 0)
            state["pr_url"] = pr.get("html_url", "")
            log.info("PR #%s: %s", state["pr_number"], pr.get("title", ""))
    return state


async def create_pr_node(state: dict) -> dict:
    """Create a migration PR (branch + commit + PR) using PyGithub."""
    repo = _github_repo()
    if repo is None:
        log.warning("GITHUB_TOKEN/GITHUB_REPOSITORY unset; skipping PR creation")
        return state

    branch_name = f"codemigrate/{datetime.date.today().isoformat()}"
    try:
        base_branch = repo.get_branch(repo.default_branch)
        repo.create_git_ref(f"refs/heads/{branch_name}", base_branch.commit.sha)
    except Exception as e:  # branch may already exist — non-fatal
        log.warning("Branch creation skipped: %s", e)

    title = f"CodeMigrate: {state.get('source_language')} -> {state.get('target_language')}"

    # Commit the migrated code onto the new branch (best effort).
    source_path = state.get("source_path")
    migrated_code = state.get("migrated_code")
    if source_path and migrated_code:
        try:
            contents = repo.get_contents(source_path, ref=branch_name)
            repo.update_file(
                contents.path, title, migrated_code, contents.sha, branch=branch_name
            )
        except Exception:  # file doesn't exist yet -> create it
            try:
                repo.create_file(source_path, title, migrated_code, branch=branch_name)
            except Exception as e:
                log.warning("Commit skipped for %s: %s", source_path, e)

    body = (
        "Automated migration by CodeMigrateAI.\n\n"
        f"Source: {state.get('source_language')} {state.get('source_version')}\n"
        f"Target: {state.get('target_language')} {state.get('target_version')}\n"
    )
    try:
        pr = repo.create_pull(
            title=title, body=body, head=branch_name, base=repo.default_branch
        )
        state["pr_url"] = pr.html_url
        state["pr_number"] = pr.number
        log.info("Created PR: %s", pr.html_url)
    except Exception as e:
        log.warning("PR creation skipped: %s", e)
    return state


async def wait_ci_node(state: dict) -> dict:
    """Poll the migration PR's combined status until it settles (≤10 min)."""
    repo = _github_repo()
    if repo is None or not state.get("pr_number"):
        state["ci_status"] = "unknown"
        return state

    pr = repo.get_pull(state["pr_number"])
    for _ in range(60):  # 60 × 10s = 10 minutes
        pr.update()
        combined = pr.get_combined_status()
        if combined.state == "success":
            state["ci_status"] = "passed"
            return state
        if combined.state == "failure":
            state["ci_status"] = "failed"
            return state
        time.sleep(10)

    state["ci_status"] = "timeout"
    return state


async def auto_merge_node(state: dict) -> dict:
    """Squash-merge the migration PR when CI passed."""
    if state.get("ci_status") != "passed":
        log.info("CI status is %r, skipping auto-merge", state.get("ci_status"))
        return state

    repo = _github_repo()
    if repo is None or not state.get("pr_number"):
        return state

    pr = repo.get_pull(state["pr_number"])
    pr.merge(merge_method="squash")
    state["auto_merge_attempted"] = True
    log.info("Auto-merged PR #%s", state["pr_number"])
    return state


def _merge_condition(state: dict) -> str:
    """Route: CI passed -> auto_merge, otherwise end."""
    return "auto_merge" if state.get("ci_status") == "passed" else END


def build_cicd_graph():
    """Construct and compile the CI/CD migration workflow graph (Phase 4)."""
    workflow = StateGraph(CICDState)

    workflow.add_node("check_pr", check_pr_node)
    workflow.add_node("create_pr", create_pr_node)
    workflow.add_node("wait_ci", wait_ci_node)
    workflow.add_node("auto_merge", auto_merge_node)

    workflow.set_entry_point("check_pr")
    workflow.add_edge("check_pr", "create_pr")
    workflow.add_edge("create_pr", "wait_ci")
    workflow.add_conditional_edges(
        "wait_ci",
        _merge_condition,
        {"auto_merge": "auto_merge", END: END},
    )
    workflow.add_edge("auto_merge", END)

    app = workflow.compile()
    log.info("CI/CD graph compiled (Phase 4)")
    return app


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        app = build_cicd_graph()
        result = await app.ainvoke(
            {
                "source_code": "",
                "source_path": os.environ.get("SOURCE_PATH", ""),
                "migrated_code": "",
                "source_language": os.environ.get("SOURCE_LANG", "java"),
                "source_version": os.environ.get("SOURCE_VER", "7"),
                "target_language": os.environ.get("TARGET_LANG", "java"),
                "target_version": os.environ.get("TARGET_VER", "17"),
                "migration_type": "upgrade_version",
                "retry_count": 0,
                "max_retries": 2,
                "best_effort_code": "",
                "reports": [],
                "errors": [],
                "agents_completed": [],
                "pr_number": 0,
                "pr_url": "",
                "ci_status": "pending",
                "auto_merge_attempted": False,
            }
        )
        # Persist for the composite action's upload-artifact and cicd/create_pr.py.
        with open("migration_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())
