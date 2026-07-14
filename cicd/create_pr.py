#!/usr/bin/env python3
"""Create a migration PR for a single file from a migration-result JSON.

Reads ``--result`` (the JSON produced by ``backend.graph.cicd_graph``), takes its
``migrated_code``, commits it to a per-file ``codemigrate/<stem>`` branch, and
opens a pull request. Requires ``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY``.
"""

import argparse
import json
import os
from pathlib import Path

from github import Github


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Source file path")
    parser.add_argument("--result", required=True, help="Migration result JSON")
    args = parser.parse_args()

    if not os.path.exists(args.result):
        print(f"Result file not found: {args.result}")
        return

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)

    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo_name:
        print("GITHUB_TOKEN and GITHUB_REPOSITORY required")
        return

    migrated_code = result.get("migrated_code", "")
    if not migrated_code:
        print("No migrated code in result; nothing to open a PR for")
        return

    source_path = Path(args.file)
    if not source_path.exists():
        print(f"Source file not found: {args.file}")
        return

    g = Github(token)
    repo = g.get_repo(repo_name)

    # Create the branch (ignore "already exists").
    branch_name = f"codemigrate/{source_path.stem}"
    try:
        base = repo.get_branch(repo.default_branch)
        repo.create_git_ref(f"refs/heads/{branch_name}", base.commit.sha)
    except Exception:
        pass

    # Commit the migrated code (update if present, else create).
    posix_path = source_path.as_posix()
    commit_msg = f"Migrate {source_path.name}"
    try:
        contents = repo.get_contents(posix_path, ref=branch_name)
        repo.update_file(
            contents.path, commit_msg, migrated_code, contents.sha, branch=branch_name
        )
    except Exception:
        repo.create_file(posix_path, commit_msg, migrated_code, branch=branch_name)

    pr = repo.create_pull(
        title=f"CodeMigrate: {source_path.name}",
        body=f"Automated migration by CodeMigrateAI.\n\n{result.get('inline_plan', '')}",
        head=branch_name,
        base=repo.default_branch,
    )
    print(f"Created PR: {pr.html_url}")


if __name__ == "__main__":
    main()
