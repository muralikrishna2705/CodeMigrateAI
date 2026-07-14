#!/usr/bin/env python3
"""Scan the current PR diff for files matching migration-candidate patterns.

Writes ``candidates.json`` (a compact, single-line JSON array so it can be piped
straight into ``$GITHUB_OUTPUT``) and echoes a pretty version for the CI log.
Each candidate carries the source file plus the target migration parameters.
"""

import json
import os
import subprocess
import sys

# Language detection by file extension.
EXTENSION_MAP = {
    ".py": {"language": "python"},
    ".java": {"language": "java"},
    ".js": {"language": "javascript"},
    ".ts": {"language": "typescript"},
    ".cs": {"language": "csharp"},
    ".go": {"language": "go"},
    ".kt": {"language": "kotlin"},
    ".rs": {"language": "rust"},
    ".cpp": {"language": "cpp"},
    ".cxx": {"language": "cpp"},
    ".cc": {"language": "cpp"},
    ".hpp": {"language": "cpp"},
    ".h": {"language": "cpp"},
}

# Target language/version per source language. Override by editing this map or
# wiring in a repo-level config file.
MIGRATION_TARGETS = {
    "python": {"target_language": "python", "target_version": "3.12"},
    "java": {"target_language": "java", "target_version": "17"},
    "javascript": {"target_language": "javascript", "target_version": "ES2022"},
    "typescript": {"target_language": "typescript", "target_version": "5.x"},
    "csharp": {"target_language": "csharp", "target_version": "12"},
    "go": {"target_language": "go", "target_version": "1.22"},
    "kotlin": {"target_language": "kotlin", "target_version": "2.0"},
    "rust": {"target_language": "rust", "target_version": "1.80"},
    "cpp": {"target_language": "cpp", "target_version": "23"},
}


def _git_diff_names(rev_range: str) -> list[str] | None:
    """Return changed file paths for ``rev_range``, or ``None`` on git error."""
    result = subprocess.run(
        ["git", "diff", "--name-only", rev_range],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_changed_files() -> list[str]:
    """Get the list of files changed in the PR (with a sensible fallback)."""
    # Prefer the GitHub event payload: diff the PR's base..head range.
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, encoding="utf-8") as f:
                event = json.load(f)
            pr = event.get("pull_request")
            if pr:
                base = pr.get("base", {}).get("sha", "main")
                head = pr.get("head", {}).get("sha", "HEAD")
                changed = _git_diff_names(f"{base}...{head}")
                if changed is not None:
                    return changed
        except (OSError, json.JSONDecodeError):
            pass

    # Fallback: diff against origin/main.
    return _git_diff_names("origin/main...") or []


def detect_language(filepath: str) -> dict | None:
    """Detect the source language from a file extension."""
    _, ext = os.path.splitext(filepath)
    return EXTENSION_MAP.get(ext.lower())


def build_candidates(changed_files: list[str]) -> list[dict]:
    """Map changed files to migration candidates."""
    candidates: list[dict] = []
    for filepath in changed_files:
        if not os.path.exists(filepath):
            continue
        detected = detect_language(filepath)
        if not detected:
            continue
        lang = detected["language"]
        target = MIGRATION_TARGETS.get(lang)
        if not target:
            continue
        candidates.append(
            {
                "path": filepath,
                "source_language": lang,
                "source_version": "",  # detected downstream by analyzing the file
                **target,
            }
        )
    return candidates


def main() -> None:
    changed_files = get_changed_files()
    candidates = build_candidates(changed_files) if changed_files else []

    # Compact for $GITHUB_OUTPUT (single line); pretty for the log.
    with open("candidates.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(candidates))

    if not changed_files:
        print("No changed files detected")
    print(json.dumps(candidates, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
