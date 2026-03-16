#!/usr/bin/env python3
"""Prepare ABI-specific review context for a diff against a base ref."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONSENSUS_WRAPPER = (
    Path.home()
    / ".codex"
    / "skills"
    / "multi-cli-communication-expert"
    / "scripts"
    / "run_tricli_consensus.sh"
)
ABI_MARKERS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".zigversion",
    "build.zig",
    "src/root.zig",
)

CATEGORY_ORDER = (
    "build-system",
    "toolchain",
    "cli",
    "tui",
    "docs",
    "features",
    "feature-flag-surface",
    "wdbx",
    "database",
    "network-dist",
    "training",
    "tasks-planning",
)


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_repo_root(repo_arg: str) -> Path:
    repo_path = Path(repo_arg).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"error: repo path does not exist: {repo_path}")
    try:
        root = run_git(repo_path, "rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "not a git repository"
        raise SystemExit(f"error: unable to resolve git repo root: {stderr}") from exc
    return Path(root).resolve()


def verify_abi_checkout(repo_root: Path) -> dict[str, bool]:
    marker_state = {marker: (repo_root / marker).exists() for marker in ABI_MARKERS}
    if not all(marker_state.values()):
        missing = [marker for marker, present in marker_state.items() if not present]
        joined = ", ".join(missing)
        raise SystemExit(
            f"error: repo does not look like an ABI checkout; missing required markers: {joined}"
        )
    return marker_state


def read_zig_version(repo_root: Path) -> str:
    return (repo_root / ".zigversion").read_text(encoding="utf-8").strip()


def collect_changed_files(repo_root: Path, merge_base: str) -> list[str]:
    output = run_git(repo_root, "diff", "--name-only", merge_base, "HEAD", "--")
    if not output:
        return []
    return [line for line in output.splitlines() if line]


def collect_diff_stats(repo_root: Path, merge_base: str) -> dict[str, int]:
    files_changed = 0
    insertions = 0
    deletions = 0
    output = run_git(repo_root, "diff", "--numstat", merge_base, "HEAD", "--")
    for line in output.splitlines():
        if not line:
            continue
        added, removed, _path = line.split("\t", 2)
        files_changed += 1
        if added.isdigit():
            insertions += int(added)
        if removed.isdigit():
            deletions += int(removed)
    return {
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
    }


def categorize_paths(paths: list[str]) -> list[str]:
    categories: set[str] = set()
    for path in paths:
        if path in {"build.zig", "build.zig.zon", ".zigversion"} or path.startswith(
            "build/"
        ):
            categories.add("build-system")
        if (
            path in {"build.zig.zon", ".zigversion"}
            or path.startswith(".cel/")
            or path.startswith("tools/scripts/")
        ):
            categories.add("toolchain")
        if path.startswith("tools/cli/"):
            categories.add("cli")
        if path.startswith("tools/cli/terminal/") or path.endswith(
            "tui_tests_root.zig"
        ):
            categories.add("tui")
        if (
            path.startswith("tools/gendocs/")
            or path.startswith("docs/")
            or path in {"README.md", "CLAUDE.md", "AGENTS.md"}
        ):
            categories.add("docs")
        if path.startswith("src/features/"):
            categories.add("features")
        if path.startswith("tasks/"):
            categories.add("tasks-planning")
        if path.startswith("src/core/database/"):
            categories.add("wdbx")
        if path.startswith("src/features/database/"):
            categories.add("database")
        if path.startswith("src/core/database/dist/") or path.startswith(
            "src/features/network/"
        ):
            categories.add("network-dist")
        if path.startswith("src/features/ai/training/") or "training" in path:
            categories.add("training")

        feature_parts = path.split("/")
        if (
            len(feature_parts) >= 4
            and feature_parts[0] == "src"
            and feature_parts[1] == "features"
            and feature_parts[-1] in {"mod.zig", "stub.zig"}
        ):
            categories.add("feature-flag-surface")
        if path in {
            "build/options.zig",
            "build/flags.zig",
            "src/core/feature_catalog.zig",
        }:
            categories.add("feature-flag-surface")

    return [category for category in CATEGORY_ORDER if category in categories]


def recommend_commands(paths: list[str]) -> list[dict[str, str]]:
    if not paths:
        return []

    recommended: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(command: str, reason: str) -> None:
        if command in seen:
            return
        seen.add(command)
        recommended.append({"command": command, "reason": reason})

    add("zig build full-check", "ABI default pre-close gate for non-trivial changes.")
    add(
        "zig build verify-all",
        "ABI release-style umbrella gate; run on a host where the toolchain links.",
    )

    if any(
        path.startswith("tools/cli/") or path == "docs/data/commands.zon"
        for path in paths
    ):
        add(
            "zig build cli-tests",
            "CLI files changed; cover command parsing and help/output behavior.",
        )
        add(
            "zig build refresh-cli-registry",
            "CLI metadata or command files changed; refresh the generated registry snapshot.",
        )
        add(
            "zig build check-cli-registry",
            "CLI metadata or command files changed; verify the registry snapshot stayed in sync.",
        )

    if any(
        path.startswith("tools/cli/terminal/") or path.endswith("tui_tests_root.zig")
        for path in paths
    ):
        add(
            "zig build tui-tests",
            "TUI files changed; cover panels, dashboard logic, and terminal behavior.",
        )

    if any(
        path.startswith("tools/gendocs/")
        or path.startswith("docs/")
        or path in {"README.md", "CLAUDE.md"}
        for path in paths
    ):
        add(
            "zig build check-docs",
            "Docs or docs-generation files changed; catch drift in generated references.",
        )

    if any(
        path.startswith("src/core/database/")
        or path.startswith("src/features/database/")
        for path in paths
    ):
        add(
            "zig build wdbx-fast-tests",
            "WDBX or database paths changed; run the focused database gate.",
        )

    if any(
        path in {"build/options.zig", "build/flags.zig", "src/core/feature_catalog.zig"}
        or path.startswith("src/features/")
        and path.endswith(("mod.zig", "stub.zig"))
        for path in paths
    ):
        add(
            "zig build validate-flags",
            "Feature surfaces or build flag wiring changed; verify disabled-build compatibility.",
        )

    return recommended


def build_reminders(zig_version: str) -> list[str]:
    return [
        f"ABI pins Zig via .zigversion: {zig_version}. Review changes as Zig 0.16-dev/master-era code, not as a stable 0.15 or 0.16 release port.",
        "Review build wiring for modern Zig patterns: b.createModule(...), .root_module, and no deprecated LazyPath.path usage.",
        "For src/features changes, verify mod.zig and stub.zig stay aligned when public surfaces move, and keep feature imports relative.",
        "Known Darwin hosts may fail binary-emitting steps with stock Zig; treat that as environment noise unless the patch changes .cel, bootstrap, or toolchain detection.",
        (
            "AGENTS.md documents a best-effort consensus wrapper at "
            f"{CONSENSUS_WRAPPER}. Record it as unavailable if it is missing locally."
        ),
    ]


def make_payload(repo_root: Path, base_ref: str) -> dict[str, object]:
    markers = verify_abi_checkout(repo_root)
    zig_version = read_zig_version(repo_root)
    branch = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    head = run_git(repo_root, "rev-parse", "HEAD")
    base_resolved = run_git(repo_root, "rev-parse", base_ref)
    merge_base = run_git(repo_root, "merge-base", "HEAD", base_ref)
    changed_files = collect_changed_files(repo_root, merge_base)
    diff_stats = collect_diff_stats(repo_root, merge_base)
    categories = categorize_paths(changed_files)
    recommendations = recommend_commands(changed_files)
    status = "no changes to review" if not changed_files else "changes ready for review"

    return {
        "status": status,
        "repo_root": str(repo_root),
        "abi_markers": markers,
        "zig_version": zig_version,
        "git": {
            "branch": branch,
            "head": head,
            "base_ref": base_ref,
            "base_resolved": base_resolved,
            "merge_base": merge_base,
        },
        "consensus_wrapper": {
            "path": str(CONSENSUS_WRAPPER),
            "available": CONSENSUS_WRAPPER.exists(),
        },
        "diff_stats": diff_stats,
        "changed_files": changed_files,
        "categories": categories,
        "recommended_commands": recommendations,
        "review_reminders": build_reminders(zig_version),
    }


def render_text(payload: dict[str, object]) -> str:
    git_info = payload["git"]
    diff_stats = payload["diff_stats"]
    changed_files = payload["changed_files"]
    categories = payload["categories"]
    commands = payload["recommended_commands"]
    reminders = payload["review_reminders"]
    consensus = payload["consensus_wrapper"]

    lines = [
        "ABI review prep",
        f"Status: {payload['status']}",
        f"Repo: {payload['repo_root']}",
        f"Zig pin: {payload['zig_version']}",
        f"Branch: {git_info['branch']}",
        f"HEAD: {git_info['head']}",
        f"Base ref: {git_info['base_ref']} -> {git_info['base_resolved']}",
        f"Merge base: {git_info['merge_base']}",
        (
            "Consensus wrapper: "
            f"{'available' if consensus['available'] else 'missing'} at {consensus['path']}"
        ),
        (
            "Diff stats: "
            f"{diff_stats['files_changed']} files, +{diff_stats['insertions']} / -{diff_stats['deletions']}"
        ),
    ]

    if changed_files:
        lines.append("Changed files:")
        lines.extend(f"  - {path}" for path in changed_files)
    else:
        lines.append("Changed files: none")

    if categories:
        lines.append("Categories: " + ", ".join(categories))
    else:
        lines.append("Categories: none")

    if commands:
        lines.append("Recommended commands:")
        lines.extend(f"  - {item['command']}: {item['reason']}" for item in commands)
    else:
        lines.append("Recommended commands: none")

    lines.append("Review reminders:")
    lines.extend(f"  - {reminder}" for reminder in reminders)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to the ABI repository. Defaults to the current directory.",
    )
    parser.add_argument(
        "--base", default="main", help="Base ref to compare against. Defaults to main."
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo)
    payload = make_payload(repo_root, args.base)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
