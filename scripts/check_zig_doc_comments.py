#!/usr/bin/env python3
"""Validate that every tracked Zig source begins with a module doc comment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def iter_tracked_zig_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.zig"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def has_module_doc_comment(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        return line.startswith("//!")
    return False


def main() -> int:
    failures: list[Path] = []
    for path in iter_tracked_zig_files():
        if not has_module_doc_comment(path):
            failures.append(path)

    if failures:
        print("Missing module documentation comments:")
        for path in failures:
            print(f" - {path}")
        print("\nAdd a leading `//!` comment describing the file's purpose.")
        return 1

    print("All tracked Zig files start with module documentation comments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
