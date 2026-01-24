#!/usr/bin/env python3
"""
Fix all time import paths to point to src/shared/time.zig.
"""

import os
import re
import sys


def calculate_relative_path(from_path, to_path):
    """Calculate relative path from one file to another."""
    from_dir = os.path.dirname(from_path)

    # Normalize paths
    from_dir = os.path.normpath(from_dir)
    to_path = os.path.normpath(to_path)

    # Get relative path
    rel_path = os.path.relpath(to_path, from_dir)

    # Convert Windows paths to forward slashes for Zig imports
    rel_path = rel_path.replace("\\", "/")

    return rel_path


def fix_file_time_import(filepath):
    """Fix time import path in a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find time imports
    time_imports = re.findall(
        r'const\s+time\s*=\s*@import\("[^"]*time[^"]*"\)', content
    )

    if not time_imports:
        return False

    # Target time.zig is at src/shared/time.zig
    time_target = os.path.join(os.path.dirname(__file__), "src", "shared", "time.zig")

    new_content = content
    fixed = False

    for import_line in time_imports:
        # Extract the path inside quotes
        match = re.search(r'@import\("([^"]+)"\)', import_line)
        if not match:
            continue

        old_path = match.group(1)

        # Calculate correct relative path
        correct_rel_path = calculate_relative_path(filepath, time_target)

        if old_path != correct_rel_path:
            # Replace old path with new path
            new_import_line = import_line.replace(old_path, correct_rel_path)
            new_content = new_content.replace(import_line, new_import_line)
            fixed = True
            print(f"Fixed {filepath}: {old_path} -> {correct_rel_path}")

    if fixed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

    return fixed


def main():
    src_dir = os.path.join(os.path.dirname(__file__), "src")

    zig_files = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if "migration_backup" not in d]
        for file in files:
            if file.endswith(".zig"):
                zig_files.append(os.path.join(root, file))

    fixed_count = 0
    for filepath in zig_files:
        if fix_file_time_import(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
