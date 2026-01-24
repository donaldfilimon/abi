#!/usr/bin/env python3
"""
Fix double semicolons in import statements.
"""

import os
import re
import sys


def fix_file(filepath):
    """Fix double semicolons in a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace double semicolons with single semicolon in import lines
    new_content = re.sub(r'(@import\("[^"]+"\));;', r"\1;", content)

    if new_content != content:
        print(f"Fixed: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


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
        if fix_file(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
