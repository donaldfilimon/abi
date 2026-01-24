#!/usr/bin/env python3
"""
Fix incorrect time import paths in Zig files.
"""

import os
import re
import sys


def get_correct_import_path(filepath):
    """Get correct import path for time.zig based on file location."""
    rel_path = os.path.relpath(filepath, os.path.dirname(__file__))

    if "src/ai/personas/" in filepath.replace("\\", "/"):
        # From src/ai/personas/* to src/shared/time.zig
        return "../../shared/time.zig"
    elif "src/ai/orchestration/" in filepath.replace("\\", "/"):
        # From src/ai/orchestration/ to src/shared/time.zig
        return "../../shared/time.zig"
    elif "src/ai/personas/embeddings/" in filepath.replace("\\", "/"):
        # From src/ai/personas/embeddings/ to src/shared/time.zig
        return "../../../shared/time.zig"
    elif "src/ai/personas/aviva/" in filepath.replace("\\", "/"):
        # From src/ai/personas/aviva/ to src/shared/time.zig
        return "../../shared/time.zig"
    elif "src/ai/personas/abi/" in filepath.replace("\\", "/"):
        # From src/ai/personas/abi/ to src/shared/time.zig
        return "../../shared/time.zig"
    elif "src/ai/training/" in filepath.replace("\\", "/"):
        # From src/ai/training/ to src/shared/time.zig
        return "../../../shared/time.zig"
    elif "src/cloud/" in filepath.replace("\\", "/"):
        # From src/cloud/ to src/shared/time.zig
        return "../shared/time.zig"
    elif "src/database/" in filepath.replace("\\", "/"):
        # From src/database/ to src/shared/time.zig
        return "../shared/time.zig"
    elif "src/web/" in filepath.replace("\\", "/"):
        # From src/web/ to src/shared/time.zig
        return "../shared/time.zig"
    elif "src/shared/security/" in filepath.replace("\\", "/"):
        # From src/shared/security/ to src/shared/time.zig
        return "../time.zig"
    else:
        print(f"WARNING: Unknown location for {filepath}")
        return None


def fix_import_paths(filepath):
    """Fix time import paths in a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all time imports
    time_imports = re.findall(r'const\s+time\s*=\s*@import\("[^"]+"\)', content)

    if not time_imports:
        return False

    new_content = content
    fixed = False

    for import_line in time_imports:
        # Get the quoted import path
        match = re.search(r'@import\("([^"]+)"\)', import_line)
        if match:
            old_path = match.group(1)
            correct_path = get_correct_import_path(filepath)

            if correct_path and old_path != correct_path:
                new_import = f'const time = @import("{correct_path}");'
                new_content = new_content.replace(import_line, new_import)
                fixed = True
                print(f"Fixed {filepath}: {old_path} -> {correct_path}")

    if fixed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

    return fixed


def main():
    src_dir = os.path.join(os.path.dirname(__file__), "src")

    zig_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip migration_backup directories
        dirs[:] = [d for d in dirs if "migration_backup" not in d]
        for file in files:
            if file.endswith(".zig"):
                zig_files.append(os.path.join(root, file))

    fixed_count = 0
    for filepath in zig_files:
        if fix_import_paths(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
