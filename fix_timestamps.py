#!/usr/bin/env python3
"""
Fix std.time.timestamp() usage in Zig 0.16 for ABI framework.
Replaces std.time.timestamp() with time.unixSeconds() and adds time import.
"""

import os
import re
import sys


def fix_file(filepath):
    """Fix timestamp usage in a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if file already has time import
    has_time_import = "const time = @import(" in content

    # Replace std.time.timestamp() with time.unixSeconds()
    new_content = re.sub(r"std\.time\.timestamp\(\)", "time.unixSeconds()", content)

    # If we replaced something and don't have time import, add it
    if new_content != content and not has_time_import:
        # Find where to add the import (after std import)
        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if 'const std = @import("std")' in line:
                # Look for next import or add after std import
                insert_pos = i + 1
                # Find where imports end
                while (
                    insert_pos < len(lines)
                    and lines[insert_pos].strip().startswith("const")
                    and (
                        "@import" in lines[insert_pos]
                        or lines[insert_pos].strip().startswith("pub const")
                    )
                ):
                    insert_pos += 1

                # Add time import
                time_import = 'const time = @import("../time.zig");'
                if "src/shared/security/" in filepath:
                    time_import = 'const time = @import("../time.zig");'
                elif "src/ai/" in filepath:
                    time_import = 'const time = @import("../../shared/time.zig");'
                elif "src/cloud/" in filepath:
                    time_import = 'const time = @import("../shared/time.zig");'
                elif "src/database/" in filepath:
                    time_import = 'const time = @import("../shared/time.zig");'
                elif "src/web/" in filepath:
                    time_import = 'const time = @import("../shared/time.zig");'

                lines.insert(insert_pos, time_import)
                new_content = "\n".join(lines)
                break

    if new_content != content:
        print(f"Fixed: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def main():
    # Find all Zig files in src/ except migration_backup
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
        if fix_file(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files.")

    # Also fix probe_error.zig in root
    probe_file = os.path.join(os.path.dirname(__file__), "probe_error.zig")
    if os.path.exists(probe_file):
        if fix_file(probe_file):
            print(f"Fixed: {probe_file}")
            fixed_count += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
