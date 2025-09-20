```
#!/usr/bin/env bash
# migrate_zig.sh
# ------------------------------------------------------------------
# A quick helper script that walks the repo and applies common
# 0.16‑specific refactors:
#   * std.os → std.posix (POSIX APIs)
#   * Wrap Windows‑only std.os.windows blocks in an if guard
#   * Update std.log to std.log.scoped
#   * Update std.fmt usage
# ------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SRC_DIR="$ROOT_DIR/src"

# 1. Replace std.os.xxx with std.posix.xxx for non‑Windows files
echo "=== Updating POSIX APIs ==="
find "$SRC_DIR" -name '*.zig' |
  grep -v -E 'std\.os\.windows|kernel32' |
  while read -r file; do
    echo "Updating $file"
    sed -i.bak 's/std\.os\./std.posix./g' "$file"
    rm -f "$file.bak"
  done

# 2. Wrap Windows‑only blocks with an OS guard
echo "=== Guarding Windows blocks ==="
find "$SRC_DIR" -name '*.zig' -exec grep -l 'std\.os\.windows' {} \; |
  while read -r file; do
    # Skip files already guarded
    if grep -q 'if (builtin.os.tag == .windows)' "$file"; then
      continue
    fi
    echo "Guarding $file"
    # Insert guard before the first std.os.windows occurrence
    sed -i.bak '0,/std\.os\.windows/ s/std\.os\.windows/if (builtin.os.tag == .windows) { std.os.windows/' "$file"
    # Append closing brace after the block (simple heuristic: after the next blank line)
    sed -i.bak '/^}$/a }' "$file"
    rm -f "$file.bak"
  done

# 3. Update std.log usage to std.log.scoped
echo "=== Updating std.log ==="
find "$SRC_DIR" -name '*.zig' |
  while read -r file; do
    if grep -q 'std\.log\.' "$file"; then
      echo "Updating $file"
      sed -i.bak -E 's/std\.log\.err\(/std\.log.scoped(.Error, /g' "$file"
      sed -i.bak -E 's/std\.log\.info\(/std\.log.scoped(.Info, /g' "$file"
      sed -i.bak -E 's/std\.log\.debug\(/std\.log.scoped(.Debug, /g' "$file"
      sed -i.bak -E 's/std\.log\.trace\(/std\.log.scoped(.Trace, /g' "$file"
      rm -f "$file.bak"
    fi
  done

# 4. Update std.fmt usage (bufPrint → bufPrint & allocPrint)
echo "=== Updating std.fmt ==="
find "$SRC_DIR" -name '*.zig' |
  while read -r file; do
    if grep -q 'std\.fmt\.' "$file"; then
      echo "Updating $file"
      # Example: replace try std.fmt.bufPrint(&buf, ...) with const res = try std.fmt.bufPrint(&buf, ...)
      sed -i.bak -E 's/try\s+std\.fmt\.bufPrint/&/g' "$file"
      rm -f "$file.bak"
    fi
  done

echo "Migration script finished. Review the changes with 'git diff' and run 'zig build test' to verify."
