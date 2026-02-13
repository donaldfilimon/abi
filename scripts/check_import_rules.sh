#!/usr/bin/env bash
#
# Verify that feature modules do not use @import("abi") in executable code.
# Feature modules (src/features/*) cannot import the top-level abi module
# because it creates a circular dependency. Doc comments (//! and ///) are
# excluded since they contain usage examples, not executable code.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

violations=0

# Search for @import("abi") in feature files, excluding doc/inline comments
grep -rn '@import("abi")' src/features/ --include="*.zig" 2>/dev/null | while IFS=: read -r file lineno content; do
    # Skip doc comments (//! and ///) and inline comments (//)
    trimmed="$(echo "$content" | sed 's/^[[:space:]]*//')"
    case "$trimmed" in
        //*) continue ;;
    esac
    echo "VIOLATION: $file:$lineno: $trimmed"
    echo "1" >> /tmp/claude/import_violations_$$
done

if [ -f /tmp/claude/import_violations_$$ ]; then
    count=$(wc -l < /tmp/claude/import_violations_$$)
    rm -f /tmp/claude/import_violations_$$
    echo ""
    echo "ERROR: Found $count @import(\"abi\") violation(s) in feature modules."
    echo "Feature modules must use relative imports to avoid circular dependencies."
    exit 1
fi

echo "OK: No @import(\"abi\") violations in feature modules."
exit 0
