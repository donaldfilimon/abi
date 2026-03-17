#!/bin/bash
# Audit addExecutable() calls in build.zig and check darwinRelink() wiring.
#
# On Darwin 25+ (macOS 26+), stock Zig cannot link executables due to missing
# __availability_version_check in the system stubs. Every addExecutable() must
# have a corresponding darwinRelink() call or be guarded by is_blocked_darwin.
#
# Compatible with macOS (BSD grep) and Linux (GNU grep).
set -euo pipefail

PROJECT_DIR="${1:-.}"
BUILD_FILE="$PROJECT_DIR/build.zig"

if [[ ! -f "$BUILD_FILE" ]]; then
    echo "ERROR: $BUILD_FILE not found" >&2
    exit 1
fi

echo "=== Darwin Relink Audit ==="
echo ""

# Find all addExecutable calls with their line numbers
EXEC_LINES=$(grep -n 'addExecutable(' "$BUILD_FILE" 2>/dev/null || true)
EXEC_COUNT=$(echo "$EXEC_LINES" | grep -c 'addExecutable' 2>/dev/null || echo 0)
echo "--- addExecutable() calls in build.zig ---"
echo "Found $EXEC_COUNT addExecutable() calls"
echo ""

# Find all darwinRelink calls
RELINK_COUNT=$(grep -c 'darwinRelink(' "$BUILD_FILE" 2>/dev/null || echo 0)
echo "--- darwinRelink() calls in build.zig ---"
echo "Found $RELINK_COUNT darwinRelink() calls"
echo ""

# Find is_blocked_darwin guards
GUARD_COUNT=$(grep -c 'is_blocked_darwin' "$BUILD_FILE" 2>/dev/null || echo 0)
echo "--- is_blocked_darwin guards ---"
echo "Found $GUARD_COUNT is_blocked_darwin references"
echo ""

# Check build/*.zig files too
echo "--- darwinRelink() in build/*.zig ---"
for f in "$PROJECT_DIR"/build/*.zig; do
    [[ -f "$f" ]] || continue
    CNT=$(grep -c 'darwinRelink(' "$f" 2>/dev/null || true)
    if [[ -n "$CNT" ]] && [[ "$CNT" -gt 0 ]] 2>/dev/null; then
        echo "  $(basename "$f"): $CNT call(s)"
    fi
done
echo ""

# Detailed analysis: for each addExecutable, check if it's guarded or relinked
echo "--- Per-executable analysis ---"
ISSUES=0
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    LINE_NUM="${line%%:*}"
    LINE_CONTENT="${line#*:}"

    # Extract the executable name using sed (macOS-compatible)
    EXE_NAME=$(echo "$LINE_CONTENT" | sed -n 's/.*\.name *= *"\([^"]*\)".*/\1/p' | head -1)
    if [[ -z "$EXE_NAME" ]]; then
        # Try dynamic name pattern: .name = <expr>
        EXE_NAME=$(echo "$LINE_CONTENT" | sed -n 's/.*\.name *= *\([^,}]*\).*/\1/p' | head -1)
        [[ -z "$EXE_NAME" ]] && EXE_NAME="(inline)"
    fi

    # Check surrounding context for relink or guard
    START=$((LINE_NUM > 20 ? LINE_NUM - 20 : 1))
    END_LINE=$((LINE_NUM + 30))
    CONTEXT=$(sed -n "${START},${END_LINE}p" "$BUILD_FILE")

    HAS_GUARD=false
    HAS_RELINK=false

    if echo "$CONTEXT" | grep -q 'is_blocked_darwin'; then
        HAS_GUARD=true
    fi
    if echo "$CONTEXT" | grep -q 'darwinRelink'; then
        HAS_RELINK=true
    fi

    STATUS="OK"
    if ! $HAS_GUARD && ! $HAS_RELINK; then
        STATUS="MISSING"
        ISSUES=$((ISSUES + 1))
    elif $HAS_GUARD && $HAS_RELINK; then
        STATUS="guarded+relinked"
    elif $HAS_GUARD; then
        STATUS="guarded"
    else
        STATUS="relinked"
    fi

    printf "  Line %4d: %-40s [%s]\n" "$LINE_NUM" "$EXE_NAME" "$STATUS"
done <<< "$EXEC_LINES"

echo ""
if [[ "$ISSUES" -gt 0 ]]; then
    echo "WARNING: $ISSUES executable(s) may lack Darwin relink coverage."
    echo "  Each addExecutable() should have either:"
    echo "    - A darwinRelink() call wired to its step"
    echo "    - An is_blocked_darwin guard preventing compilation on Darwin 25+"
    echo ""
    echo "  Note: Some false positives are expected for:"
    echo "    - Executables inside is_blocked_darwin conditional blocks"
    echo "    - Helper functions that are called with relink wiring elsewhere"
    exit 1
else
    echo "OK: All executables have Darwin relink or guard coverage."
fi
