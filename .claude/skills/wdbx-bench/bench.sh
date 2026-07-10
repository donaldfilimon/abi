#!/usr/bin/env bash
# wdbx-bench driver: build the abi CLI, run the in-process WDBX benchmark, and
# (optionally) the full `zig build benchmarks` suite. Asserts exit codes and the
# expected output markers. Resolves the repo root from its own location.
#
# Usage:
#   .claude/skills/wdbx-bench/bench.sh [count]     # default count=50
#   .claude/skills/wdbx-bench/bench.sh --suite     # also run `zig build benchmarks`
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

SUITE=0
COUNT=50
for a in "$@"; do
    case "$a" in
        --suite) SUITE=1 ;;
        ''|*[!0-9]*) echo "usage: bench.sh [count] [--suite]" >&2; exit 2 ;;
        *) COUNT="$a" ;;
    esac
done

ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
if ./build.sh cli; then echo "[ok] build"; else echo "[FAIL] build"; exit 1; fi
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi wdbx benchmark $COUNT"
out=$("$ABI" wdbx benchmark "$COUNT" 2>&1); rc=$?
printf '%s\n' "$out"
if [ "$rc" -ne 0 ]; then echo "[FAIL] benchmark exit $rc"; fail=$((fail+1)); fi
for marker in "benchmark (local, in-memory" "inserts:" "searches:"; do
    grep -qF -- "$marker" <<<"$out" && echo "[ok] marker: $marker" \
        || { echo "[FAIL] missing marker: $marker"; fail=$((fail+1)); }
done

if [ "$SUITE" -eq 1 ]; then
    say "zig build benchmarks (full suite)"
    if zig build benchmarks; then echo "[ok] suite"; else echo "[FAIL] suite"; fail=$((fail+1)); fi
fi

say "summary"
echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — WDBX benchmark ran." || echo "RESULT: FAIL — $fail check(s) failed."
exit "$fail"
