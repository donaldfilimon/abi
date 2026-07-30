#!/usr/bin/env bash
# wdbx-bench driver: build the abi CLI and run the in-process WDBX benchmark,
# asserting exit codes and the expected output markers. Resolves the repo root
# from its own location.
#
# Usage:
#   .claude/skills/wdbx-bench/bench.sh [count]     # default count=50
#
# There is no full-suite mode: the Zig `zig build benchmarks` step had no Rust
# successor (the workspace declares no [[bench]] targets), so the flag was
# removed rather than left pointing at something that cannot run.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

COUNT=50
for a in "$@"; do
    case "$a" in
        ''|*[!0-9]*) echo "usage: bench.sh [count]" >&2; exit 2 ;;
        *) COUNT="$a" ;;
    esac
done

ABI="$REPO_ROOT/target/debug/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

say "build cli"
if ./tools/cargo.sh build -p abi-cli; then echo "[ok] build"; else echo "[FAIL] build"; exit 1; fi
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi wdbx benchmark $COUNT"
out=$("$ABI" wdbx benchmark "$COUNT" 2>&1); rc=$?
printf '%s\n' "$out"
if [ "$rc" -ne 0 ]; then echo "[FAIL] benchmark exit $rc"; fail=$((fail+1)); fi
for marker in "benchmark (local, in-memory" "inserts:" "searches:"; do
    grep -qF -- "$marker" <<<"$out" && echo "[ok] marker: $marker" \
        || { echo "[FAIL] missing marker: $marker"; fail=$((fail+1)); }
done

say "summary"
echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — WDBX benchmark ran." || echo "RESULT: FAIL — $fail check(s) failed."
exit "$fail"
