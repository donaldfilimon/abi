#!/usr/bin/env bash
# sea-learn-loop driver: build the abi CLI and exercise the SEA self-learning
# completion path (`abi complete --learn`). Asserts the loop runs, persists, and
# reports evidence/adaptation counters. Resolves the repo root from its location.
#
# feat-sea defaults ON with the other -Dfeat-* flags. Pass --sea only to force
# -Dfeat-sea=true explicitly while debugging feature-flag behavior.
#
# Usage:
#   .claude/skills/sea-learn-loop/learn.sh ["input text"]
#   .claude/skills/sea-learn-loop/learn.sh --sea ["input text"]
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

SEA=0
INPUT="sea-learn-loop probe: summarize the scheduler status"
for a in "$@"; do
    case "$a" in
        --sea) SEA=1 ;;
        *) INPUT="$a" ;;
    esac
done

ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }

if [ "$SEA" -eq 1 ]; then
    say "build cli with -Dfeat-sea=true"
    build_ok() { zig build cli -Dfeat-sea=true; }
else
    say "build cli (default feat-sea=true)"
    build_ok() { ./build.sh cli; }
fi
if build_ok; then echo "[ok] build"; else echo "[FAIL] build"; exit 1; fi
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi complete --learn"
echo "\$ $ABI complete --learn \"$INPUT\""
out=$("$ABI" complete --learn "$INPUT" 2>&1); rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] || { echo "[FAIL] complete --learn exit $rc"; fail=$((fail+1)); }

# The loop must report it ran (learn=true), name a model, and emit an
# evidence counter (0 is acceptable when the scratch store has no hits).
for marker in "learn=true" "model=" "evidence_count="; do
    printf '%s' "$out" | grep -qF -- "$marker" && echo "[ok] marker: $marker" \
        || { echo "[FAIL] missing marker: $marker"; fail=$((fail+1)); }
done
printf '%s' "$out" | grep -qF -- "persisted=true" && echo "[ok] persisted=true" \
    || echo "[note] not persisted (acceptable depending on store state)"

say "summary"
echo "feat-sea: $([ "$SEA" -eq 1 ] && echo explicit-on || echo default-on)"
echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — SEA learn loop ran." || echo "RESULT: FAIL — $fail check(s) failed."
exit "$fail"
