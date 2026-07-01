#!/usr/bin/env bash
# auth-localcheck driver: build the abi CLI and exercise the credential-status
# surface WITHOUT storing, deleting, or transmitting anything — `auth status`
# (provider table) and `auth signin` with no service (usage banner only, stores
# nothing). Deliberately does NOT run `auth logout` (destructive: it deletes any
# stored credentials) or a real `auth signin <svc>` (would store creds).
# Asserts exit codes + output markers. Resolves repo root from own path.
#
# Usage: .claude/skills/auth-localcheck/auth.sh
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
markers() { local out="$1"; shift; for m in "$@"; do
    printf '%s' "$out" | grep -qF -- "$m" && echo "[ok] marker: $m" \
        || { echo "[FAIL] missing marker: $m"; fail=$((fail+1)); }; done; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] $ABI not produced"; exit 1; }

say "abi auth status"
out=$("$ABI" auth status 2>&1)
printf '%s\n' "$out"
markers "$out" "Authentication Status:" "OpenAI:" "Anthropic:" "Twilio:"

say "abi auth signin (usage banner — stores nothing)"
out=$("$ABI" auth signin 2>&1)
printf '%s\n' "$out"
markers "$out" "usage: abi auth signin"

say "summary"; echo "failed checks: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — auth status + signin wiring verified (no creds touched)." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
