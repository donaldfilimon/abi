#!/usr/bin/env bash
# connector-localcheck driver: build the abi CLI and exercise the connector
# surfaces that run fully LOCALLY — the Twilio ConversationRelay simulation and
# the auth-status report. No credentials, no network, no `.live` transport.
#
# Usage: .claude/skills/connector-localcheck/connectors.sh ["caller utterance"]
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
ABI="$REPO_ROOT/zig-out/bin/abi"
UTTER="${1:-I need account help}"
fail=0
say() { printf '\n=== %s ===\n' "$*"; }
check() { local label="$1" expect="$2"; shift 2; local out; out=$("$@" 2>&1)
    printf '%s\n' "$out"
    printf '%s' "$out" | grep -qF -- "$expect" && echo "[ok] $label" || { echo "[FAIL] $label (missing: $expect)"; fail=$((fail+1)); }; }

say "build cli"
./build.sh cli >/dev/null 2>&1 && echo "[ok] build" || { echo "[FAIL] build"; exit 1; }
[ -x "$ABI" ] || { echo "[FAIL] no binary"; exit 1; }

check "twilio simulate (local)" "Twilio ConversationRelay simulation" "$ABI" twilio simulate "$UTTER"
check "twilio response line"     "response:"                            "$ABI" twilio simulate "$UTTER"
check "twilio escalation line"   "escalation:"                          "$ABI" twilio simulate "$UTTER"
check "auth status (local)"      "Authentication Status:"               "$ABI" auth status

say "summary"; echo "failed: $fail"
[ "$fail" -eq 0 ] && echo "RESULT: PASS — local connector surfaces verified." || echo "RESULT: FAIL — $fail check(s)."
exit "$fail"
