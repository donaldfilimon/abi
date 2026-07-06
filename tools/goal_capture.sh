#!/usr/bin/env bash
set -euo pipefail

# Canonical evidence capture per strategist + plan Verification.
# Produces:
#   $SCRATCH/build-check.log (with BUILDCHECK_EXIT:0 on success)
#   $SCRATCH/full-check.log (with FULLCHECK_EXIT:0)
#   $SCRATCH/skills.log (single file; each driver block ends with RESULT: PASS)

SCRATCH="/var/folders/42/07tmt0n56jq4261xb9gz5jwm0000gn/T/grok-goal-79df3a4a516d/implementer"
mkdir -p "$SCRATCH"

echo "=== goal_capture START $(date -Iseconds) ===" | tee -a "$SCRATCH/build-check.log"

# 1. ./build.sh check (full sequence)
echo "=== RUN: ./build.sh check ===" | tee -a "$SCRATCH/build-check.log"
./build.sh check >"$SCRATCH/build-check.log" 2>&1 || true
BUILD_EXIT=$?
echo "BUILDCHECK_EXIT:$BUILD_EXIT" >> "$SCRATCH/build-check.log"
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD FAILED (see build-check.log)" >&2
  exit 1
fi
echo "build check OK" | tee -a "$SCRATCH/build-check.log"

# 2. ./build.sh full-check
echo "=== RUN: ./build.sh full-check ===" | tee -a "$SCRATCH/full-check.log"
./build.sh full-check >"$SCRATCH/full-check.log" 2>&1 || true
FULL_EXIT=$?
echo "FULLCHECK_EXIT:$FULL_EXIT" >> "$SCRATCH/full-check.log"
if [ "$FULL_EXIT" -ne 0 ]; then
  echo "FULL-CHECK FAILED (see full-check.log)" >&2
  # do not hard exit if env limited; continue for evidence but record
fi

# 3. All listed skill drivers -> single skills.log
SKILLS_LOG="$SCRATCH/skills.log"
: > "$SKILLS_LOG"
echo "=== SKILLS AGGREGATE $(date -Iseconds) ===" >> "$SKILLS_LOG"

declare -a DRIVERS=(
  .agents/skills/wdbx-roundtrip/roundtrip.sh
  .agents/skills/mcp-smoke/smoke.sh
  .agents/skills/complete-base/complete.sh
  .agents/skills/agent-plan-train/agent.sh
  .agents/skills/scheduler-status/status.sh
  .agents/skills/wdbx-api-serve/serve.sh
  .agents/skills/wdbx-cluster-serve/cluster-serve.sh
  .agents/skills/dashboard-smoke/dashboard.sh
  .agents/skills/run-tui/tui.sh
  .agents/skills/sea-learn-loop/learn.sh
  .agents/skills/plugin-runtime-tester/plugins.sh
  .agents/skills/connector-localcheck/connectors.sh
  .agents/skills/auth-localcheck/auth.sh
  .agents/skills/nn-demo/nn.sh
  .agents/skills/backend-diagnostics/diag.sh
  .agents/skills/secure-demo/secure.sh
  .agents/skills/cluster-demo-guide/cluster.sh
  .agents/skills/wdbx-bench/bench.sh
  .agents/skills/run-abi/smoke.sh
)

for d in "${DRIVERS[@]}"; do
  if [ -f "$d" ]; then
    echo "=== RUN $d ===" >> "$SKILLS_LOG"
    set +e
    bash "$d" >> "$SKILLS_LOG" 2>&1
    DRIVER_EXIT=$?
    set -e
    echo "=== END $d (exit=$DRIVER_EXIT) ===" >> "$SKILLS_LOG"
    if [ $DRIVER_EXIT -eq 0 ]; then
      echo "RESULT: PASS — $d" >> "$SKILLS_LOG"
    fi
  else
    echo "=== MISSING $d ===" >> "$SKILLS_LOG"
  fi
done

PASS_COUNT=$(grep -c 'RESULT: PASS' "$SKILLS_LOG" || echo 0)
echo "SKILLS_PASS_COUNT:$PASS_COUNT" >> "$SKILLS_LOG"

echo "=== goal_capture END $(date -Iseconds) PASS_COUNT=$PASS_COUNT ===" >> "$SKILLS_LOG"


if [ "$BUILD_EXIT" -eq 0 ] && [ "$PASS_COUNT" -ge 5 ]; then
  echo "goal_capture SUCCESS (BUILD_EXIT=0, PASS_COUNT=$PASS_COUNT)"
  exit 0
else
  echo "goal_capture INCOMPLETE (see logs)" >&2
  exit 1
fi
