---
name: baseline-sync
description: Tracks test pass/skip counts from test runs and reports drift from previous baselines. Use when asked to "sync baseline", "update test counts", "check test baseline", or after adding/removing tests. Covers both full test suite and 8 focused test lanes (messaging, agents, orchestration, etc.).
---

# Baseline Sync

Track expected test pass/skip counts and detect drift from actual test results.

## When to Use

Activate this skill when:
- User asks to "sync baseline", "update test counts", "check test baseline"
- Tests have been added, removed, or modified and you need to verify counts
- After a significant feature change to confirm no regressions

## Procedure

### Step 1: Run tests and capture output

On macOS 26.4+:
```bash
./build.sh test --summary all 2>&1
```

On Linux / older macOS:
```bash
zig build test --summary all 2>&1
```

The Zig test runner prints a summary line in this format:
```
<N> passed; <N> skipped; <N> failed.
```

### Step 2: Extract counts

Parse the summary line for:
- **passed**: number of passing tests
- **skipped**: number of skipped tests
- **failed**: number of failing tests (MUST be 0)

### Step 3: Compare against previous baseline

If a previous baseline is known (from a prior run or documentation), compare:
- Report if pass count changed significantly (>50 tests difference)
- Report if skip count changed
- Fail any non-zero fail count immediately

### Step 4: Record the new baseline

For tracking, record counts in a simple format:
```
<date> | passed: <N> | skipped: <N> | failed: <N> | zig: <version>
```

Store in conversation context or suggest updating relevant docs if counts changed significantly.

## Rules

- Never accept a baseline with `failed > 0`. Fix the failures first.
- Skip count changes are normal — features may be conditionally disabled.
- The `zig build doctor` step can confirm build configuration before running tests.
- Commit baseline updates with: `chore: sync test baseline counts (passed: <N>, skipped: <N>)`

## Current Baseline (2026-03-27)

```
3677 passed, 4 skipped, 0 failed (exit 0) | zig: 0.16.0-dev.2984+cb7d2b056
Build Summary: 6/6 steps succeeded
Note: macOS 26.4+ requires ./build.sh wrapper
```

## Important Notes

- There is no `tools/scripts/baseline.zig` file — baselines are tracked conversationally.
- **Zig 0.16 does NOT print "N passed; N skipped; N failed" summary lines** — it exits silently with code 0 when all pass. The `--summary all` flag controls build step summaries, not test counts. Use exit code to verify: 0 = all pass, non-zero = failure.
- `zig build feature-tests` runs integration + parity tests; `zig build test --summary all` runs both unit and integration tests.
- `zig build full-check` and `zig build verify-all` are aliases for `zig build check`. There is no `zig build check-test-baseline`.
- Focused test lanes: `zig build messaging-tests`, `agents-tests`, `multi-agent-tests`, `orchestration-tests`, `gateway-tests`, `inference-tests`, `secrets-tests`, `pitr-tests` — each runs a paired unit + integration test set for one feature.
- The Zig test runner may print "failed command" even when all tests pass — this is caused by runtime warnings (OCSP, GPU simulation, auth JWT). Always check exit code directly.
