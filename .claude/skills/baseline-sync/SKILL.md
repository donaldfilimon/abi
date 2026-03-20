---
name: baseline-sync
description: This skill should be used when the user asks to "sync baseline", "update test counts", "fix baseline drift", or when a PostToolUse hook reports "test baseline drift detected". Updates baseline.zig test counts after test runs show count drift.
---

# Baseline Sync

Synchronize the expected test pass/fail/skip counts in `tools/scripts/baseline.zig` with the actual results from the latest test runs.

## When to Use

Activate this skill when:
- A PostToolUse hook reports "test baseline drift detected"
- Tests have been added, removed, or modified
- `zig build full-check` fails due to baseline mismatch

## Procedure

### Step 1: Run both test suites and capture output

Run both commands and capture the summary lines:

```bash
zig build test --summary all 2>&1 | tail -20
```

```bash
zig build feature-tests --summary all 2>&1 | tail -20
```

If the build fails due to the Darwin linker issue, ensure a host-built Zig matching `.zigversion` (`0.16.0-dev.2934+47d2e5de9`) is on PATH. The legacy `run_build.sh` wrapper has been removed.

### Step 2: Extract counts from the summary output

The Zig test runner prints a summary line in this format:

```
<N> passed, <N> skipped, <N> failed.
```

Extract these numbers from each test suite:
- **Primary tests** (`zig build test`): pass, skip, fail counts
- **Feature tests** (`zig build feature-tests`): pass, skip, fail counts

### Step 3: Update baseline.zig

Read `tools/scripts/baseline.zig` first, then edit to update the test baseline constants. Preserve the existing `zig_version` constant exactly as-is (managed separately by version pin discipline).

Update the `primary_tests` and `feature_tests` structs with the extracted counts.

Rules:
- The `.failed` count MUST be 0. If tests are failing, fix them before updating the baseline.
- If compile errors were reported, do NOT update the baseline. Fix the errors first.
- Preserve the `zig_version` constant — it is managed by version pin changes.

### Step 4: Update approximate counts in docs

If the counts changed significantly (more than 50 tests difference), also update approximate counts in `CLAUDE.md` and `AGENTS.md` where they appear in comments like `# Primary test suite (~1290 tests)`. Use the new passed count rounded to the nearest 10.

### Step 5: Verify the update

Confirm syntactic validity and consistency:

```bash
zig ast-check tools/scripts/baseline.zig
zig build check-test-baseline 2>&1
```

If both pass, the baseline is synchronized.

## Important Notes

- Never update the baseline on a branch with known test failures. The baseline represents a green state.
- Skip count changes are normal (features may be conditionally disabled).
- Commit the updated baseline with: `chore: sync test baseline counts (primary: <N>, feature: <N>)`
