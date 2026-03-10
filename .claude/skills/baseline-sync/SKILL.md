---
name: baseline-sync
description: Update baseline.zig test counts after test runs show count drift
---

# Baseline Sync

Synchronize the expected test pass/fail/skip counts in `tools/scripts/baseline.zig`
with the actual results from the latest test runs.

## When to use

Run this skill whenever:
- A PostToolUse hook reports "test baseline drift detected"
- You have added, removed, or modified tests
- `zig build full-check` fails due to baseline mismatch

## Procedure

### Step 1: Run both test suites and capture output

Run both commands and carefully capture the summary lines:

```bash
zig build test --summary all 2>&1 | tail -20
```

```bash
zig build feature-tests --summary all 2>&1 | tail -20
```

If the build fails due to the Darwin linker issue, use the wrapper:

```bash
./tools/scripts/run_build.sh test --summary all 2>&1 | tail -20
```

```bash
./tools/scripts/run_build.sh feature-tests --summary all 2>&1 | tail -20
```

### Step 2: Extract counts from the summary output

The Zig test runner prints a summary line in this format:

```
<N> passed, <N> skipped, <N> failed.
```

or for compile errors:

```
<N> passed, <N> skipped, <N> failed, <N> compile errors.
```

Extract these three (or four) numbers from EACH test suite:
- **Primary tests** (`zig build test`): pass, skip, fail counts
- **Feature tests** (`zig build feature-tests`): pass, skip, fail counts

### Step 3: Update `tools/scripts/baseline.zig`

Edit `tools/scripts/baseline.zig` to include the extracted counts. The file must
keep the existing `zig_version` constant and add (or update) the test baseline
constants. The canonical format is:

```zig
// Canonical Zig version pin for this repo. Must match .zigversion and build.zig.zon.
// Used by check_zig_version_consistency.zig and gendocs.
pub const zig_version = "0.16.0-dev.1503+738d2be9d";

// Test baseline counts — updated by /baseline-sync skill.
// These reflect the last verified test run on the main branch.
pub const primary_tests = .{
    .passed = <N>,
    .skipped = <N>,
    .failed = 0,
};

pub const feature_tests = .{
    .passed = <N>,
    .skipped = <N>,
    .failed = 0,
};
```

Replace each `<N>` with the actual count from Step 2.

Rules:
- The `.failed` count MUST be 0. If tests are failing, fix them before updating
  the baseline. Do NOT record a nonzero fail count.
- If compile errors were reported, do NOT update the baseline. Fix the errors first.
- Preserve the `zig_version` constant exactly as-is (it is managed separately).

### Step 4: Update CLAUDE.md approximate counts

If the counts have changed significantly (more than 50 tests difference), also
update the approximate counts in `CLAUDE.md` and `AGENTS.md` where they appear
in comments like `# Primary test suite (~1290 tests)`. Use the new passed count
rounded to the nearest 10.

### Step 5: Verify the update

After editing, confirm the file is syntactically valid:

```bash
zig ast-check tools/scripts/baseline.zig
```

Then re-run the consistency check:

```bash
zig build check-test-baseline 2>&1
```

If both pass, the baseline is synchronized.

## Important notes

- Never update the baseline on a branch with known test failures. The baseline
  represents a green state.
- If skip counts change, that is normal (features may be conditionally disabled).
- Commit the updated baseline with a message like:
  `chore: sync test baseline counts (primary: <N>, feature: <N>)`
