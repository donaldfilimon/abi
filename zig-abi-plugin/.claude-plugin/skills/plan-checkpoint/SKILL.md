---
name: plan-checkpoint
description: Validate plan execution progress by checking commit atomicity, verification gates, lessons.md updates, and roadmap catalog consistency
---

# Plan Checkpoint Validation

Validates that a plan's execution conforms to project workflow standards. Run this skill at the end of each plan phase or at plan completion to catch drift early.

## Inputs

- **plan_file**: Path to the plan document (e.g., `docs/plans/<slug>.md`)
- **phase_count**: Number of phases the plan defines
- **commits_since_start**: Number of commits since plan execution began (use `git log --oneline` to count)

## Step 1: Verify Commit Atomicity

Run:

```bash
git log --oneline -<N>
```

where `<N>` is the number of commits since the plan started.

Check each commit against the plan phases:

- Each commit message should map to exactly **one** plan phase.
- Flag any commit that bundles work from multiple phases (violates atomic commit convention).
- Flag any plan phase that has no corresponding commit (missed phase).
- Commit messages should follow conventional commits (`fix:`, `feat:`, `docs:`, `chore:`, `style:`, `refactor:`).

Report mapping as:

| Phase | Expected Work | Commit(s) | Atomic? |
|-------|--------------|-----------|---------|
| 1     | ...          | abc1234   | YES     |
| 2     | ...          | (none)    | MISSING |

## Step 2: Check Verification Gates

Run the strongest available verification gate. Try in order:

1. **Full check** (requires pinned Zig matching `.zigversion` on PATH):
   ```bash
   zig build full-check
   ```
2. **Format check** (always works, fallback):
   ```bash
   zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
   ```

Record which gate was used (`full-check` or `fmt`) and whether it passed or failed. If the full check fails due to Darwin linker issues (undefined symbols like `_malloc_size`), fall back to format check and note the linker limitation.

## Step 3: Validate Roadmap Catalog

Only applies if the plan modified `src/services/tasks/roadmap_catalog.zig`.

Check the file for:

1. **Valid status transitions**: Status must progress `planned` -> `in_progress` -> `done`. Never skip a state (e.g., `planned` -> `done` directly is invalid).

2. **Plan slug references**: Every slug referenced in task entries must have a corresponding entry in `plan_specs`. Run:
   ```bash
   grep -oP '\.plan\s*=\s*"([^"]+)"' src/services/tasks/roadmap_catalog.zig
   ```
   and verify each slug appears in the `plan_specs` array.

3. **`nonDoneEntryCount()` accuracy**: Count the entries that are NOT `.done` and compare against the value returned by `nonDoneEntryCount()`. They must match.

If the file was not modified, report "N/A — roadmap catalog unchanged."

## Step 4: Check lessons.md Updates

If any corrections were made during plan execution (reverted commits, failed gates that required fixes, wrong approaches that were changed), verify that `tasks/lessons.md` was updated with:

- A root cause description under the appropriate topic heading
- A prevention rule so the mistake is not repeated

To detect corrections, look for:
- Commits with `fix:` prefix that fix issues introduced during this plan
- Multiple attempts at the same phase
- Reverted or amended commits

If no corrections were needed, report "N/A — no corrections during execution."

## Step 5: Checkpoint Summary

Output a summary table:

```
## Plan Checkpoint Report

Plan: <plan name>
Branch: <current branch>
Date: <current date>

| Phase | Commit  | Gate       | Status |
|-------|---------|------------|--------|
| 1     | abc1234 | full-check | PASS   |
| 2     | def5678 | fmt        | PASS   |
| 3     | —       | —          | SKIP   |

### Gate Result
- Gate used: full-check | fmt
- Result: PASS | FAIL (details)

### Roadmap Catalog
- Modified: YES | NO
- Status transitions: VALID | INVALID (details)
- Slug references: VALID | INVALID (details)
- nonDoneEntryCount: MATCH | MISMATCH (expected X, got Y)

### Lessons
- Corrections detected: YES | NO
- lessons.md updated: YES | NO | N/A
```

## Failure Handling

- If any check fails, do NOT stop. Complete all checks and report all failures together.
- Suggest specific fixes for each failure (e.g., "Split commit abc1234 into two: one for phase 2 catalog update, one for phase 3 implementation").
- If the verification gate fails, include the error output in the report.
