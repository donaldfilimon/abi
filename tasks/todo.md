# Task Plan

## Task
- Restore required `check-consistency` markers in `CLAUDE.md` while preserving normalization intent.
- Re-run consistency checks to confirm marker-related failures are resolved.
- Document remaining blocked path for archive-only toolchain alignment.

## Assumptions
- Toolchain alignment remains blocked until an internal archive path or URL is provided.
- This task updates docs/process files only.

## Execution Checklist
- [x] Review `tasks/lessons.md` before starting.
- [x] Confirm exact required marker strings from consistency scripts.
- [x] Add dedicated compliance marker block to `CLAUDE.md`.
- [x] Add matching compliance marker block to `.claude/rules/zig.md` (required by baseline checker).
- [x] Run `zig run tools/scripts/check_test_baseline_consistency.zig`.
- [x] Run `zig run tools/scripts/check_zig_version_consistency.zig`.
- [x] Run `zig build check-consistency`.
- [x] Record verification evidence and residual risks.

## Verification Evidence
- `zig run tools/scripts/check_test_baseline_consistency.zig` (before zig.md update) -> failed: missing expected markers in `.claude/rules/zig.md`.
- `zig run tools/scripts/check_zig_version_consistency.zig` (before docs update) -> failed:
  - active Zig mismatch vs `.zigversion`
  - missing Zig pin marker in `CLAUDE.md`.
- `zig build check-consistency` (before docs update) -> failed with two categories:
  - missing baseline markers
  - active Zig mismatch.
- Updated `CLAUDE.md` with required literals:
  - `0.16.0-dev.2637+6a9510c0e`
  - `1290 pass, 6 skip (1296 total)`
  - `2360 pass (2365 total)`
- Updated `.claude/rules/zig.md` with the same required literals.
- `zig run tools/scripts/check_test_baseline_consistency.zig` (after updates) -> `OK: Test baseline consistency checks passed`.
- `zig run tools/scripts/check_zig_version_consistency.zig` (after updates) -> only failure is active Zig mismatch vs `.zigversion`.
- `zig build check-consistency` (after updates) -> only failing step is `abi-check-zig-version-consistency` for active Zig mismatch.

## Review
- Task:
  - Restore consistency markers while preserving normalized docs intent.
- Scope:
  - Added compact script-validated marker blocks in `CLAUDE.md` and `.claude/rules/zig.md`.
  - Did not reintroduce broad stale narrative counts.
- Verification:
  - Baseline consistency checker passes.
  - Repo consistency gate now fails only on toolchain version mismatch.
- Risks/Follow-ups:
  - Active Zig remains `0.16.0-dev.2637+6a9510c0e` while `.zigversion` requires `0.16.0-dev.2637+6a9510c0e`.
  - Archive-only toolchain alignment remains blocked pending internal archive path/URL input.
