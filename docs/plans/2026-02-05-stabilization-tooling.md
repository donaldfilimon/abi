# Stabilization & Tooling Plan (2026-02-05)

**Goal:** Unblock native HTTP downloads and the toolchain CLI while keeping
examples and documentation aligned with current APIs.

**Status:** Complete. Native HTTP downloads and toolchain CLI have been
re-enabled, and examples/docs alignment is complete.

## Scope

- In: Native HTTP download support, toolchain CLI re-enable, example maintenance,
  documentation alignment.
- Out: New feature development or large refactors unrelated to tooling.

## Tasks

1. Enabled native HTTP downloads and removed manual-only fallback paths.
2. Re-enabled toolchain CLI with Zig 0.16 compatible APIs.
3. Verified example programs for API drift and updated where needed.
4. Updated planning docs (`ROADMAP.md`, `PLAN.md`, `TODO.md`) to reflect
   completion.

## Validation

- `zig build docs-site`
- `zig build test --summary all`

## References

- `ROADMAP.md`
- `PLAN.md`
- `TODO.md`
