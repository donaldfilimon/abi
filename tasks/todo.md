# Codebase Perfection & JSON to ZON Migration

## Objective
Thoroughly verify and clean the `examples/`, `tools/`, `benchmarks/`, and entire codebase, fixing any compilation errors and fully migrating all internal static JSON configuration/data files to Zig Object Notation (ZON).

## Scope
- In scope:
  - Check and fix compilation of all Zig examples in `examples/`.
  - Restore and fix the `benchmarks/` suite, integrating it into the `zig build` pipeline.
  - Migrate all documentation metadata files (`docs/data/*.json`) to `.zon`.
  - Refactor configuration parsing logic (`src/services/shared/utils/config.zig`, `src/services/tasks/persistence.zig`, `tools/cli/commands/plugins.zig`) to use Zig 0.16's `std.zon.parse.fromSliceAlloc`.
  - Fix any legacy `std.Io` or `ArrayList` API patterns flagged by the consistency checks.
  - Ensure `zig build verify-all --summary all` passes with 0 errors.

## Verification Criteria
- `zig build verify-all --summary all` completes successfully.
- `zig build examples` compiles all examples successfully.
- `zig build benchmarks` runs the benchmark suite successfully.
- `docs/data/` contains only `.zon` configuration files.

## Checklist
- [x] Restore and fix `benchmarks/` and `examples/c_test.c`.
- [x] Convert `docs/data/*.json` to `.zon` format.
- [x] Update documentation generator (`tools/gendocs/`) and static site JS to parse `.zon`.
- [x] Update runtime parsers to `std.zon.parse.fromSliceAlloc` and implement `ArenaAllocator` to fix memory leaks.
- [x] Fix legacy `std.ArrayList.init` and `std.fs.cwd()` patterns.
- [x] Fix missing variables, shadows, and formatting errors in `plugins.zig`, `generate_cli_registry.zig`, and `mod.zig`.
- [x] Run `zig build verify-all` and ensure 100% success.

## Review
- **Trigger:** User request to perfect the codebase and migrate JSON to ZON.
- **Impact:** Codebase is fully modernized to Zig 0.16, all missing examples and benchmarks are restored and building, and configuration parsing is native, avoiding external JSON parser dependencies.
- **Plan change:** Reverted python-based aggressive `sed` replacements across the codebase that broke syntax, opting for precise replacements and Arena-based memory management for ZON parsing.
- **Verification change:** Executed `zig build verify-all --summary all` until 0 errors reported.

---

## Follow-up: Review Findings Remediation (2026-03-01)

### Objective
Address the three confirmed review findings in docs data loading, CLI command docs extraction, and AI inference stub error mapping.

### Checklist
- [x] Fix docs CLI-command discovery so `docs/data/commands.zon` is populated under generated registry wiring.
- [x] Fix inference stub `get(feature)` to return feature-specific disabled errors.
- [x] Replace fragile `loadZon` regex conversion with deterministic parser handling generated ZON.
- [x] Regenerate docs data artifacts and verify drift checks.
- [x] Run targeted validation commands and record outcomes.

### Review
- **Result:** All three review findings were addressed with source fixes and regenerated docs artifacts.
- **Validation:**
  - `zig build toolchain-doctor`
  - `zig build typecheck`
  - `zig build gendocs -- --no-wasm --untracked-md`
  - `zig build gendocs -- --check --no-wasm --untracked-md`
  - `zig build check-docs`
  - `zig build check-cli-registry`
- **Outcome:** docs drift checks pass; command metadata is restored in `docs/data/commands.zon`; inference stub now returns feature-appropriate disabled errors.
