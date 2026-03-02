# Task Plan - Project Stabilization & Refactor (2026-03-01)

## Scope
- Track stabilization and refactor work items that affect repository correctness and workflow policy.

## Verification Criteria
- `zig build check-workflow-orchestration-strict --summary all` passes before marking tasks complete.

## Big-Bang Strict v2 Migration
### Objective
Hard-remove legacy ABI v1 aliases and helpers and migrate all internal call sites to strict v2 API usage (`abi.App`, `abi.AppBuilder`, `abi.features.*`, `abi.services.*`).

### Baseline Snapshot
- Toolchain baseline verified and pinned:
  - `zig version` -> `0.16.0-dev.2682+02142a54d`
  - `zig build toolchain-doctor` -> pass
- Pre-migration legacy reference count: `1683`

### Checklist
- [x] Remove legacy exports/helpers from `src/abi.zig`.
- [x] Apply ordered v1->v2 codemod across `src/`, `tools/`, `examples/`, and `benchmarks/`.
- [x] Manually fix edge cases (strings, historical comments, tests expecting legacy names).
- [x] Update docs/help/examples to present v2-only entry points.
- [x] Extend `tools/scripts/check_zig_016_patterns.zig` with strict-v2 forbidden patterns.
- [x] Regenerate/check deterministic docs and CLI registry artifacts if affected.
- [x] Run full validation matrix (`typecheck`, consistency, docs, tests, `verify-all`).

### Review
- **Result:** strict-v2 migration completed in one change set. Legacy exports/helpers were removed from `src/abi.zig`, call sites were migrated to `abi.App` / `abi.features.*` / `abi.services.*`, and consistency checks now enforce v2-only usage.
- **Validation:** All checks passed (typecheck, consistency, cli-registry, docs, tests).
- **Residual risks:** External downstream forks will break until migrated.

---

## WDBX Stabilization Next Improvements
### Objective
Stabilize `db.neural`/WDBX on Zig 0.16 by removing compatibility blockers, hardening runtime correctness, and tightening compile/test gate coverage.

### Checklist
- [x] Replace legacy WDBX `std.ArrayList(...).init` usage with Zig 0.16-compatible unmanaged patterns.
- [x] Replace WDBX `std.Thread.RwLock` usage with shared compatibility lock implementation.
- [x] Add runtime config validation (`Config.validateRuntime`) and wire engine init through it.
- [x] Fix Manhattan metric handling in HNSW and engine search scoring/distance paths.
- [x] Deep-copy and free metadata ownership in `Engine` to avoid dangling external slices.
- [x] Improve cache eviction determinism and use `segments` as a real sharding/eviction dimension.
- [x] Run validation matrix (`check-consistency`, WDBX tests, `typecheck`, `full-check`, `verify-all`).

### Review
- **Result:** WDBX stabilized with 0.16 unmanaged patterns, corrected metrics, and hardened metadata ownership.
- **Validation:** All tests pass, including new cache contention tests.

---

## Follow-up: Review Findings Remediation
### Objective
Address the three confirmed review findings in docs data loading, CLI command docs extraction, and AI inference stub error mapping.

### Checklist
- [x] Fix docs CLI-command discovery so `docs/data/commands.zon` is populated.
- [x] Fix inference stub `get(feature)` to return feature-specific disabled errors.
- [x] Replace fragile `loadZon` regex conversion with deterministic parser handling generated ZON.
- [x] Regenerate docs data artifacts and verify drift checks.

### Review
- **Result:** All three review findings were addressed with source fixes and regenerated docs artifacts.
- **Validation:** docs drift checks pass; command metadata is restored; inference stub fixed.

---

## Task: Organize Zig 0.16 master files
### Plan
- [x] Centralize ZVM master Zig path helpers in `src/services/shared/utils/zig_toolchain.zig`.
- [x] Update LSP client and SPIR-V compiler bridge to reuse the shared helper.
- [x] Run targeted validation for touched modules and record outcomes.

### Review
- **Result:** Path helpers centralized; LSP and SPIR-V bridges updated.
- **Validation:** Modules touched now use the centralized helper.

---

## Task: Docs Folder Refactor & Stabilization
### Objective
Refactor `docs/index.js` to reduce duplication and improve maintainability without changing search behavior.

### Checklist
- [x] Identify repetitive patterns in docs search result construction.
- [x] Define reusable result builders/config for docs entity types.
- [x] Run a JavaScript syntax check for `docs/index.js`.
- [x] Add a short review summary with verification notes.

### Review
- **Result:** Refactor attempted; however, the state was reverted to `origin/main` to resolve merge conflicts and ensure a clean baseline.
- **Validation:** `docs/index.js` restored to upstream state.
- **Note:** `addResult(results, query, score, payload)` refactor was discarded in favor of upstream stability.

---

## Follow-up: Resolve Merge Conflicts & Normalize History (2026-03-01)
### Objective
Perform a fix-forward cleanup of the repository state after a messy merge, including marker removal and task file normalization.

### Checklist
- [x] Restore `docs/index.js` to `origin/main`.
- [x] Normalize `tasks/todo.md` (remove markers, deduplicate).
- [x] Normalize `tasks/lessons.md` (remove markers, deduplicate).
- [x] Perform final verification (no markers, build/syntax checks).
- [x] Commit fix-forward resolution.

### Review
- **Result:** Repository state normalized. Merge markers removed from task files and documentation folders.
- **Validation:** `grep` confirms no remaining markers.

---

## P0 Stabilization Pack (2026-03-02)
### Objective
Apply repository-wide stabilization: fix symbols filter, enforce strict conflict marker removal, and normalize task state.

### Checklist
- [x] Fix symbols filter in `tools/gendocs/assets/index.js`.
- [x] Add strict conflict marker enforcement to `tools/scripts/check_workflow_orchestration.zig`.
- [x] Normalize `tasks/todo.md` and `tasks/lessons.md`.
- [x] Regenerate `docs/index.js`.
- [x] Verify with `zig build check-workflow-orchestration-strict`.

### Review
- **Result:** P0 stabilization pack implemented. Checker now enforces no markers; symbols filter fixed.
- **Validation:** `check-workflow-orchestration-strict` passes.

---

## Task: Fix Current Breakage (2026-03-02)
### Objective
Identify the present failing path in the workspace, fix root cause with minimal change, and verify the repair.

### Scope
- Resolve the strict workflow-orchestration contract failure with the smallest valid `tasks/todo.md` update.

### Verification Criteria
- `zig build check-workflow-orchestration-strict --summary all` passes.

### Checklist
- [x] Run mandatory multi-CLI consensus preflight (best-effort) and continue with available outputs.
- [x] Reproduce a concrete failing check/build/test locally.
- [x] Implement a minimal root-cause fix in `tasks/todo.md` by adding required section coverage.
- [x] Re-run the failing command and relevant follow-up checks.
- [x] Document results and residual risk in a review section.

### Review
- **Result:** Added explicit top-level `Scope` and `Verification Criteria` sections to `tasks/todo.md`, which satisfies strict workflow-contract section requirements.
- **Validation:** `zig build check-workflow-orchestration-strict --summary all` passes.
