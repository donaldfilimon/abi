# Task Plan - Zig 0.16 Feature Velocity Waves (2026-03-06)

## Objective
Execute a decision-complete, wave-based Zig 0.16 roadmap with feature velocity priority while preserving deterministic gates and catalog-first planning.

## Scope
- Keep `zig-master` as the execution contract with exact pin `0.16.0-dev.2694+74f361a5c`.
- Treat `src/services/tasks/roadmap_catalog.zig` as the sole source for plans/roadmap metadata.
- Maintain only active planning interfaces in `tasks/` (`todo.md`, `lessons.md`).
- Run required gate suites at each wave boundary before status changes.

## Verification Criteria
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `zig build gendocs -- --check --no-wasm --untracked-md`
- `zig build check-docs`
- `zig build check-workflow-orchestration-strict --summary all`
- `zig build typecheck`
- `zig build cli-tests`
- `zig build tui-tests`
- `zig build feature-tests --summary all`
- `zig build check-cli-registry`
- `zig build verify-all --summary all`

## Checklist
### Now
- [x] Planning refresh: exact-pin `zig-master` contract + active-task hard trim + catalog sync completed.
- [ ] Wave 1 (CLI/AI feature velocity): descriptor-first routing completion, provider/plugin consistency, and CLI help/assertion drift cleanup.
- [ ] Wave maintenance rule: after Wave 1, update catalog statuses/milestones, regenerate `docs/data/plans.zon` + `docs/data/roadmap.zon`, and update `tasks/todo.md` + `tasks/lessons.md` in the same change set.

### Next
- [ ] Wave 2 (TUI modular completion): finalize launcher/dashboard modular extraction and input/layout correctness.
- [ ] Wave 3 (GPU redesign closure): finalize strict backend policy and pool lifecycle safety across targets.
- [ ] Wave 4 (integration gate restoration): restore deterministic `cli-tests-full` while interim gates remain green.
- [ ] Wave 5 (feature-module restructure): complete boundary cleanup and stale facade/import removal.
- [ ] Apply the same per-wave maintenance rule at every subsequent wave boundary.

## Review
- Result: Planning baseline is active and aligned to a five-wave feature-velocity sequence.
- Validation: Pending wave execution updates.
- Residual risk: Large pre-existing dirty worktree can hide wave-specific regressions without strict scoped status review per wave.
