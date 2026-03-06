# Task Plan - Zig 0.16 Aggressive-5 Execution (2026-03-06)

## Objective
Apply and maintain the Aggressive-5 reprioritization across canonical catalog, generated roadmap/plan artifacts, and workflow tracking files while preserving deterministic `zig-master` validation gates.

## Scope
- Keep `zig-master` as the execution contract with exact pin `0.16.0-dev.2694+74f361a5c`.
- Treat `src/services/tasks/roadmap_catalog.zig` as the only source of truth for plan/roadmap content and status.
- Keep all plan artifacts synchronized in one wave: catalog, `docs/data/*.zon`, `docs/plans/*.md`, `docs/_docs/roadmap.md`, and `tasks/`.
- Maintain Aggressive-5 active state: CLI, Docs sync, TUI, GPU, and Feature Modules are in progress; Integration remains blocked with explicit unblock criteria.

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
- `zig build full-check`
- `zig build check-cli-registry`
- `zig build verify-all`
- `zig build check-workflow-orchestration-strict --summary all`

## Checklist
### Now
- [x] Aggressive-5 reprioritization encoded in `roadmap_catalog.zig` with no slug/ID/schema changes.
- [x] Generated artifacts refreshed from canonical source (`docs/data/plans.zon`, `docs/data/roadmap.zon`, `docs/plans/*.md`, `docs/_docs/roadmap.md`).
- [x] Active-plan profile set: 5 plans `In Progress`, 1 plan `Blocked`.
- [ ] Wave 1 (CLI framework + local-agent fallback): close descriptor/runtime parity and help/assertion drift.
- [ ] Wave 2 (TUI modular extraction): finalize module boundaries and layout/input correctness.
- [ ] Wave 3 (GPU redesign): complete strict backend policy and pool lifecycle hardening.
- [ ] Wave 5 (feature-module restructure): finish facade removal and boundary consolidation.
- [ ] Per-wave maintenance rule: after each milestone/status update, regenerate docs artifacts and update `tasks/todo.md` + `tasks/lessons.md` in the same change set.

### Next
- [ ] Wave 4 unblock path (integration gates): satisfy unblock criteria and restore deterministic `cli-tests-full`.
- [ ] Keep interim integration policy green while blocked: `cli-tests`, `tui-tests`, `ui launch --help`, `ui gpu --help`.
- [ ] Continue planned backlog without changing current horizon policy: RM-006, RM-010, RM-011, RM-012.

## Review
- Result: Aggressive-5 state is now canonical-first and synchronized across all plan surfaces.
- Validation: Full `zig-master` close-out gates are required before merge acceptance.
- Residual risk: Parallel active waves increase coordination load; missing same-wave regeneration of docs/tasks can reintroduce drift.
