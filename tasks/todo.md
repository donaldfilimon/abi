# Task Plan - ABI Zig 0.16 Breaking Cleanup (2026-03-06)

## Objective
Execute the approved breaking cleanup wave for the ABI Zig 0.16 codebase by simplifying build/test orchestration, hard-removing legacy compatibility surfaces, and replacing the current `ui` split architecture with one canonical shared-runtime shell entrypoint plus focused view commands.

## Scope
- Treat this as a repo-wide breaking cleanup, not a narrow `ui` fix.
- Use `[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)` as the validation contract.
- Run the mandatory multi-CLI consensus before implementation and treat surviving outputs as advisory input.
- Replace the current dirty `ui` worktree approach rather than preserving it.

## Verification Criteria
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `zig build gendocs -- --check --no-wasm --untracked-md`
- `zig build check-docs`
- `zig build typecheck`
- `zig build cli-tests`
- `zig build tui-tests`
- `zig build full-check`
- `zig build check-cli-registry`
- `zig build verify-all`
- `zig build check-workflow-orchestration-strict --summary all`

## Checklist
### Now
- [x] Toolchain pin verified against `.zigversion`.
- [x] Review existing `tasks/lessons.md` before implementation.
- [x] Prepare and run tri-CLI consensus prompt packet for this cleanup wave.
- [ ] Refactor build/test orchestration so `full-check` and `verify-all` compose only from leaf steps.
- [ ] Replace hand-maintained CLI smoke coverage with descriptor-driven generation plus a minimal safe functional allowlist.
- [ ] Make `build/test_discovery.zig` the only tracked feature-test source of truth and generate the feature-test root in build cache.
- [ ] Simplify baseline/consistency checks so generated expectations replace stale hard-coded markers.
- [ ] Remove legacy build flag aliases, compatibility namespaces, fallback paths, deprecated forwards, and `(legacy: ...)` CLI/docs messaging.
- [ ] Collapse `ui` to one canonical shell entrypoint plus focused views on the shared dashboard runtime.
- [ ] Port `ui gpu` and `ui brain` onto shared dashboard/panel contracts.
- [ ] Regenerate docs/registry artifacts only after the public cleanup lands.

### Review
- [ ] Full `zig-master` close-out sequence passes, or any remaining failure is isolated as a pre-existing flake with evidence.
- [ ] `tasks/lessons.md` captures any new correction-driven rule discovered during execution.

---

# Task Plan - Integrate Dirty Worktrees Into Main (2026-03-06)

## Objective
Preserve the dirty local work from the main worktree on an integration branch, validate the meaningful Zig/UI/build changes under the `zig-master` workflow, and merge only the real surviving work back into `main`.

## Scope
- Capture the current dirty `main` worktree state on `integrate-all-work` instead of mutating `main` in place.
- Treat local no-op branches as already merged unless they gain new commits.
- Keep the reviewed stash separate unless it is explicitly recovered later.
- Record repo-local improvements separately from any environment-level Zig toolchain blocker.

## Verification Criteria
- `git fetch --all --prune`
- `git -C /Users/donaldfilimon/abi status --short --branch`
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `zig build v3-lib`
- `git diff --check`

## Checklist
- [x] Review `tasks/lessons.md` before implementation.
- [x] Run mandatory multi-CLI consensus for the integration question.
- [x] Verify `origin/main` and local side branches are still aligned.
- [x] Preserve the dirty `main` worktree on `integrate-all-work`.
- [x] Validate the dirty `main` worktree changes and isolate remaining blockers.
- [x] Commit the recovered integration branch changes with verification evidence.
- [x] Merge the finalized integration branch back into `main`.

## Review
- [x] Distinguish repo-local fixes from external/toolchain blockers with evidence.
- [x] Confirm the final `main` ref contains every meaningful local branch/worktree change.

---

# Task Plan - Repair Main Documentation Integrity (2026-03-06)

## Objective
Clean up the committed markdown merge artifacts on `main`, restore the most important broken documentation links and command examples, and leave the repository in a reviewable state without pulling stale stash content back in.

## Scope
- Treat the old stash and side worktree as historical inputs only; do not reintroduce obsolete doc content just to satisfy the earlier review.
- Remove committed conflict markers from tracked markdown files.
- Repair the highest-signal broken links and commands in root docs.
- Add minimal canonical top-level guidance files when current docs link to files that do not exist.

## Verification Criteria
- `rg -n '^(<<<<<<<|=======|>>>>>>>)' . --glob '*.md' --glob '*.zig'`
- `git diff --check`
- `test -f CONTRIBUTING.md`
- `test -f CLAUDE.md`

## Checklist
- [x] Confirm `main` already contains all unique local branch/worktree commits.
- [x] Remove committed conflict markers from tracked markdown files.
- [x] Repair broken root documentation links and command paths.
- [x] Add minimal top-level guidance files required by live references.
- [x] Re-run markdown integrity checks and record any remaining risk.

## Review
- [x] No tracked markdown files contain merge conflict markers.
- [x] Root documentation no longer points at missing high-signal files or obviously obsolete command paths.
- [x] Remaining `zig build` validation blocker is external to the repo and should be reported as residual risk.

---

# Task Plan - Canonicalize WDBX + Persona Architecture (2026-03-06)

## Objective
Introduce wave-1 canonical internal Zig APIs for semantic storage, coordination, and profiles so the repo describes the memory/persona architecture with neutral technical contracts while preserving the branded WDBX/persona surfaces as compatibility aliases.

## Scope
- Keep `abi.features` and `abi.services` as the v2 namespace roots.
- Add canonical `semantic_store`, `coordination`, and `profiles` module surfaces.
- Preserve `abi.features.database.wdbx` and `abi.features.ai.personas` as compatibility aliases for one wave.
- Add provenance/influence-trace contracts and route AI memory through canonical retrieval metadata where feasible without breaking callers.
- Update canonical architecture docs and README language to present WDBX/Abbey/Aviva/Abi as aliases over the technical model.

## Verification Criteria
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `git diff --check`
- `rg -n "semantic_store|coordination|profiles|InfluenceTrace|BehaviorProfile" src README.md docs`
- `zig build gendocs -- --check --no-wasm --untracked-md`
- `zig build check-docs`
- `zig build typecheck`
- `zig build cli-tests`
- `zig build tui-tests`
- `zig build full-check`
- `zig build check-cli-registry`
- `zig build verify-all`
- `zig build check-workflow-orchestration-strict --summary all`

## Checklist
- [x] Review `tasks/lessons.md` before implementation.
- [x] Run mandatory multi-CLI consensus prompt for the wave-1 cut.
- [ ] Add canonical database `semantic_store` module and compatibility aliases.
- [ ] Add canonical AI `coordination` and `profiles` modules and compatibility aliases.
- [ ] Introduce provenance/influence-trace types and wire AI memory retrieval to canonical metadata where safe.
- [ ] Update public exports/docs to point at the canonical surfaces first.
- [ ] Add compile-level parity coverage for old and new import paths.
- [ ] Re-run available validation gates and isolate the external toolchain blocker with evidence.

## Review
- [ ] New canonical imports compile and old branded imports still compile.
- [ ] The refactor introduces no new top-level feature roots or build flags.
- [ ] Any remaining validation failures are either repo-local regressions with evidence or the known external Darwin/libc linker blocker.
