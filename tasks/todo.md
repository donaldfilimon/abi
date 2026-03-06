# Task Tracker

This file is the canonical execution tracker for active ABI work.
Active and in-progress plans stay at the top. Completed plans move into the
archive section with their evidence preserved.

## Active Queue

### Task Plan - Docs + Assistant Canonical Sync Around `zig-master` (2026-03-06)

#### Objective
Align the repo workflow contract, Zig validation policy, assistant-facing docs,
todo/status markdown, and generated docs around one canonical model:
`AGENTS.md` for repo workflow, `[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)`
for Zig validation, `tasks/todo.md` for active execution, and
`tasks/lessons.md` for correction-only lessons.

#### Scope
- Update `src/services/tasks/roadmap_catalog.zig` first for plan/roadmap metadata
  changes that flow into generated docs.
- Update generator/template sources instead of hand-editing repeated generated
  markdown across `docs/_docs/`, `docs/api/`, and `docs/plans/`.
- Normalize repo-written status/docs surfaces so `TODO.md`, `PLAN.md`, and
  `ROADMAP.md` stop competing with `tasks/todo.md`.
- Update `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `CONTRIBUTING.md`, `README.md`,
  `docs/ABI_PROJECT_MEMORY.md`, and the external `zig-master` skill so they
  agree on ownership and routing.
- Fold standalone task-plan duplication under `tasks/` into this file or archive
  it as a non-canonical artifact.

#### Verification Criteria
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `zig build gendocs -- --no-wasm --untracked-md`
- `zig build gendocs -- --check --no-wasm --untracked-md`
- `zig build check-docs`
- `zig build typecheck`
- `zig build cli-tests`
- `zig build tui-tests`
- `zig build full-check`
- `zig build check-cli-registry`
- `zig build verify-all`
- `zig build check-workflow-orchestration-strict --summary all`

#### Checklist
##### Now
- [x] Review `tasks/lessons.md` before implementation.
- [x] Confirm active Zig matches `.zigversion`.
- [x] Run mandatory multi-CLI consensus for the doc-sync implementation order and
  capture surviving outputs.
- [x] Reshape `tasks/todo.md` into active-first tracking with an archive section.
- [x] Update `src/services/tasks/roadmap_catalog.zig` first for any generated
  plan/roadmap wording or status changes in this wave.
- [x] Align `tools/gendocs/model.zig`, `tools/gendocs/render_guides_md.zig`, and
  the contributing/plans templates with the canonical workflow and `zig-master`
  wording.
- [x] Update handwritten assistant/status docs so `AGENTS.md` and
  `[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)` are
  the canonical workflow and Zig validation entrypoints.
- [x] Fold `tasks/breaking_cleanup_plan.md` into this tracker or archive it so
  it no longer competes as an active checklist.
- [ ] Regenerate the affected docs outputs with `--no-wasm --untracked-md`
  once the local Zig/macOS linker blocker is resolved.

##### Review
- [ ] Generated docs reflect the updated workflow and `zig-master` contract
  without hand-edited drift.
- [x] No handwritten markdown surface still routes Zig policy through stale `.claude`
  policy paths or through `CLAUDE.md` as the canonical policy source.
- [x] `TODO.md`, `PLAN.md`, and `ROADMAP.md` behave as summary/index surfaces,
  not competing execution trackers.
- [ ] Full `zig-master` close-out passes, or every remaining failure is isolated
  with evidence.

##### Evidence
- `which zig`, `zig version`, and `cat .zigversion` all report
  `0.16.0-dev.2694+74f361a5c`.
- Mandatory tri-CLI consensus was run for the implementation order; only the
  OpenCode lane returned a usable answer, and its fallback recommendation was
  followed alongside local repo inspection.
- `rg -n '\\.claude/rules/zig\\.md'` returns no matches in the repository after
  the handwritten-doc updates.
- `zig build gendocs -- --no-wasm --untracked-md` fails before docs generation
  with unresolved Darwin/libc symbols such as `__availability_version_check`,
  `_abort`, and `_arc4random_buf`.
- Re-running the same command with
  `SDKROOT=$(xcrun --show-sdk-path) zig build gendocs -- --no-wasm --untracked-md`
  fails with the same undefined-symbol set, so docs regeneration is currently
  blocked by the local Zig/macOS linker environment rather than by repo-local
  markdown changes.
- `zig build gendocs -- --check --no-wasm --untracked-md` and
  `zig build check-docs` fail with the same unresolved Darwin/libc symbols,
  which blocks the normal docs close-out sequence before generated outputs can
  be refreshed.

##### Residual Risk
- Generated docs under `docs/_docs/`, `docs/api/`, and `docs/plans/` still
  reflect the pre-sync wording until the Darwin linker blocker is resolved and
  `zig build gendocs` can run successfully.

### Task Plan - ABI Zig 0.16 Breaking Cleanup (2026-03-06)

#### Objective
Execute the approved breaking cleanup wave for the ABI Zig 0.16 codebase by
simplifying build/test orchestration, hard-removing legacy compatibility
surfaces, and replacing the current `ui` split architecture with one canonical
shared-runtime shell entrypoint plus focused view commands.

#### Scope
- Treat this as a repo-wide breaking cleanup, not a narrow `ui` fix.
- Use `[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)` as
  the validation contract.
- Run the mandatory multi-CLI consensus before implementation and treat
  surviving outputs as advisory input.
- Keep this file as the canonical active checklist; do not rely on
  `tasks/breaking_cleanup_plan.md` as a second live tracker.

#### Verification Criteria
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

#### Checklist
##### Now
- [x] Toolchain pin verified against `.zigversion`.
- [x] Review existing `tasks/lessons.md` before implementation.
- [x] Prepare and run tri-CLI consensus prompt packet for this cleanup wave.
- [ ] Refactor `full-check` and `verify-all` so they compose only from leaf
  steps.
- [ ] Replace hand-maintained CLI smoke coverage with descriptor-driven
  generation plus a minimal safe functional allowlist.
- [ ] Make `build/test_discovery.zig` the only tracked feature-test source of
  truth and generate the feature-test root in build cache.
- [ ] Simplify baseline and consistency checks so generated expectations replace
  stale hard-coded markers.
- [ ] Remove legacy build flag aliases, compatibility namespaces, fallback
  paths, deprecated forwards, and `(legacy: ...)` CLI/docs messaging.
- [ ] Collapse `ui` to one canonical shell entrypoint plus focused views on the
  shared dashboard runtime.
- [ ] Port `ui gpu` and `ui brain` onto shared dashboard/panel contracts.
- [ ] Regenerate docs and registry artifacts only after the public cleanup lands.

##### Review
- [ ] Full `zig-master` close-out sequence passes, or any remaining failure is
  isolated as a pre-existing flake with evidence.
- [ ] `tasks/lessons.md` captures any new correction-driven rule discovered
  during execution.

## Archive

### Completed - Canonicalize WDBX + Persona Architecture (2026-03-06)

#### Objective
Refactor the branded WDBX/persona surfaces into canonical neutral internal APIs
for semantic store, coordination, profiles, and provenance while preserving
compatibility aliases for the current release wave.

#### Evidence
- Added canonical semantic-store, coordination, and profiles surfaces while
  keeping compatibility aliases in place for the release wave.
- Threaded provenance and retrieval-hit metadata through AI memory retrieval
  flows.
- Rewrote high-signal docs so neutral technical APIs are canonical and branded
  terms are glossary aliases.
- Available `zig-master` validation passed or blockers were isolated during the
  wave.

#### Residual Risk
- No correction-driven lesson was added for this wave; if a later review finds a
  repeatable planning or compatibility miss, capture it in `tasks/lessons.md`.

### Completed - Integrate Worktree And Branches Into `main` (2026-03-06)

#### Objective
Preserve local worktree state, determine which branches still carry unique work,
and merge actual outstanding work into `main` without losing state.

#### Evidence
- Inspected worktrees, local branches, and merge-base relationships before
  moving `main`.
- Ran mandatory multi-CLI consensus for the merge approach.
- Confirmed the relevant work was already represented in `main`, so no separate
  preservation branch was needed before final cleanup.
- Follow-on branch cleanup work archived below closed the remaining local and
  remote branch state.

#### Residual Risk
- None beyond the preserved global stash tracked in the branch-cleanup archive
  entry.

### Completed - Investigate `build.zig:465` `addStaticLibrary` Breakage (2026-03-06)

#### Objective
Identify the minimal Zig 0.16 migration needed for the `build.zig:465` blocker
where `b.addStaticLibrary` fails under the repo-pinned toolchain, without
modifying tracked repository source.

#### Evidence
- `zig build v3-lib` on the original source failed at `build.zig:465:21` with
  `error: no field or member function named 'addStaticLibrary' in 'Build'`.
- Local stdlib inspection showed `std.Build.addLibrary` at
  `/Users/donaldfilimon/.zvm/master/lib/std/Build.zig:839` and
  `LibraryOptions.linkage` at
  `/Users/donaldfilimon/.zvm/master/lib/std/Build.zig:820-823`, with no
  `addStaticLibrary` symbol present.
- The same `build.zig` already used the Zig 0.16 pattern at `build.zig:448-452`:
  `b.addLibrary(.{ ... .linkage = .static, ... })`.
- A temporary in-place substitution of `build.zig:465-472` from
  `b.addStaticLibrary(.{ ... })` to
  `b.addLibrary(.{ ... .linkage = .static, ... })` moved `zig build v3-lib`
  past the method-missing failure and into unrelated Darwin linker errors.

#### Residual Risk
- This isolated the first Zig 0.16 source migration needed at `build.zig:465`;
  additional blockers remain after that point, but they are no longer
  `addStaticLibrary` API errors.

### Completed - Merge Remaining Branches Into `main` And Prune Useless Refs (2026-03-06)

#### Objective
Ensure `main` contains all surviving branch work, then remove local and remote
branches that are already fully merged and no longer needed.

#### Evidence
- `git branch --merged main` showed all non-`main` local branches were already
  ancestors of `main`, so no additional merge into `main` was required.
- Mandatory consensus output under `tasks/branch-cleanup-consensus-out/`
  supported worktree-first cleanup followed by local and remote pruning.
- `git worktree remove /Users/donaldfilimon/.codex/worktrees/9a24/abi`
  succeeded, then merged local branches were deleted.
- `git fetch --prune origin` removed the stale remote-tracking ref, and
  `git ls-remote --heads origin` confirmed only `refs/heads/main` remains
  remotely.
- Final verification showed only local `main`, remote `origin/main`, and the
  primary worktree remained.

#### Residual Risk
- A global stash remains as
  `stash@{0}: On dev: pre-merge major-rewrite: local dev changes`. It was
  intentionally preserved because it is not a branch and deleting it was outside
  that request.
