# Task Plan - Canonicalize WDBX + Persona Architecture (2026-03-06)

## Objective
Refactor the branded WDBX/persona surfaces into canonical neutral internal APIs for semantic store, coordination, profiles, and provenance while preserving compatibility aliases for the current release wave.

## Scope
- Add canonical `abi.features.database.semantic_store`.
- Add canonical `abi.features.ai.coordination` and `abi.features.ai.profiles`.
- Preserve existing `wdbx` and persona-facing module paths as compatibility aliases in this wave.
- Thread provenance and retrieval metadata through AI memory surfaces.
- Update public docs so neutral technical terms are canonical and branded names are glossary aliases.

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
- [x] Review `tasks/lessons.md` before implementation.
- [x] Confirm active Zig matches `.zigversion`.
- [x] Run mandatory multi-CLI consensus for the refactor packet and capture surviving outputs.
- [x] Add canonical semantic-store module and compatibility alias the existing WDBX surface.
- [x] Add canonical AI profiles and coordination modules with temporary persona compatibility.
- [x] Thread provenance and retrieval-hit metadata through AI memory retrieval surfaces.
- [x] Expose canonical namespaces from public AI/database exports and keep compatibility aliases.
- [x] Rewrite high-signal docs to make neutral technical APIs canonical and branded terms glossary-only.

### Review
- [x] Available `zig-master` validation passes, or each blocker is isolated with evidence.
- [ ] `tasks/lessons.md` is updated if this wave uncovers a new correction-driven rule.

---

# Task Plan - Integrate Worktree And Branches Into `main` (2026-03-06)

## Objective
Preserve the uncommitted work in this worktree, determine which local branches or worktrees still carry unique commits, and merge the actual outstanding work into `main` without losing state.

## Scope
- Inspect current worktrees, local branches, and merge-base relationships before changing refs.
- Run the mandatory best-effort multi-CLI consensus for the integration approach.
- Preserve the current dirty branch state before checking out or updating `main`.
- Merge only branches/worktrees that are not already contained in `main`.

## Verification Criteria
- `git status --short --branch`
- `git worktree list --porcelain`
- `git branch -vv`
- `git fetch --all --prune`
- `git rev-list --left-right --count main...<branch>`
- `git status --short --branch` on `main` after integration

## Checklist
### Now
- [x] Review current workflow rules and lessons before integration work.
- [x] Inspect worktree/branch topology and identify unique work.
- [x] Run mandatory multi-CLI consensus for the merge strategy.
- [ ] Preserve the dirty worktree state on a branch or commit before switching targets.
- [ ] Update `main` and merge any branches with unique commits.

### Review
- [ ] Verify `main` contains the intended work and report any branches that were already fully merged.

---

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

# Task Plan - Investigate `build.zig:465` `addStaticLibrary` Breakage (2026-03-06)

## Objective
Identify the minimal Zig 0.16 migration needed for the current `build.zig:465` blocker where `b.addStaticLibrary` fails under the repo-pinned toolchain, and report exact file/line context without modifying repository source.

## Scope
- Reproduce the failure under `0.16.0-dev.2694+74f361a5c`.
- Inspect the local Zig stdlib build API for the canonical replacement.
- Validate the smallest source-level change in a throwaway build-file copy only.
- Do not change tracked source files as part of this investigation.

## Verification Criteria
- `which zig`
- `zig version`
- `cat .zigversion`
- `zig build toolchain-doctor`
- `zig build v3-lib`
- Local stdlib inspection for `std.Build.addLibrary`
- Throwaway `--build-file` validation of the replacement pattern

## Checklist
- [x] Review `tasks/lessons.md` before investigation.
- [x] Confirm active Zig matches `.zigversion`.
- [x] Run mandatory multi-CLI consensus for the migration question.
- [x] Reproduce the `build.zig:465` failure and capture the exact diagnostic.
- [x] Inspect local `std.Build` API around the missing method.
- [x] Validate the minimal replacement pattern in a throwaway build-file copy and with a temporary in-place rerun.

## Review
- [x] Evidence recorded for the exact blocker and minimal fix.
- [x] Residual risk noted if further Zig 0.16 migration errors remain after the first blocker.

### Evidence
- `zig build v3-lib` on the original source fails at `build.zig:465:21` with `error: no field or member function named 'addStaticLibrary' in 'Build'`.
- Local stdlib inspection shows `std.Build.addLibrary` at `/Users/donaldfilimon/.zvm/master/lib/std/Build.zig:839` and `LibraryOptions.linkage` at `/Users/donaldfilimon/.zvm/master/lib/std/Build.zig:820-823`, with no `addStaticLibrary` symbol present.
- The same `build.zig` already uses the Zig 0.16 pattern at `build.zig:448-452`: `b.addLibrary(.{ ... .linkage = .static, ... })`.
- A temporary in-place substitution of `build.zig:465-472` from `b.addStaticLibrary(.{ ... })` to `b.addLibrary(.{ ... .linkage = .static, ... })` moved `zig build v3-lib` past the method-missing failure and into unrelated Darwin linker errors (`undefined symbol: _abort`, `_clock_gettime`, etc.).

### Residual Risk
- This investigation isolates the first Zig 0.16 source migration needed at `build.zig:465`; additional build blockers remain after that point, but they are no longer `addStaticLibrary` API errors.

---

# Task Plan - Merge Remaining Branches Into `main` And Prune Useless Refs (2026-03-06)

## Objective
Ensure `main` contains all surviving branch work, then remove local and remote branches that are already fully merged and no longer needed.

## Scope
- Inspect local branches, remote branches, and linked worktrees before deleting any refs.
- Run the mandatory best-effort multi-CLI consensus for the cleanup strategy.
- Merge only if any branch still contains commits missing from `main`.
- Delete only branches that are confirmed ancestors of `main`.
- Remove linked worktrees that keep merged branches alive before deleting those branches.

## Verification Criteria
- `git status --short --branch`
- `git branch -vv`
- `git branch --merged main`
- `git branch -r`
- `git worktree list --porcelain`
- `git rev-list --left-right --count main...<branch>`

## Checklist
### Now
- [x] Review `tasks/lessons.md` before cleanup work.
- [x] Inspect local/remote branch ancestry and identify branches already merged into `main`.
- [x] Run mandatory multi-CLI consensus for the branch cleanup approach.
- [x] Remove any linked worktree that points at a fully merged disposable branch.
- [x] Delete fully merged local branches other than `main`.
- [x] Delete fully merged disposable remote branches other than `origin/main`.

### Review
- [x] Verify only `main` remains locally unless a surviving branch still has unique work.
- [x] Verify only `origin/main` remains remotely unless a surviving remote branch still has unique work.
- [x] Record evidence and residual risk for any branch intentionally left in place.

### Evidence
- `git branch --merged main` showed all non-`main` local branches (`integrate-all-work`, `integrate-stash-work`, `plan-next-zigmaster-steps`, `please-review-my-uncommitted-changes`) were already ancestors of `main`, so no additional merge into `main` was needed.
- Mandatory consensus artifact was captured under `tasks/branch-cleanup-consensus-out/`; surviving responders agreed on worktree-first cleanup followed by local and remote branch pruning, with Gemini failing and Ollama producing a parse-failure fallback.
- `git status --short --branch` in `/Users/donaldfilimon/.codex/worktrees/9a24/abi` showed the linked `please-review-my-uncommitted-changes` worktree was clean before removal.
- `git worktree remove /Users/donaldfilimon/.codex/worktrees/9a24/abi` succeeded, after which `git branch -d integrate-all-work integrate-stash-work plan-next-zigmaster-steps please-review-my-uncommitted-changes` deleted all merged local branches.
- `git push origin --delete please-review-my-uncommitted-changes` reported the remote ref no longer existed; `git fetch --prune origin` then removed the stale tracking ref, and `git ls-remote --heads origin` confirmed only `refs/heads/main` remains remotely.
- Final verification:
  - `git branch -vv` shows only local `main`.
  - `git branch -r` shows only `origin/main` and `origin/HEAD -> origin/main`.
  - `git worktree list --porcelain` shows only `/Users/donaldfilimon/abi` on `main`.

### Residual Risk
- No branch refs remain outside `main`, but a global stash still exists as `stash@{0}: On dev: pre-merge major-rewrite: local dev changes`. It was intentionally preserved because it is not a branch and deleting it was outside this request.
