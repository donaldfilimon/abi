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
