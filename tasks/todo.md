# Codebase Improvement Plan

## 38. Stabilize ABI on the Zig 0.17 Dev Line
- [x] Repin the repo and zigly metadata to `0.17.0-dev.135+9df02121d`.
- [x] Fix `build.sh` and zigly resolution so the wrapper resolves the exact pinned toolchain through `tools/zigly --status`.
- [x] Remove live `0.17` drift from CLI/MCP/TUI surfaces and active docs.
- [x] Validate the upgrade with focused toolchain, wrapper, CLI, and hygiene checks.

### Notes
- Opened on April 27, 2026 in `/Users/donaldfilimon/abi` on a dirty worktree; leave the pre-existing `src/protocols/mcp/handlers/ai.zig` edit untouched.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- `tools/scripts/run_build.sh` is absent in this checkout, so the ABI skill's macOS wrapper validation falls back to the repo-root `./build.sh` plus direct focused Zig checks.
- Completed on April 27, 2026 by repinning the project/tool manifests, feeding the compiler version through `build_options`, fixing `build.sh` default/link/bootstrap behavior and `tools/zigly --status` resolution, switching zigly's ZLS lookup to exact-version-first with ZVM/zig-only fallback, and updating the live CLI/MCP/TUI/docs surfaces plus focused regression tests.
- Validation passed with:
  - `~/.zvm/bin/zig fmt --check build.zig build/flags.zig build/cross.zig build/validation.zig tools/zigly_cli/src/cli.zig src/foundation/utils/zig_toolchain.zig src/cli.zig src/main.zig src/protocols/mcp/handlers/status.zig src/features/tui/app/dashboard/view_overview.zig src/mcp_main.zig test/integration/cli_test.zig test/integration/tui_test.zig`
  - `~/.zvm/bin/zig test tools/zigly_cli/src/cli.zig -lc`
  - `~/.zvm/bin/zig test tools/zigly_cli/src/core.zig`
  - `~/.zvm/bin/zig build typecheck --summary all`
  - `./tools/zigly --status`
  - `./tools/zigly --check`
  - `./build.sh --status`
  - `./build.sh typecheck --summary all`
  - `./build.sh cli`
  - `./build.sh cli-tests --summary all`
  - `./build.sh tui-tests --summary all`
  - `./zig-out/bin/abi`
  - `./zig-out/bin/abi version`
  - `git diff --check`
- Residual risk: exact prebuilt ZLS artifacts for dev snapshots remain external; the new contract intentionally keeps Zig resolution working and emits a warning when only Zig is available.

## 39. AiOps Adapter Cast-Helper Hardening
- [ ] Add focused unit coverage for `src/features/gpu/ai_ops/adapters.zig` that validates vtable dispatch through the centralized opaque-pointer cast helper.
- [ ] Keep the change isolated to adapter test coverage with no public API surface changes.
- [ ] Validate with targeted formatting + focused test command(s), then record residual risk.

### Notes
- Opened on April 28, 2026 in `/Users/donaldfilimon/abi` during the ongoing refactor wave with a heavily dirty worktree; this slice must stay confined to adapter coverage and workflow notes only.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.

## 37. Align MCP Status Factory Test With Actual Registration Set
- [x] Keep the status factory test aligned with `createStatusServer()`'s real tool set.
- [x] Validate the touched MCP factory surface with focused formatting and test checks.
- [x] Commit the cleanup on top of `main`.
- [x] Record the validation outcome and residual risk.

### Notes
- Opened on April 27, 2026 in `/Users/donaldfilimon/abi` after the prior MCP consolidation commit left one tracked factory test edit in the worktree.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Validation outcome:
  - `~/.zvm/bin/zig fmt --check src/protocols/mcp/factories.zig`
  - `./build.sh test --summary all -- --test-filter "createStatusServer registers 5 tools"` failed before build execution because ZVM could not activate Zig `0.17.0-dev.27+0dd99c37c`
  - `./build.sh test --summary all -- --test-filter "createCombinedServer registers database and ZLS tools"` failed for the same ZVM activation reason
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "createStatusServer registers 5 tools"` ran the targeted single filtered test successfully, but the overall command still exited non-zero because the pre-existing `features.core.database.storage.wal.test.wal resume appending to existing file` failure remains in the broader suite
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "createCombinedServer registers database and ZLS tools"` likewise ran the targeted single filtered test successfully, but the command exited non-zero for the same unrelated WAL failure
  - `git diff --check`
- Residual risk: this cleanup only updates the factory test expectation; broader suite health is still limited by the existing WAL failure and the current ZVM activation issue in `./build.sh`.

## 36. Merge MCP Cleanup Into `main`
- [x] Stage the MCP handler and registry cleanup.
- [x] Validate the touched MCP files with focused formatting and test checks.
- [x] Commit the cleanup on top of `main`.
- [x] Record the validation outcome and any residual risk.

### Notes
- Opened on April 27, 2026 in `/Users/donaldfilimon/abi` while consolidating the current `main` checkout.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Validation passed with:
  - `~/.zvm/bin/zig fmt --check src/protocols/mcp/handlers/ai.zig src/protocols/mcp/registry.zig`
  - `./build.sh test --summary all -- --test-filter "ToolDef format"`
  - `./build.sh test --summary all -- --test-filter "ResourceDef format"`
- Residual risk: validation was intentionally narrow to the two touched MCP surfaces and did not broaden to unrelated protocol lanes.

## 35. Zig 0.17.0-dev.3091 Toolchain + Inference Safety Sweep
- [x] Bump the repo pin and minimum supported Zig version to `0.17.0-dev.3091+557caecaa` everywhere current-toolchain references are surfaced.
- [x] Fix the in-progress CLI/database/inference cleanup so the new tests compile and ownership/locking remain sound.
- [x] Restore or intentionally prune the accidental `.claude` deletions after review.
- [x] Validate with focused fmt/tests before broader verification.

### Notes
- Completed on April 5, 2026 in `/Users/donaldfilimon/abi`.
- Validation passed with:
  - `~/.zvm/bin/zig fmt --check src/main.zig src/features/core/database/cli.zig src/inference/engine.zig test/integration/cognitive_pipeline_test.zig test/integration/inference_stress_test.zig src/foundation/utils/zig_toolchain.zig tools/zigly_cli/src/cli.zig src/features/ai/abbey/discord.zig src/features/ai/agents/types.zig test/mod.zig build.zig.zon`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "cognitive pipeline"`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "inference stress"`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "preferred source"`
  - `~/.zvm/bin/zig test tools/zigly_cli/src/cli.zig -lc`
  - `git diff --check`
- Residual note: the standalone `zig test src/foundation/utils/zig_toolchain.zig` path still needs a proper module-path wrapper, so the build-backed test lane was used for that file instead.

## 0G. Review Follow-Up: DiskANN Surface + Abbey Gateway Cleanup
- [x] Restore the `abi.database.retrieval.diskann` export in the real retrieval facade.
- [x] Remove the overlapping gateway bridge destroy path in Abbey startup.
- [x] Validate with the targeted database and Abbey gateway tests, then rerun `check-parity`.

### Notes
- Opened on April 4, 2026 in `/Users/donaldfilimon/abi` as a follow-up to the live regression repair plan.
- Completed on April 4, 2026 by restoring the real `diskann` export, keeping the real/stub retrieval surfaces aligned, fixing Abbey gateway bridge ownership so startup cleanup runs exactly once, and migrating the remaining DiskANN/block temp-file paths off removed `std.posix.unlink` calls.
- Validation passed with:
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "diskann"`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "gateway bridge init and deinit"`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "abbey discord bot gateway lifecycle"`
  - `~/.zvm/bin/zig build check-parity`
- Residual limitation: the repo-local `./build.sh` / zigly wrapper is still known to fail in this checkout with `Undefined error: 0`, so the verification above used the pinned `~/.zvm/bin/zig` fallback.

## 0F. Review Follow-Up: Dashboard + Profile Router Regression Repair
- [x] Wave 1: restore the `abi.tui.dashboard` helper surface after the dashboard split.
- [x] Wave 1: fix `MultiProfileRouter.executeParallel()` error mapping, join ordering, and response ownership cleanup without widening the public contract.
- [x] Wave 1: add router coverage for the extracted parallel response-selection helper.
- [x] Wave 2: restore deferred `refAllDecls` handling in `src/runtime/engine/mod.zig`.
- [x] Wave 2: restore deferred `refAllDecls` handling in `src/features/core/database/memory/mod.zig`.
- [x] Validate with targeted fmt/typecheck/TUI/profile gates first, then broaden if the wrapper environment permits.

### Notes
- Opened on April 4, 2026 in `/Users/donaldfilimon/abi` against an already-dirty worktree; this follow-up must stay confined to the review finding files plus `tasks/todo.md`.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Completed on April 4, 2026 by restoring the dashboard facade helpers (`computeLayout`, `handleKey`, `hasVisibleCell`, `containsText`), fixing the decomposed dashboard view import paths that blocked TUI compilation, tightening `MultiProfileRouter.executeParallel()` to keep `ProfileError`-only failures with single-join ownership cleanup, adding focused helper tests, and reinstating the deferred `refAllDecls` comments in the two known-broken modules.
- Validation outcome:
  - `~/.zvm/bin/zig fmt --check src/features/tui/app/dashboard.zig src/features/tui/app/dashboard/view_overview.zig src/features/tui/app/dashboard/view_features.zig src/features/tui/app/dashboard/view_runtime.zig src/features/ai/profile/router.zig src/runtime/engine/mod.zig src/features/core/database/memory/mod.zig`
  - `./tools/zigly --status` failed with `/Users/donaldfilimon/abi/tools/zigly_cli/zig-out/bin/zigly: Undefined error: 0`
  - `./build.sh typecheck --summary all` failed for the same zigly-wrapper reason, so the existing pinned Zig was used as fallback evidence only
  - `~/.zvm/bin/zig build typecheck --summary all`
  - `~/.zvm/bin/zig build tui-tests --summary all`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "profile"`
  - `~/.zvm/bin/zig build test --summary all`
  - `git diff --check -- tasks/todo.md src/features/tui/app/dashboard.zig src/features/tui/app/dashboard/view_overview.zig src/features/tui/app/dashboard/view_features.zig src/features/tui/app/dashboard/view_runtime.zig src/features/ai/profile/router.zig src/runtime/engine/mod.zig src/features/core/database/memory/mod.zig`
- Residual limitation: the repo-local `./build.sh` path remains blocked by the current zigly binary state, so this follow-up is validated with direct `~/.zvm/bin/zig build ...` fallback rather than the preferred Darwin wrapper.

## 0E. ZVM-First Toolchain Alignment + Zig Pin Bump
- [x] Bump `.zigversion` to `0.17.0-dev.3070+b22eb176b`.
- [x] Make `tools/zigly` resolve/install the pinned Zig through ZVM first when ZVM is present.
- [x] Align internal Zig path helpers and auto-update flow with the same ZVM-first resolution order.
- [x] Refresh toolchain docs/comments to describe the ZVM-first contract and the new pin.
- [x] Validate the new pin and resolver behavior with ZVM checks plus repo build gates.

### Notes
- Opened on April 3, 2026 in `/Users/donaldfilimon/abi` with a clean tracked worktree on `main`; this wave is toolchain-focused and should avoid unrelated repo cleanup.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Current drift: `.zigversion` still pins `0.17.0-dev.2984+cb7d2b056`, `build.sh` resolves Zig through `tools/zigly --status`, `tools/zigly_cli/src/cli.zig` only returns the zigly cache path, and `src/foundation/utils/zig_toolchain.zig` still prefers the legacy `~/.cache/abi-zig` path plus `~/.zvm/master/zig`.
- Environment note before implementation: `zvm v0.8.14` rejects the explicit snapshot `0.17.0-dev.3070+b22eb176b` as unsupported, but `zvm install master` / `zvm use master` does expose `~/.zvm/bin/zig` at that exact version. `~/.zvm/versions-zls.json` was also permission-blocked until it was removed from the user-writable directory.
- Completed on April 3, 2026 with `.zigversion` pinned to `0.17.0-dev.3070+b22eb176b`, `tools/zigly --status` returning `/Users/donaldfilimon/.zvm/bin/zig`, and the native `zigly` bootstrap updated to rebuild when its sources change.
- `tools/zigly_cli/src/cli.zig`, `src/foundation/utils/zig_toolchain.zig`, `build.sh`, `tools/crossbuild.sh`, and `tools/auto_update.sh` now agree on the ZVM-first lookup order: use `~/.zvm/bin/zig` when its reported version matches `.zigversion`, otherwise fall back to the pinned zigly cache.
- Validation passed with:
  - `zig fmt --check src/foundation/utils/zig_toolchain.zig tools/zigly_cli/build.zig tools/zigly_cli/src/cli.zig tools/zigly_cli/src/core.zig`
  - `~/.zvm/bin/zig test tools/zigly_cli/src/cli.zig -lc`
  - `~/.zvm/bin/zig test tools/zigly_cli/src/core.zig`
  - `tools/auto_update.sh --check` (reported `Already up to date.` on `0.17.0-dev.3070+b22eb176b`)
  - `./tools/zigly --status`
  - `./tools/zigly --install`
  - `zvm use --sync`
  - `zig version`
  - `~/.zvm/bin/zig version`
  - `./build.sh typecheck --summary all`
  - `./build.sh check --summary all`
- Residual environment caveat: `zvm v0.8.14` still needs the `master` alias fallback to reach this exact snapshot, and the active ZVM `zls` remains `0.17.0-dev.296+ef64fa01` even while `zig` is aligned to `0.17.0-dev.3070+b22eb176b`.

## 0D. Merge Attached Workspaces Into `main`
- [x] Add a short merge/cleanup checklist here before mutating git history.
- [x] Exclude accidental `.claude/worktrees/*` index entries from the consolidation commit.
- [x] Commit the dirty root `main` workspace as one intentional consolidation commit.
- [x] Cherry-pick `f6c3abe080b9c77e1ce90d496e44efcf5d3489fa` from `worktree-agent-aea73b27` onto `main`.
- [x] Validate the integrated `main` workspace with parity, CLI, TUI, and full-check gates.
- [x] Remove attached worktrees and delete their local branches once validation passes.

### Notes
- Opened on April 3, 2026 in `/Users/donaldfilimon/abi`, which is already checked out on `main`; this wave is about integrating attached worktree history and cleaning up the local workspace topology without rebasing or resetting against `origin/main`.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Current git topology shows two attached worktrees under `.claude/worktrees/`: `worktree-agent-ad29d5e3` has no unique commits versus `main`, while `worktree-agent-aea73b27` contributes one unique inference fix commit on `src/inference/engine/backends.zig`.
- The root `main` workspace is intentionally dirty and includes prior staged/unstaged repo changes plus accidental staged `.claude/worktrees/*` entries that must stay out of the consolidation commit.
- Completed on April 3, 2026 with root consolidation commit `2f45c45` (`chore: consolidate local main workspace`) followed by cherry-picked inference fix `de77cc1` on `main`.
- The cherry-pick hit a content conflict in `src/inference/engine/backends.zig`; it was resolved by keeping the new `error.UnsupportedProvider` behavior for bare model IDs and then updating the stale local test to assert that contract instead of the old echo fallback.
- Attached worktrees `/Users/donaldfilimon/abi/.claude/worktrees/agent-ad29d5e3` and `/Users/donaldfilimon/abi/.claude/worktrees/agent-aea73b27` were removed, and the now-obsolete local branches `worktree-agent-ad29d5e3` and `worktree-agent-aea73b27` were deleted after validation.
- Validation passed with:
  - `./build.sh test --summary all -- --test-filter "dispatchToConnector: no slash returns UnsupportedProvider"`
  - `./build.sh check-parity --summary all`
  - `./build.sh cli-tests --summary all`
  - `./build.sh tui-tests --summary all`
  - `./build.sh check --summary all`

## 0C. Dashboard Surface Alignment Wave
- [x] Make the shared CLI renderers authoritative for status/help so `src/main.zig` stops duplicating dashboard/help copy.
- [x] Update dashboard wording across CLI/docs to describe the developer diagnostics shell and its overview/features/runtime views.
- [x] Add a short fallback note everywhere user-facing that non-interactive dashboard runs point to `abi doctor`.
- [x] Tighten CLI/help tests around the shared dashboard contract and descriptor alignment.
- [x] Validate the CLI/docs alignment wave with targeted fmt, typecheck, cli-tests, and real CLI invocations.

### Notes
- Opened on April 3, 2026 in `/Users/donaldfilimon/abi` on top of the already-validated TUI shell rewrite from §0B; this wave should not reopen dashboard layout or interaction behavior unless the shared CLI contract exposes a regression.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Current drift: `src/cli.zig` already owns `writeStatus()` / `writeHelp()`, but `src/main.zig` still hardcodes separate status/help text while README/AGENTS/CLAUDE continue to describe `abi dashboard` generically as an interactive TUI.
- Completed on April 3, 2026 by routing `printStatus()` / `printHelp()` through the shared CLI writers, centralizing the dashboard wording in `src/cli.zig`, and updating README/AGENTS/CLAUDE to describe `abi dashboard` as the developer diagnostics shell with overview/features/runtime views plus the `abi doctor` fallback note.
- Validation passed with:
  - `zig fmt --check src/cli.zig src/main.zig test/integration/cli_test.zig`
  - `git diff --check -- src/cli.zig src/main.zig README.md AGENTS.md CLAUDE.md test/integration/cli_test.zig tasks/todo.md`
  - `./build.sh typecheck --summary all`
  - `./build.sh cli-tests --summary all`
  - `./build.sh cli --summary all`
  - `./zig-out/bin/abi dashboard` → `TUI is disabled. Rebuild with -Dfeat-tui=true`
  - `./build.sh -Dfeat-tui=true cli --summary all`
  - `./zig-out/bin/abi`
  - `./zig-out/bin/abi help`
  - `./zig-out/bin/abi dashboard` → `TUI dashboard requires an interactive terminal.` / `Use 'abi doctor' for non-interactive diagnostics.`

## 0B. TUI Diagnostic Shell Rethink
- [x] Replace the static two-panel dashboard with a mode-aware developer diagnostics shell.
- [x] Drive the features view from `src/core/feature_catalog.zig` instead of a hand-maintained flag list.
- [x] Add resize-aware layout breakpoints plus nav/detail/help interaction state.
- [x] Expand TUI tests from smoke coverage to layout, navigation, and catalog-alignment assertions.
- [x] Validate with targeted TUI gates first, then the broader dashboard/TUI lanes, and record outcomes here.

### Notes
- Opened on April 3, 2026 in `/Users/donaldfilimon/abi` with a dirty worktree containing large unrelated staged changes, especially the out-of-scope `src/features/core/**` migration; this TUI slice must avoid touching those files.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The current dashboard implementation in `src/features/tui/app/dashboard.zig` still renders a fixed two-column split using a handwritten 33-flag inventory and only smoke-level interaction coverage.
- Completed on April 3, 2026 with `src/features/tui/app/dashboard.zig` rewritten around `View`/`FocusRegion` state, catalog-driven feature rendering, runtime/service diagnostics, compact/minimal breakpoints, and a help overlay while keeping the existing `abi dashboard` entrypoint and non-interactive fallback text.
- The public integration lane in `test/integration/tui_test.zig` now asserts layout modes, navigation/help transitions, catalog alignment against `abi.meta.features.feature_count`, compact/minimal rendering, and resize-safe redraws instead of only checking that the dashboard does not crash.
- Validation passed with:
  - `zig fmt --check src/features/tui/app/dashboard.zig test/integration/tui_test.zig`
  - `git diff --check -- src/features/tui/app/dashboard.zig test/integration/tui_test.zig tasks/todo.md`
  - `./build.sh typecheck --summary all`
  - `./build.sh -Dfeat-tui=true test --summary all -- --test-filter "tui:"`
  - `./build.sh dashboard-smoke --summary all`
  - `./build.sh tui-tests --summary all`
  - `./build.sh -Dfeat-tui=true cli --summary all`
  - `./zig-out/bin/abi dashboard` (non-interactive fallback verified: `TUI dashboard requires an interactive terminal.` / `Use 'abi doctor' for non-interactive diagnostics.`)
- Environment note:
  - `./tools/zigly --status fmt --check ...` is currently unusable in this checkout because it tries to download a missing Zig fmt tarball and exits with `curl: (22) ... 404`, so local `zig fmt --check` was used instead.

## 0. AI Feature Graph Expansion + ZVM Pin Sync
- [x] Expand `src/core/feature_catalog.zig` to cover the full public `abi.ai` graph, including distinct `profile` and `profiles` entries.
- [x] Move public AI module parity coverage into `src/feature_parity_tests.zig` Tier 1 and leave only non-`abi.ai` internal modules in the manual appendix.
- [x] Remove the handwritten `src/core/registry/stub.zig` feature enum drift by aliasing the canonical catalog-backed `Feature` type and updating parent tests.
- [x] Update CLI/docs/tests to reflect the expanded catalog count and prefer derived expectations over hardcoded literals.
- [x] Sync ZVM to `.zigversion`, run targeted validation, and record outcomes plus residual risk here.

### Notes
- Opened on April 3, 2026 in `/Users/donaldfilimon/abi` with a dirty staged worktree that already contains a large out-of-scope `src/features/core/**` migration; this slice must not modify or depend on that staged tree.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; direct repo inspection is the source of truth for this slice.
- Validation passed with:
  - `zig fmt --check src/core/config/mod.zig src/core/feature_catalog.zig src/core/registry/stub.zig src/core/registry/types.zig src/core/registry/mod.zig src/feature_parity_tests.zig src/main.zig test/integration/cli_test.zig`
  - `./build.sh check-parity --summary all`
  - `./build.sh cli-tests --summary all`
  - `./build.sh typecheck --summary all`
  - `./build.sh test --summary all`
- ZVM outcome:
  - `.zigversion` pins `0.17.0-dev.2984+cb7d2b056`
  - `zvm install 0.17.0-dev.2984+cb7d2b056` failed in `zvm v0.8.14` with `unsupported Zig version`
  - `zvm use --sync` was a no-op until `~/.zvm/bin` was repointed to a local compatibility install assembled from `~/.zigly/versions/0.17.0-dev.2984+cb7d2b056`
  - Final verification: `zig version` now reports `0.17.0-dev.2984+cb7d2b056`, and `zvm list` shows both `0.17.0-dev.2984+cb7d2b056` and `master`
- Residual risk:
  - `~/.zvm/versions-zls.json` is root-owned in this environment, which breaks `zvm list --all` metadata refreshes and likely contributed to the native `zvm install` limitation for older dev snapshots.

## 1. Architectural Inconsistencies
- [x] Rename `feat_profiling` to `feat_observability` in `src/core/feature_catalog.zig` and `src/root.zig`.
- [x] Move nested features like `pages` to their own top-level directories or clarify their sub-feature status.
- [x] Decouple components within the `gpu` module. *(Note: Addressed the immediate import breakages. Further decoupling has been deferred to the Massive Update Plan to ensure stability).*

## 2. Code Quality
- [x] Fix the compilation error in `src/core/framework/context_init.zig` and `shutdown.zig` introduced by dynamic initialization refactoring.
- [x] Investigate and clean up manual boilerplate feature initialization to use dynamic registration via `feature_catalog`.

## 3. Test Coverage
- [x] Enhance integration tests to include deep behavioral scenarios rather than just API surface checks in `gpu_test.zig`.
- [x] Add functional/behavioral tests for the `ai` module, testing core orchestration logic.

## 4. Exploration & Massive Update Planning
- [x] Explore existing agent definitions, skills (`.claude/skills`, `zig-abi-plugin/skills`), memory files, and documentation (`.md` files) across the repository.
- [x] Synthesize findings to formulate a plan for a massive codebase and tooling update based on the discovered capabilities and guidelines.

## 5. Current ABI Patch
- [x] Fix CLI routing for `serve` and `acp serve`, including serve option parsing.
- [x] Repair framework plugin ownership, allocator usage, and shutdown cleanup.
- [x] Harden plugin path loading behind `allow_untrusted`.
- [x] Make async inference own request strings, free async results, and wait for in-flight jobs.
- [x] Fix `StateBlockChain` chain ID ownership and block insertion cleanup. *(Backported to `BlockChain` and `DistributedBlockChain`; no `StateBlockChain` symbol exists in this checkout.)*

## 6. Serve / Plugin / Transport Regression Fixes
- [x] Reformat serve host+port composition to bracket IPv6 hosts.
- [x] Restore `PluginConfig.withPaths()` compatibility with validation.
- [x] Add teardown for `PendingRequest` and host-staged thread-pool sync primitives.
- [x] Add smoke tests for the above paths.

### Notes
- `zig build test --summary all` now passes; the repository build gate is green after the stabilization fixes.
- The current tree still does not contain a `StateBlockChain` symbol, so the cleanup was backported to the existing block-chain implementations instead.
- Validation for this regression batch passed with `./build.sh test --summary all` on Darwin.

## 7. WDBX Search Performance
- [x] Add selectable `exact` / `hnsw` / `diskann` search backends to the in-memory database config.
- [x] Wire ANN search through the HNSW and DiskANN backends with exact fallback when the index is unavailable.
- [x] Add quantization-aware vector compression/decompression helpers for index builds.
- [x] Add DiskANN save/load/search plumbing and regression coverage for small-corpus smoke tests.
- [x] Keep bulk-load and optimize paths in sync with the active search index.

## 8. CLI Arg Refactor
- [x] Extract chat message joining into a reusable CLI helper.
- [x] Update the CLI wording, README example, and integration coverage for `chat <message...>`.

### Notes
- `./build.sh cli` and `./build.sh test --summary all` passed after the refactor.
- The `abi lsp` branch was intentionally left unchanged in this pass.

## 9. Feature Matrix Gate Canonicalization
- [x] Add a canonical `zig build feature-tests` gate that mirrors the current TUI, mobile, and minimal feature matrix.
- [x] Replace the root-level scratch helper with a tracked `tools/feature_tests.sh` wrapper.
- [x] Document the canonical gate in `tools/README.md`.
- [x] Validate with `./build.sh feature-tests` and `./build.sh test --summary all`.

## 10. In-Flight Refactor Wave: Abbey / TUI / MCP Extraction
- [x] Finish wiring the extracted Abbey helpers through the `src/features/ai/abbey` facade and keep parity with the stub surface.
- [x] Finish the TUI dashboard split so the parent wrapper stays thin and the new view/state modules stay covered by tests.
- [x] Finish the MCP transport framing extraction and keep the public stdio/SSE transport contract unchanged.
- [x] Validate the refactor wave with parity, typecheck, and the impacted agents, TUI, Abbey/profile, and MCP lanes.

### Notes
- Opened on April 4, 2026 in `/Users/donaldfilimon/abi` as the follow-up cleanup after the `src/` canonicalization pass.
- Completed on April 4, 2026 by wiring the Abbey convenience/re-export split, keeping the TUI dashboard facade thin around the extracted `dashboard/` modules, and moving MCP framing into `src/protocols/mcp/transport/framing.zig`.
- Validation passed with:
  - `~/.zvm/bin/zig build check-parity`
  - `~/.zvm/bin/zig build typecheck --summary all`
  - `~/.zvm/bin/zig build agents-tests --summary all`
  - `~/.zvm/bin/zig build tui-tests --summary all`
  - `~/.zvm/bin/zig build mcp-tests --summary all`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "abbey"`
  - `~/.zvm/bin/zig build test --summary all -- --test-filter "profile"`
- Residual limitation: the repo-local `./build.sh` / zigly wrapper still fails in this checkout with `Undefined error: 0`, so the pinned `~/.zvm/bin/zig` toolchain remains the authoritative validation path here.

### Notes
- The multi-CLI consensus helper is unavailable in this checkout, so this task proceeded with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`, matching the startup limitation recorded in `tasks/lessons.md`.
- Residual scope: this pass canonicalizes the three-case feature matrix only; broader roadmap aliases like `cli-tests` and `tui-tests` remain separate work.

## 10. Validation Contract Parity Wave
- [x] Add truthful validation aliases in `build.zig`: `cli-tests`, `tui-tests`, `dashboard-smoke`, `typecheck`, `validate-flags`, `full-check`, and `verify-all`.
- [x] Restore the canonical `feature-tests` target in `build.zig` so the tracked wrapper and validation docs point at a real build step again.
- [x] Rewrite `src/tasks/roadmap_catalog.zig` so every active validation gate references only implemented commands.
- [x] Update the public build-command lists in `README.md`, `CLAUDE.md`, and `AGENTS.md`.
- [x] Validate the lane with host bring-up, aggregate gates, and cross-target `typecheck` coverage.

### Notes
- The multi-CLI consensus helper is unavailable in this checkout, so this task proceeds with the ABI best-effort fallback.
- Existing unrelated edits in `src/features/ai/abbey/stub.zig`, `src/features/ai/llm/unified_orchestrator/stub.zig`, `src/features/ai/prompts/stub.zig`, and `src/inference/engine.zig` remain out of scope and must be preserved.
- Validation passed with `$(./tools/zigly --status) fmt --check build.zig`, `./build.sh --help`, `./build.sh validate-flags`, `./build.sh cli-tests`, `./build.sh tui-tests`, `./build.sh dashboard-smoke`, `./build.sh full-check`, `./build.sh verify-all`, `./build.sh -Dtarget=x86_64-linux-gnu typecheck`, `./build.sh -Dtarget=x86_64-windows-gnu typecheck`, `./build.sh -Dtarget=aarch64-macos typecheck`, and `./tools/feature_tests.sh`.
- The first `aarch64-macos` `typecheck` attempt exposed that compile-only validation could not share the linked static-library path; the dedicated compile-only target is tracked separately so cross-target checks stay link-free.

## 11. GPU Redesign v3
- [x] Wave 3A: finalize strict backend request handling across creation paths.
- [x] Wave 3B: harden pool deinit/ownership rules for mixed backend graphs.
- [x] Wave 3C: close remaining cross-target policy parity gaps and lock tests.

### Notes
- Wave 3A enforced strict backend request handling via `GpuConfig.strict_backend`, preventing silent fallbacks and returning `error.NoDeviceAvailable` on explicit backend failures.
- Wave 3B hardened pool ownership by binding device-mapped buffers to their allocating backend VTable, ensuring safe `deinit()` even when the default/active context changes in mixed topologies.
- Wave 3C locked down the cross-target policy assertions across macOS, Linux, Windows, Web, and Android via `catalog.zig` tests.

## 12. CLI Framework + Local-Agent Fallback
- [x] Wave 1A: standardize runtime-to-descriptor assertions for single-token commands.
- [x] Wave 1B: harden untrusted provider plugin limits to fallback modes only.
- [x] Wave 1C: unify standard pipeline filtering in stdout paths.

### Notes
- Wave 1A exported a `cli.single_token_commands` descriptor catalog, asserting it directly tracks implemented logic in `dispatch()` and fixing assertion drift for hidden commands like `lsp`.
- Wave 1B removed `plugin_http` and `plugin_native` from automatic `model_name_chain` and `file_model_chain` lists, ensuring untrusted provider plugins execute only when explicitly requested or listed in an explicit fallback chain.
- Wave 1C unified stdout diagnostic formatting in `src/main.zig` via a `printHeader` utility, which intelligently strips ANSI decoration and layout lines when `os.isatty()` is false, enabling clean pipelining to external automation tools. Additionally resolved a missing `CoreGraphics`/`CoreFoundation` framework linkage block affecting MacOS test environments.

## 13. TUI Modular Extraction v2
- [x] Wave 2A: complete launcher/dashboard extraction onto shared render/layout primitives.
- [x] Wave 2B: close input routing and focus-state correctness gaps.
- [x] Wave 2C: expand unit and integration-style TUI tests for layout and hit-testing.

### Notes
- The multi-CLI consensus helper is unavailable in this checkout, so Wave 2C proceeded with the ABI best-effort fallback.
- `tools/scripts/run_build.sh` is also absent in this checkout, so Darwin verification used the direct `./build.sh` gates after `zig fmt --check`.
- `tui-tests` now runs a dedicated `feat_tui=true` unit/integration lane, exercising the real TUI dashboard, layout, render, and consumer paths instead of the stubbed default build.
- Validation passed with `$(./tools/zigly --status) fmt --check build.zig src/features/tui/app/dashboard.zig src/features/tui/layout.zig src/features/tui/render.zig test/integration/tui_test.zig`, `./build.sh tui-tests`, `./build.sh feature-tests`, and `git diff --check`.
- `./build.sh full-check` and `./build.sh verify-all` remain blocked by pre-existing repo-wide formatting drift in `src/features/gpu/policy/catalog.zig`, `src/features/tui/types.zig`, and `src/features/ai/llm/types.zig`.

## 14. Cross-Target GPU Policy Contract Hardening
- [x] Make `typecheck` a truthful compile-only gate in `build.zig`.
- [x] Add a compile-only GPU policy contract root that asserts macOS, Linux, and Windows target policy invariants.
- [x] Fix the Darwin `build.sh` relink fallback to select only host-native `libcompiler_rt.a` archives.
- [x] Remove `-Dgpu-backend=auto` from roadmap and workflow validation contract examples.
- [x] Update public `typecheck` wording in `README.md`, `CLAUDE.md`, and `AGENTS.md`.
- [x] Validate the lane with host `typecheck`, three cross-target `typecheck` runs, and `cross-check`.

### Notes
- Reopened in this checkout because the repo still reflected the pre-RM-006 state: `typecheck` aliased `check`, the Darwin wrapper chose the newest `libcompiler_rt.a`, and the roadmap contract still advertised `-Dgpu-backend=auto`.
- The multi-CLI consensus helper is unavailable in this checkout, so RM-006 proceeded with the ABI best-effort fallback.
- `typecheck` now depends on compile-only object builds for `src/root.zig` and `src/features/gpu/policy/target_contract.zig`, so it no longer aliases `check` or triggers fmt/test/parity side effects.
- The Darwin wrapper now derives the build-runner architecture from the cached `build` binary and only selects `libcompiler_rt.a` archives whose Mach-O members match that host architecture.
- Validation passed with `$(./tools/zigly --status) fmt --check build.zig src/features/gpu/policy/target_contract.zig src/tasks/roadmap_catalog.zig`, `./build.sh typecheck`, `./build.sh -Dtarget=x86_64-linux-gnu typecheck`, `./build.sh -Dtarget=x86_64-windows-gnu typecheck`, `./build.sh -Dtarget=aarch64-macos typecheck`, `./build.sh cross-check`, and `git diff --check`.

## 15. Full-Check Remediation
- [x] Fix the `zig fmt --check` failure in `src/platform/smc.zig`.
- [x] Restore `src/protocols/ha/mod.zig` parity with `src/protocols/ha/stub.zig` for `init`, `deinit`, `isEnabled`, and `isInitialized`.
- [x] Validate with `$(./tools/zigly --status) fmt --check src/platform/smc.zig src/protocols/ha/mod.zig src/protocols/ha/stub.zig`, `./build.sh check-parity`, and `./build.sh full-check`.

### Notes
- Opened on branch `fix/full-check-formatting-drift` after `./build.sh full-check` failed on March 24, 2026.
- Current gate output shows two blockers: `src/platform/smc.zig` formatting drift and `src/feature_parity_tests.zig` reporting that `ha/mod.zig` is missing `init`, `deinit`, `isEnabled`, and `isInitialized` from `ha/stub.zig`.
- Validation passed with `$(./tools/zigly --status) fmt --check src/platform/smc.zig src/protocols/ha/mod.zig src/protocols/ha/stub.zig`, `./build.sh check-parity`, `./build.sh full-check`, and `./build.sh verify-all`.

## 16. Connectors Module Organization
- [x] Split `src/connectors/mod.zig` into focused auth, env, provider-registry, and loader helpers while preserving the current public API.
- [x] Reduce repetitive connector `tryLoad*` boilerplate via shared loader helpers.
- [x] Validate with targeted fmt, `./build.sh typecheck`, and connector-focused tests.

### Notes
- Opened on branch `refactor/connectors-module-layout` on March 24, 2026 to make the connectors surface easier to navigate without changing caller-facing imports.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- `src/connectors/mod.zig` is now a thin public index; the extracted logic lives in `src/connectors/auth.zig`, `src/connectors/env.zig`, `src/connectors/providers.zig`, and `src/connectors/loaders.zig`.
- The loader refactor keeps `abi.connectors.load*` / `tryLoad*` stable while centralizing the missing-config-to-null behavior in a typed helper.
- Validation passed with `$(./tools/zigly --status) fmt --check src/connectors/mod.zig src/connectors/auth.zig src/connectors/env.zig src/connectors/providers.zig src/connectors/loaders.zig`, `./build.sh typecheck`, `./build.sh test --summary all`, and `git diff --check`.
- `./build.sh test --summary all --test-filter connectors` is not supported by this build wrapper, so the standard test gate was used instead.

## 17. Codebase Improvement Roadmap vNext
- [x] Audit previously completed roadmap items against the current tree and rewrite stale status entries before starting another large refactor wave. *(Done in §18)*
- [x] Normalize the feature model: resolve `feat_profiling` vs observability naming drift and decide whether `pages` stays a standalone feature or folds into observability. *(Done in §18)*
- [x] Decompose `build.zig` into focused build helpers for flags, linking, target matrices, and validation steps. *(Done in §19, §21)*
- [x] Break down oversized mixed-responsibility modules in AI, GPU, inference, gateway, protocols, and security into smaller entrypoints and internal helpers. *(Done in §22-29)*
- [x] Expand targeted build/test lanes so domain refactors can be validated without always paying the cost of the full-suite path. *(Done in §21-29: 27 focused test lanes)*
- [x] Clean repo hygiene drift (`.DS_Store`, generated artifacts, doc/build contract mismatches) and automate checks where feasible. *(Done in §18)*

### Notes
- Requested on March 24, 2026 as a planning-only pass; no code changes should be made under this section until the audit item is closed.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this roadmap was prepared with the ABI best-effort fallback.
- Current structural hotspots by size include `src/features/ai` (~124,874 LOC), `src/features/gpu` (~80,590 LOC), `src/core` (~42,467 LOC), `src/connectors` (~11,521 LOC), and several 1,000+ line files such as `src/features/ai/agents/agent.zig`, `src/features/gateway/mod.zig`, `src/inference/engine.zig`, `src/protocols/mcp/server.zig`, `src/protocols/ha/pitr.zig`, `src/foundation/security/secrets.zig`, and the GPU codegen generators.
- The current tree still contains `feat_profiling` / `feat_pages` naming and layout references in `build.zig`, `src/root.zig`, `src/core/feature_catalog.zig`, `src/core/comptime_meta.zig`, `src/core/framework/context_init.zig`, `src/core/framework/feature_imports.zig`, and `CLAUDE.md`, even though earlier local roadmap sections marked the rename/layout work complete.
- The integration surface is broad (`test/integration/*.zig` spans connectors, protocols, feature domains, CLI, TUI, and GPU), but the build wrapper currently lacks targeted test filtering support, which increases the cost of safe refactors.
- Repo hygiene follow-ups should include ignoring and removing `.DS_Store` files found under the repo root, `test/`, `docs/`, and `zig-abi-plugin/`.

## 18. Wave 1 Multi-Agent Execution
- [x] Audit stale roadmap status vs the current tree and rewrite the active roadmap to reflect reality.
- [x] Rename the code-level observability build flag surface from `feat_profiling` to `feat_observability`, with a deliberate compatibility decision for CLI/build usage.
- [x] Decide and implement the `pages` feature boundary: either keep it as a documented observability sub-feature or fold it into observability and remove the separate gate.
- [x] Remove stray `.DS_Store` files from the repository workspace and confirm no tracked files depend on them.
- [x] Validate Wave 1 with targeted fmt, `./build.sh typecheck`, and the narrowest truthful broader gate that still covers the changed surfaces.

### Notes
- Opened on branch `refactor/wave1-feature-model` on March 24, 2026 as the first implementation wave under the new roadmap.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this wave proceeds with the ABI best-effort fallback.
- This wave uses parallel agents: one implementation worker for feature-model normalization, one explorer for the `pages` boundary decision, and one explorer for future `build.zig` decomposition seams.
- Explorer outcome: keep `pages` as a separately gated observability sub-feature for Wave 1. Folding it into observability would widen the public API/config/test blast radius too far for this pass.
- Compatibility decision: `feat_observability` is now the canonical build flag surface and `-Dfeat-profiling` remains accepted as a deprecated CLI alias in `build.zig`.
- The current repo no longer contains stray `.DS_Store` files under the root, `test/`, `docs/`, or `zig-abi-plugin/`.
- Validation passed with `$(./tools/zigly --status) fmt --check build.zig src/root.zig src/core/comptime_meta.zig src/core/config/mod.zig src/core/framework/context_init.zig src/core/framework/feature_imports.zig src/core/feature_catalog.zig src/protocols/mcp/real.zig src/foundation/utils/config.zig src/features/observability/mod.zig src/features/observability/stub.zig src/features/ai/metrics.zig src/features/ai/profiles/mod.zig src/features/ai/streaming/metrics.zig src/features/ai/streaming/server/mod.zig src/features/ai/streaming/server/handlers.zig`, `./build.sh --help`, `./build.sh typecheck`, `./build.sh validate-flags`, `./build.sh full-check`, and `git diff --check`.
- Next build-system refactor seam from the explorer: extract `FeatureFlags`, `hasBackend`, and `addAllBuildOptions` into `build/flags.zig` before attempting larger `build.zig` splits.

## 19. Build Flags Extraction
- [x] Extract `FeatureFlags`, `hasBackend`, and `addAllBuildOptions` from `build.zig` into `build/flags.zig`.
- [x] Extend formatter coverage so tracked build helper files are included in `lint` / `fix`.
- [x] Validate the extraction with build help, `typecheck`, and `full-check`.

### Notes
- Completed on branch `refactor/wave1-feature-model` immediately after Wave 1 landed, using the staged split recommendation from the build explorer.
- `build.zig` remains the sole entrypoint that owns step names and build graph wiring; `build/flags.zig` only contains pure flag/build-option helpers.
- Validation passed with `$(./tools/zigly --status) fmt --check build.zig build/flags.zig`, `./build.sh --help`, `./build.sh typecheck`, `./build.sh full-check`, and `git diff --check`.

## 20. Wave 2 Next-Step Plan
- [x] Extract repeated Darwin/native linking blocks from `build.zig` into a focused helper such as `build/linking.zig`, shared by the library, executables, and test roots without changing step names. *(Done in §21)*
- [x] Extract validation-step construction from `build.zig` into a helper such as `build/validation.zig`, covering `test`, `tui-tests`, `check-parity`, `feature-tests`, `typecheck`, and the alias wiring while preserving the current command contract. *(Done in §21)*
- [x] Extract the cross-target feature matrix and compile-only object setup into a helper such as `build/cross.zig` or `build/targets.zig`, keeping the current compile-only semantics and platform policy comments intact. *(Done in §21)*
- [x] Add one or two truthful targeted lanes for the first post-build refactor target so medium-sized module splits do not require the full-suite path on every edit. *(Done in §21-29: 27 lanes)*
- [x] After the build helpers and narrower lanes land, start the first runtime module decomposition on a bounded hotspot with clear seams, preferring `src/protocols/mcp/server.zig`, `src/inference/engine.zig`, or `src/features/gateway/mod.zig` before larger AI/GPU codegen surfaces. *(Done in §21-29)*

### Notes
- Planned on March 24, 2026 after the `build/flags.zig` extraction; this is a planning-only update with no source changes beyond the roadmap.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this plan records the ABI best-effort fallback.
- Ordering rationale: `build.zig` is still 590 lines and still owns repeated Darwin linking, test-root wiring, cross-target policy setup, and validation aliases. That makes build-helper extraction the highest-leverage remaining item from Section 17.
- Defer broad AI/GPU monolith splits until targeted lanes exist. Without narrower validation, each decomposition would keep paying the `full-check` cost and make regressions harder to isolate.
- Baseline evidence for this planning pass: `./build.sh typecheck --summary all` and `./build.sh full-check` both passed on the current `refactor/wave1-feature-model` head, and `git status --short` was clean before the roadmap edit.
- Residual risk: moving build helpers can easily regress Darwin framework linkage or cross-target compile-only guarantees, so each Wave 2 slice should keep `build.zig` as the single graph-entry surface and validate with `./build.sh --help`, `./build.sh typecheck`, one cross-target `typecheck`, `./build.sh full-check`, and `git diff --check`.

## 21. Wave 2 Continuation Implementation
- [x] Extract Darwin/native linking into `build/linking.zig` with role-based helpers that preserve the current framework matrix.
- [x] Extract test/validation step construction into `build/validation.zig` without changing existing public step names.
- [x] Extract compile-only `typecheck` and the cross-target matrix into `build/cross.zig`, building target options via `build/flags.zig`.
- [x] Add a truthful `mcp-tests` lane backed by a dedicated `test/mcp_mod.zig` root and update docs/help expectations.
- [x] Split `src/protocols/mcp/server.zig` into a facade plus internal `server/` helpers while preserving behavior and tests.
- [x] Validate with help, host+cross `typecheck`, feature/flag gates, `mcp-tests`, `full-check`, and `verify-all` if the host remains link-clean.

### Notes
- Opened on March 24, 2026 from the current clean `main` baseline to implement the previously planned Wave 2 continuation.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for this wave.
- `build.zig` is now a thinner graph entrypoint, with Darwin linkage in `build/linking.zig`, validation/test graph wiring in `build/validation.zig`, and compile-only plus cross-target matrix setup in `build/cross.zig`.
- `mcp-tests` now runs through a dedicated `test/mcp_mod.zig` integration root and is documented alongside the other public validation gates in `README.md`, `CLAUDE.md`, and `AGENTS.md`.
- `src/protocols/mcp/server.zig` now keeps the public `Server` surface while delegating I/O, dispatch, tool handling, resource handling, and JSON escaping to `src/protocols/mcp/server/io_loop.zig`, `dispatch.zig`, `tools.zig`, `resources.zig`, and `json_write.zig`.
- Validation passed with `./build.sh --help`, `$(./tools/zigly --status) fmt --check build.zig build/linking.zig build/cross.zig build/validation.zig test/mcp_mod.zig src/protocols/mcp/server.zig src/protocols/mcp/server/io_loop.zig src/protocols/mcp/server/json_write.zig src/protocols/mcp/server/tools.zig src/protocols/mcp/server/resources.zig src/protocols/mcp/server/dispatch.zig`, `./build.sh typecheck --summary all`, `./build.sh -Dtarget=x86_64-linux-gnu typecheck --summary all`, `./build.sh mcp-tests --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh test --summary all`, `./build.sh full-check --summary all`, and `./build.sh verify-all --summary all`.

## 22. Wave 3A Gateway Decomposition
- [x] Add a truthful `gateway-tests` step in `build/validation.zig` with a gateway unit root and a dedicated gateway integration root.
- [x] Add `test/gateway_mod.zig` plus deeper public-runtime gateway integration coverage for lifecycle, route removal/reindexing, dispatch success, rate-limit rejection, circuit-breaker rejection, and upstream failure recording.
- [x] Split `src/features/gateway/mod.zig` into a public facade backed by `state.zig`, `routes.zig`, and `pipeline.zig` without changing the `abi.gateway` public surface.
- [x] Preserve `src/features/gateway/stub.zig` parity and keep feature-gated import boundaries relative within `src/features/gateway/`.
- [x] Validate with targeted fmt, `./build.sh gateway-tests --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- Opened on March 24, 2026 on top of the active Wave 2 worktree after `./build.sh --help`, `./build.sh full-check --summary all`, and `git diff --check` passed on the current base.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this slice proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for Wave 3.
- Keep gateway public names stable: `Context`, `GatewayConfig`, `GatewayError`, `HttpStatus`, `RequestResult`, and the existing lifecycle/dispatch/stat functions stay exported from `mod.zig`.
- `src/features/gateway/mod.zig` is now a thin facade over `state.zig`, `routes.zig`, and `pipeline.zig`, while `gateway-tests` exercises a `src/`-anchored unit wrapper plus `test/gateway_mod.zig` because Zig's module-path rules do not allow `src/features/gateway/mod.zig` to be tested as the literal root when it imports `../../foundation/...`.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/gateway_mod_test.zig src/features/gateway/mod.zig src/features/gateway/state.zig src/features/gateway/routes.zig src/features/gateway/pipeline.zig test/gateway_mod.zig test/integration/gateway_runtime_test.zig`, `./build.sh gateway-tests --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 23. Wave 3B Inference Decomposition
- [x] Add a truthful `inference-tests` step in `build/validation.zig` with an inference unit root and a dedicated inference integration root.
- [x] Add `test/inference_mod.zig` plus deeper public async integration coverage for `generateAsyncWithTimeout`, timeout/abandon cleanup, shutdown waiting for in-flight async work, and returned-result ownership.
- [x] Split `src/inference/engine.zig` into a public facade backed by `engine/types.zig`, `engine/backends.zig`, and `engine/async.zig` without changing the `abi.inference` public surface.

## 24. Wave 3C Multi-Agent Coordinator Surface Split
- [x] Move the canonical shared public multi-agent types into `src/features/ai/multi_agent/types.zig`, including coordinator and runner-facing types that `mod.zig` currently owns directly. *(Done in §29)*
- [x] Extract coordinator runtime behavior from `src/features/ai/multi_agent/mod.zig` into a new `src/features/ai/multi_agent/coordinator/` helper set while keeping imports relative within `src/`. *(Done in §29)*
- [x] Reduce `src/features/ai/multi_agent/mod.zig` and `src/features/ai/multi_agent/stub.zig` to thin facades/re-exports that preserve the existing `abi.ai.multi_agent` source surface. *(Done in §29)*
- [x] Validate with targeted formatting/parity/typecheck commands and record residual risk without editing `runner.zig` or adding tests. *(Done in §29)*

### Notes
- Opened on March 24, 2026 in `/Users/donaldfilimon/abi` with `git status --short` showing only the unrelated untracked `.claude/worktrees/` path before implementation.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this slice proceeds with the ABI best-effort fallback.
- Scope guard: only `src/features/ai/multi_agent/mod.zig`, `types.zig`, `stub.zig`, and new files under `src/features/ai/multi_agent/coordinator/` are in scope for this worker; `runner.zig` ownership stays with another worker.
- [x] Keep `src/inference/mod.zig` as the stable public re-export surface for `Engine`, `EngineConfig`, `EngineResult`, `EngineStats`, `FinishReason`, and `Backend`.
- [x] Validate with targeted fmt, `./build.sh inference-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh test --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- This slice starts only after the gateway lane is stable so the new targeted gateway gate can serve as the first Wave 3 checkpoint.
- Reuse the existing Wave 2 build helper split instead of expanding `build.zig` again beyond the new targeted test-step wiring.
- Keep inference behavior stable across demo, connector, local, and async ownership/lifecycle paths; avoid changing request/result wire semantics in this wave.
- `src/inference/engine.zig` now re-exports stable public types from `engine/types.zig` and delegates backend generation plus async lifecycle to `engine/backends.zig` and `engine/async.zig`; `src/inference/mod.zig` stayed unchanged as the public re-export surface.
- `inference-tests` uses a `src/inference_mod_test.zig` unit wrapper for the same module-path reason as the gateway lane, and `test/integration/inference_async_test.zig` now covers ownership via `waitTimeout() + Result.deinit() + AsyncResult.destroy()`, timeout-abandon cleanup, and shutdown waiting for in-flight callbacks.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/gateway_mod_test.zig src/inference_mod_test.zig src/features/gateway/mod.zig src/features/gateway/state.zig src/features/gateway/routes.zig src/features/gateway/pipeline.zig src/inference/engine.zig src/inference/engine/types.zig src/inference/engine/backends.zig src/inference/engine/async.zig test/gateway_mod.zig test/inference_mod.zig test/integration/gateway_runtime_test.zig test/integration/inference_async_test.zig`, `./build.sh inference-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh test --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 24. Wave 3C Messaging Decomposition
- [x] Add a truthful `messaging-tests` step in `build/validation.zig` with a `src/messaging_mod_test.zig` unit root and a focused `test/messaging_mod.zig` integration root.
- [x] Move the inline messaging tests out of `src/features/messaging/mod.zig` so the feature facade stays focused on the runtime surface instead of carrying 20+ embedded test cases.
- [x] Split `src/features/messaging/mod.zig` into a thin facade backed by focused lifecycle, subscription, and query/stat helpers while preserving the current `abi.messaging` public API.
- [x] Preserve `src/features/messaging/stub.zig` parity and keep all cross-file imports within `src/features/messaging/` relative.
- [x] Document the new `messaging-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh messaging-tests --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- Added on March 24, 2026 as a planning-only continuation after the active gateway and inference decomposition waves.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this plan update follows the ABI best-effort fallback.
- The ABI review prep helper is still blocked because this checkout lacks `src/abi.zig`; that startup limitation remains recorded in `tasks/lessons.md`.
- `src/features/messaging/mod.zig` is the next clean seam because it still carries the public facade plus roughly 20 inline tests in one 700+ line file, while the domain already has internal `state.zig`, `delivery.zig`, and `pattern_matching.zig` helpers ready to support a thinner entrypoint.
- `build/validation.zig` already has the gateway and inference focused-lane pattern, so `messaging-tests` can reuse that structure without another broad build-graph redesign.
- After messaging, the next large mixed-responsibility hotspots still queued for later waves are `src/foundation/security/secrets.zig` and `src/protocols/ha/pitr.zig`; they stay deferred until the messaging lane is stable.
- Existing unrelated worktree changes in `src/features/benchmarks/mod.zig`, `src/features/benchmarks/suite.zig`, and `.claude/worktrees/` remain out of scope and must not be reverted or swept into this wave.
- `src/features/messaging/mod.zig` now delegates lifecycle, publish, subscription, and query/stat behavior to `src/features/messaging/lifecycle.zig`, `publish.zig`, `subscriptions.zig`, and `queries.zig`, while the focused unit coverage lives in `src/features/messaging/tests.zig`.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/features/messaging/mod.zig src/features/messaging/lifecycle.zig src/features/messaging/publish.zig src/features/messaging/subscriptions.zig src/features/messaging/queries.zig src/features/messaging/tests.zig src/messaging_mod_test.zig test/messaging_mod.zig`, `./build.sh messaging-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 25. Wave 4A Secrets Decomposition
- [x] Add a truthful `secrets-tests` step in `build/validation.zig` with a `src/secrets_mod_test.zig` unit root and a focused `test/secrets_mod.zig` integration root.
- [x] Split `src/foundation/security/secrets.zig` into a thin public facade over focused helpers for shared types, validation, provider loading, provider saving, and persistence/file parsing while preserving the current `foundation.security` public surface.
- [x] Move the inline secrets tests out of `src/foundation/security/secrets.zig` into focused unit/integration roots.
- [x] Keep `src/foundation/security/mod.zig` re-export names stable for `SecretsManager`, `SecretsConfig`, `SecretValue`, `SecretMetadata`, `SecretType`, `SecureString`, and `SecretsError`.
- [x] Document the new `secrets-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh secrets-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh test --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- This slice follows messaging and reuses the same focused-lane build pattern rather than widening `build.zig` again.
- `test/integration/security_test.zig` currently exercises constitution/router behavior, not the secrets manager, so this wave needs a dedicated secrets integration root instead of reusing the existing security lane.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for this wave.
- `src/foundation/security/secrets.zig` now re-exports the public secrets types from `src/foundation/security/secrets/shared.zig` and delegates validation, persistence, and provider logic to `validation.zig`, `persistence.zig`, and `providers.zig`; focused unit coverage now lives in `src/foundation/security/secrets/tests.zig`.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/foundation/security/secrets.zig src/foundation/security/secrets/shared.zig src/foundation/security/secrets/validation.zig src/foundation/security/secrets/persistence.zig src/foundation/security/secrets/providers.zig src/foundation/security/secrets/tests.zig src/secrets_mod_test.zig test/integration/secrets_test.zig test/secrets_mod.zig`, `./build.sh secrets-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh test --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 26. Wave 4B PITR Decomposition
- [x] Add a truthful `pitr-tests` step in `build/validation.zig` with a `src/pitr_mod_test.zig` unit root and a focused `test/pitr_mod.zig` integration root.
- [x] Split `src/protocols/ha/pitr.zig` into a thin public facade over focused helpers for operation capture, recovery, persistence, and retention/event emission while preserving the current `abi.ha` PITR surface.
- [x] Move the inline PITR tests out of `src/protocols/ha/pitr.zig` into focused unit/integration roots.
- [x] Preserve `src/protocols/ha/stub.zig` parity for PITR declarations and keep `src/protocols/ha/mod.zig` as the stable public re-export surface.
- [x] Document the new `pitr-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh pitr-tests --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- `test/integration/ha_test.zig` is intentionally broader than PITR, so this wave uses a dedicated PITR-focused integration root instead of treating the full HA suite as a truthful narrow gate.
- Keep the PITR public names stable: `PitrManager`, `PitrConfig`, `RecoveryPoint`, `RecoveryResult`, `Operation`, and related events/config types remain exported where they are today.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for this wave.
- `src/protocols/ha/pitr.zig` now keeps the public PITR types and `PitrManager` surface while delegating capture, recovery, persistence, and retention/event behavior to `src/protocols/ha/pitr/capture.zig`, `recovery.zig`, `persistence.zig`, and `retention.zig`; focused unit coverage now lives in `src/protocols/ha/pitr/tests.zig`.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/protocols/ha/pitr.zig src/protocols/ha/pitr/common.zig src/protocols/ha/pitr/capture.zig src/protocols/ha/pitr/recovery.zig src/protocols/ha/pitr/persistence.zig src/protocols/ha/pitr/retention.zig src/protocols/ha/pitr/tests.zig src/pitr_mod_test.zig test/integration/pitr_test.zig test/pitr_mod.zig`, `./build.sh pitr-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 27. Wave 5A AI Agents Decomposition
- [x] Add a truthful `agents-tests` step in `build/validation.zig` with a `src/agents_mod_test.zig` unit root and a focused `test/agents_mod.zig` integration root.
- [x] Promote `src/features/ai/agents/types.zig` to the shared public constants/enums/config/message/error surface used by both the real and stub agent implementations.
- [x] Reduce `src/features/ai/agents/agent.zig` to lifecycle/history orchestration and move backend dispatch plus provider/HTTP generation helpers into focused `src/features/ai/agents/agent/` helpers.
- [x] Move the inline agent tests out of `src/features/ai/agents/agent.zig` into focused unit/integration roots.
- [x] Keep `src/features/ai/agents/mod.zig`, `src/features/ai/agents/stub.zig`, `Agent`, `AgentBackend`, `AgentConfig`, `Context`, and the existing tool-registry entrypoints stable.
- [x] Document the new `agents-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh agents-tests --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- `test/integration/ai_test.zig` is a broad AI-facade suite, so this wave needs a dedicated agents integration root to keep the focused gate truthful.
- `src/features/ai/agents/types.zig` already overlaps heavily with `agent.zig`, making it the correct consolidation target for shared public declarations instead of duplicating the surface again.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for this wave.
- `src/features/ai/agents/agent.zig` now keeps `Agent` lifecycle, history, cognition, and stats orchestration only, while `src/features/ai/agents/agent/dispatch.zig`, `http_backends.zig`, `providers.zig`, and `payloads.zig` own backend routing, HTTP/provider calls, and request-payload assembly.
- `src/features/ai/agents/types.zig` is now the shared constants/config/error/message surface for both real and stub builds, and the focused coverage moved to `src/features/ai/agents/tests.zig`, `src/agents_mod_test.zig`, `test/agents_mod.zig`, and `test/integration/agents_test.zig`.
- The agents context now owns registered tool copies before placing them into the registry so the stable `registerTool()` entrypoint no longer stores stack-backed tool pointers.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/features/ai/agents/mod.zig src/features/ai/agents/stub.zig src/features/ai/agents/types.zig src/features/ai/agents/agent.zig src/features/ai/agents/agent/dispatch.zig src/features/ai/agents/agent/http_backends.zig src/features/ai/agents/agent/payloads.zig src/features/ai/agents/agent/providers.zig src/features/ai/agents/tests.zig src/agents_mod_test.zig test/agents_mod.zig test/integration/agents_test.zig`, `./build.sh agents-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 28. Wave 5B Orchestration Decomposition
- [x] Add a truthful `orchestration-tests` step in `build/validation.zig` with a `src/orchestration_mod_test.zig` unit root and a focused `test/orchestration_mod.zig` integration root.
- [x] Promote `src/features/ai/orchestration/types.zig` to the shared public type/config/error surface used by both the real and stub orchestration implementations.
- [x] Reduce `src/features/ai/orchestration/mod.zig` to a stable public facade and move orchestrator runtime behavior into focused internal helpers under `src/features/ai/orchestration/orchestrator/`.
- [x] Move the inline orchestration tests out of `src/features/ai/orchestration/mod.zig` into focused unit/integration roots.
- [x] Keep `src/features/ai/orchestration/mod.zig`, `src/features/ai/orchestration/stub.zig`, `Router`, `RoutingStrategy`, `TaskType`, `RouteResult`, `Ensemble`, `EnsembleMethod`, `EnsembleResult`, `FallbackManager`, `FallbackPolicy`, `HealthStatus`, `Orchestrator`, `OrchestrationConfig`, `ModelBackend`, `Capability`, `ModelConfig`, `ModelEntry`, `OrchestratorStats`, and `isEnabled` source-compatible.
- [x] Document the new `orchestration-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh orchestration-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- Opened on March 24, 2026 from local `main` at `9658e74`, which is one commit ahead of `origin/main`; keep the untracked `.claude/worktrees/` directory out of scope.
- `src/features/ai/orchestration/mod.zig` is the next clean seam because it still mixes the public facade, shared public types, runtime execution, selection logic, and inline tests in one 900+ line file, while `router.zig`, `ensemble.zig`, and `fallback.zig` already provide stable domain helpers to preserve.
- `src/features/ai/orchestration/types.zig` currently mirrors the public surface with stub-oriented placeholders, making it the right consolidation point for the real/stub shared public API without widening the caller-facing contract.
- The multi-CLI consensus helper remains unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- The ABI review prep helper remains blocked because this checkout still lacks `src/abi.zig`; that blocker stays out of scope for this wave.
- `src/features/ai/orchestration/types.zig` now holds the shared public config/error/enum/state surface for both real and stub builds, while `src/features/ai/orchestration/mod.zig` became a thin facade over `router.zig`, `ensemble.zig`, `fallback.zig`, and the new `src/features/ai/orchestration/orchestrator/{mod,registry,selection,execution}.zig` helper set.
- Focused orchestration coverage now lives in `src/features/ai/orchestration/tests.zig`, `src/orchestration_mod_test.zig`, `test/orchestration_mod.zig`, and `test/integration/orchestration_test.zig`, including fake-dispatch coverage for single-execution and fallback paths without live provider dependencies.
- Validation passed with `$(./tools/zigly --status) fmt --check build/validation.zig src/features/ai/orchestration/mod.zig src/features/ai/orchestration/types.zig src/features/ai/orchestration/router.zig src/features/ai/orchestration/ensemble.zig src/features/ai/orchestration/fallback.zig src/features/ai/orchestration/stub.zig src/features/ai/orchestration/orchestrator/mod.zig src/features/ai/orchestration/orchestrator/registry.zig src/features/ai/orchestration/orchestrator/selection.zig src/features/ai/orchestration/orchestrator/execution.zig src/features/ai/orchestration/tests.zig src/orchestration_mod_test.zig test/mod.zig test/orchestration_mod.zig test/integration/orchestration_test.zig`, `./build.sh orchestration-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

## 29. Wave 5C Multi-Agent Decomposition
- [x] Add a truthful `multi-agent-tests` step in `build/validation.zig` with a `src/multi_agent_mod_test.zig` unit root and a focused `test/multi_agent_mod.zig` integration root.
- [x] Promote `src/features/ai/multi_agent/types.zig` to the shared public type/config/error surface used by both the real and stub multi-agent implementations.
- [x] Reduce `src/features/ai/multi_agent/mod.zig` to a stable public facade and move coordinator runtime behavior into focused internal helpers under `src/features/ai/multi_agent/coordinator/`.
- [x] Move the inline multi-agent tests out of `src/features/ai/multi_agent/mod.zig` and `src/features/ai/multi_agent/runner.zig` into focused unit/integration roots.
- [x] Keep `src/features/ai/multi_agent/mod.zig`, `src/features/ai/multi_agent/stub.zig`, `Coordinator`, `CoordinatorConfig`, `CoordinatorStats`, `WorkflowRunner`, `RunnerConfig`, `WorkflowResult`, `WorkflowStats`, `StepResult`, `ExecutionStrategy`, `AggregationStrategy`, `AgentResult`, `AgentHealth`, `Error`, `RunError`, and `isEnabled` source-compatible.
- [x] Document the new `multi-agent-tests` lane in `README.md`, `CLAUDE.md`, and `AGENTS.md` once the build step exists.
- [x] Validate with targeted fmt, `./build.sh multi-agent-tests --summary all`, `./build.sh typecheck --summary all`, `./build.sh check-parity --summary all`, `./build.sh validate-flags --summary all`, `./build.sh feature-tests --summary all`, `./build.sh full-check --summary all`, `./build.sh verify-all --summary all`, and `git diff --check`.

### Notes
- Opened on March 24, 2026 from clean local `main` at `4669f81`, with only the untracked `.claude/worktrees/` directory remaining out of scope.
- Tests were successfully decoupled into `src/features/ai/multi_agent/tests.zig` and integration points configured via `src/multi_agent_mod_test.zig`.
- Type extraction was performed safely without breaking `multi_agent.WorkflowRunner.RunError` namespace expectations by mirroring types at the top level of the facade to reduce nested structs.
- Validation successfully verified parity on `0.17.0` Zig using the updated local runner logic for macOS SDK wrappers without any regression errors.

## 14. Workflow Lessons and Documentation Improvements
- [x] Review and update `tasks/lessons.md` with additional patterns from AGENTS.md
- [x] Add lessons for import patterns, code style, module decomposition, and test organization
- [x] Update `AGENTS.md` with GPU patterns section showing VTable pattern
- [x] Add module decomposition best practices to AGENTS.md
- [x] Add common pitfalls section to AGENTS.md covering circular imports, memory ownership, and thread safety

### Notes
- Added Lessons 11-20 to `tasks/lessons.md` covering import patterns, code style, memory patterns, error handling, module decomposition, VTable patterns, and test organization
- Updated `AGENTS.md` with GPU patterns section showing the VTable pattern for backend-agnostic interfaces
- Added module decomposition best practices and common pitfalls sections
- All changes verified with `./build.sh lint` and `./build.sh fix` for formatting

## 15. Aggressive Module Decomposition Wave
- [x] Decompose `multi_agent/workflow.zig` (743 lines) into `workflow/types.zig`, `workflow/definition.zig`, `workflow/tracker.zig`, `workflow/presets.zig`
- [x] Decompose `multi_agent/roles.zig` (467 lines) into `roles/types.zig`, `roles/presets.zig`, `roles/registry.zig`
- [x] Decompose `gpu/fusion.zig` (1066 lines) into `fusion/types.zig`, `fusion/detection.zig` (extracted 13 pattern-detection functions)
- [x] Decompose `foundation/security/jwt.zig` (1391 lines) into `jwt/types.zig`, `jwt/manager.zig`, `jwt/standalone.zig`
- [x] Fix pre-existing `src/tasks/roadmap.zig` compilation error (missing `persistence.zig` import, undefined `dupeString`)
- [x] Verify with `./build.sh typecheck`, `./build.sh test --summary all`, and `./build.sh full-check --summary all`

### Notes
- Opened on March 26, 2026 as an aggressive decomposition wave targeting the largest monolithic files in the codebase.
- `multi_agent/workflow.zig` (743 lines) split into 4 sub-modules: types (90 lines), definition (145 lines), tracker (190 lines), presets (130 lines). Parent file is now ~160 lines (thin re-export + tests).
- `multi_agent/roles.zig` (467 lines) split into 3 sub-modules: types (170 lines), presets (110 lines), registry (100 lines). Parent file is now ~140 lines (thin re-export + tests).
- `gpu/fusion.zig` (1066 lines) split into 2 sub-modules: types (250 lines with OpType, BufferHandle, OpNode, FusionPattern, FusionStats, calculateSpeedup), detection (390 lines with 13 extracted pattern-detection functions). Parent file is now ~430 lines (FusionOptimizer + tests). Detection functions were extracted as free functions taking nodes/buffer_refs/patterns parameters instead of methods on FusionOptimizer.
- `foundation/security/jwt.zig` (1391 lines) split into 3 sub-modules: types (200 lines with Algorithm, Claims, Token, JwtConfig, JwtError, utility functions), manager (530 lines with JwtManager struct), standalone (300 lines with decode/verify/base64Url* functions). Parent file is now ~400 lines (thin re-export + tests).
- Pre-existing bug: `src/tasks/roadmap.zig` imported `persistence.zig` (missing file) and used undefined `dupeString`. Fixed by removing the import and adding `dupeString` inline.
- Validation passed with `./build.sh typecheck --summary all`, `./build.sh test --summary all` (3496/3500 passed, 4 skipped), and `./build.sh full-check --summary all` (3742/3746 passed, 4 skipped).
- Remaining large files for future decomposition waves: `ai/streaming/server/mod.zig` (1105 lines), `security/password.zig` (1125 lines), `gpu/device.zig` (1002 lines), `ai/training/llm_trainer.zig` (1146 lines).
- JWT module has duplicated parsing logic between manager.zig and standalone.zig (parseHeader/parseHeaderStandalone, parseClaims/parseClaimsStandalone, etc.). Future deduplication opportunity.

## 30. Core Path Compatibility Bridge
- [x] Restore the tracked `src/core/**` surface as a compatibility layer to `src/features/core/**` without changing external callers.
- [x] Fix the stale `core/...` imports inside `src/features/core/database/**` so the moved tree is internally consistent.
- [x] Validate with targeted fmt, `./build.sh typecheck --summary all`, `./build.sh check-parity`, and `git diff --check`.

### Notes
- Opened on March 27, 2026 from local `main` with a dirty worktree that already includes unrelated AI/GPU/network edits and the untracked `.claude/worktrees/` directory; those remain out of scope.
- The multi-CLI consensus helper is unavailable in this checkout (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` missing), so this task proceeds with the ABI best-effort fallback.
- Baseline failures reproduced with `./build.sh typecheck --summary all` and `./build.sh check-parity`: both stopped on 24 `FileNotFound` imports under `src/core/**` after the implementation tree was moved to `src/features/core/**`.
- The compatibility layer landed as path-matched symlinks for all 158 legacy `src/core/**` files because Zig rejected the attempted file-scope wrapper form; this keeps the old import contract intact while leaving `src/features/core/**` as the source tree.
- Internal cleanup was limited to the 13 stale `core/...` imports in `src/features/core/database/**`, switching them to direct local paths like `../mod.zig` and `../config/mod.zig` instead of adding nested compatibility shims.
- Validation passed with `zig fmt --check src/core src/features/core/database`, `./build.sh typecheck --summary all`, `./build.sh check-parity`, and `git diff --check`.
- Residual risk: the bridge is filesystem-level compatibility, so a future cleanup wave should decide whether to keep symlinks or complete the migration to direct `src/features/core/**` imports before broadening the validation surface further.

## 31. Documentation Count Sync + Cleanup
- [x] Sync `README.md` and `GEMINI.md` with the current `src/features/` layout count and feature catalog count.
- [x] Audit the remaining docs and plan files for stale `20 feature directories`, `35 features total`, or other hardcoded count strings and update or mark them as historical context.
- [x] Prefer derived count sources from `src/core/feature_catalog.zig` for any user-facing feature inventory text that still hardcodes counts.
- [x] Re-run grep-based verification across `AGENTS.md`, `CLAUDE.md`, `README.md`, `GEMINI.md`, and `docs/superpowers/` to confirm the count drift is gone.

### Notes
- Completed after updating `README.md`, `GEMINI.md`, `docs/superpowers/specs/2026-03-24-full-codebase-improvement-design.md`, `docs/superpowers/plans/2026-03-24-full-codebase-improvement.md`, and `docs/superpowers/plans/2026-03-27-codebase-improvement-remaining.md` to match the current repository counts.
- `docs/onboarding.md` was also updated locally, but it is still ignored by `.gitignore` and needs allowlist work before it can be tracked.
- The current repository now documents 21 `src/features/` directories and 60 catalog features consistently across the tracked user-facing docs and the supporting planning/spec artifacts.
- Keep the unrelated dirty worktree changes out of scope for this wave.
- Verification command to re-run when needed: `rg -n "20 feature directories|35 features total|30 features" AGENTS.md CLAUDE.md README.md GEMINI.md docs/onboarding.md docs/superpowers`, then `git diff --check`.

## 32. Docs Allowlist + Onboarding Tracking
- [x] Add `!/docs/onboarding.md` to the markdown allowlist in `.gitignore` so the onboarding guide can be tracked.
- [x] Decide whether `docs/onboarding.md` should remain the canonical newcomer guide or be replaced by a shorter pointer from `docs/README.md` / `docs/STRUCTURE.md`.
- [x] Re-run `git check-ignore -v docs/onboarding.md` and `git status --short docs` to confirm the onboarding file is no longer ignored.
- [x] Re-run the docs count grep sweep across `docs/` after onboarding is tracked.

### Notes
- `docs/onboarding.md` added to allowlist and verified with `git check-ignore`.
- Decision: `docs/onboarding.md` remains the canonical newcomer guide.
- Docs count sweep across `AGENTS.md`, `CLAUDE.md`, `README.md`, `GEMINI.md`, and `docs/superpowers/` confirmed no count drift remains (all match 21 directories / 60 features).
- The onboarding guide is the only Markdown file in `docs/` that is currently ignored, so the next step is to make an explicit tracking decision instead of leaving it as a local-only edit.
- Historical review/spec docs under `docs/review/` and `docs/spec/` can keep their archival counts unless we intentionally restate them as live documentation.
- Keep the unrelated dirty worktree changes out of scope for this wave.

## 33. TUI Live Metrics Panel
- [x] Add a live metrics section to the TUI dashboard overview so the shell exposes runtime health instead of only static catalog data.
- [x] Render at least one live gauge or counter derived from current feature/runtime enablement, and keep it resize-safe in compact/medium layouts.
- [x] Add integration coverage for the new metrics surface and verify the TUI lane still passes.
- [x] Keep the dashboard backwards compatible for existing overview/features/runtime navigation.

### Notes
- Completed by adding a METRICS overview section with live feature-coverage and runtime-surface gauges in `src/features/tui/app/dashboard.zig`.
- The TUI lane passed after the change with `./build.sh tui-tests --summary all`, and `git diff --check` remained clean.
- This fills the most concrete, low-risk part of the TUI live-metrics gap without changing chat/database/log-viewer behavior yet.
- Keep the unrelated dirty worktree changes out of scope for this wave.

## 34. Source Layout Canonicalization
- [x] Migrate internal callers off the `src/core` compatibility tree and onto `src/features/core`.
- [x] Retire the `src/core` symlink bridge once the import sweep is clean.
- [x] Validate the post-migration tree with parity, typecheck, and the most affected feature lanes.
- Validation: `~/.zvm/bin/zig build check-parity`, `~/.zvm/bin/zig build typecheck --summary all`, `~/.zvm/bin/zig build agents-tests --summary all`, `~/.zvm/bin/zig build tui-tests --summary all`, and `~/.zvm/bin/zig build mcp-tests --summary all` passed. `database-tests` still fails on `test.database: chain integrity across multiple blocks` in both the current tree and the merge-base commit.

### Notes
- Opened on April 4, 2026 in `/Users/donaldfilimon/abi` as a follow-up organization wave.
- This wave should preserve the public `abi.*` surface while making `src/features/core/**` the canonical internal layout.
- The current dirty worktree contains unrelated edits and must remain untouched outside the source-layout sweep.
