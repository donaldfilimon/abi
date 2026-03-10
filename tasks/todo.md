# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Active — Hard-Cut CEL + Bootstrap Removal (2026-03-10 17:26 EDT)

- [x] Consensus status: best-effort tri-CLI wrapper unavailable; `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` is missing locally.
- [x] Review base recorded for post-change ABI review prep: `d76dc9d9b1c9e0fc5be5b82f3574f1fd1d542712`.
- [x] Inventory captured the tracked CEL/bootstrap surface: `./cel`, `.cel/`, `.zig-bootstrap/`, `cel.toml`, `cel.lock`, `tools/cel/`, `src/cel/`, `examples/cel/`, `tests/cel/`, `stdlib/cel/`, bootstrap build wiring, wrapper scripts, CLI commands, docs metadata, plugin guidance, `.claude` hooks, and LSP/toolchain fallback paths.
- [x] Confirmed the worktree already contained partial bootstrap-migration edits in `.claude/settings.json`, `src/core/config/{loader,lsp}.zig`, `src/services/{lsp/client.zig,shared/utils/zig_toolchain.zig}`, `src/features/gpu/dsl/spirv.zig`, `tools/cli/commands/dev/lsp.zig`, and toolchain scripts; this wave must fold those edits into the hard-cut removal instead of preserving a `.zig-bootstrap` survivor surface.
- [x] Delete the tracked CEL/bootstrap roots and obsolete scripts/commands in one wave: `.cel/`, `.zig-bootstrap/`, `cel`, `cel.toml`, `cel.lock`, `tools/cel/`, `src/cel/`, `examples/cel/`, `tests/cel/`, `stdlib/cel/`, `build/cel.zig`, `tools/scripts/{cel_doctor.zig,cel_migrate.sh,use_cel.sh,use_zig_bootstrap.sh,zig_bootstrap_migrate.sh}`, and `tools/cli/commands/dev/toolchain.zig`.
- [x] Remove bootstrap build-step wiring and CLI registry/docs coupling from `build.zig`, `tools/cli/generated/cli_registry_snapshot.zig`, `tools/cli/commands/mod.zig`, `tools/cli/registry/overrides.zig`, docs metadata, launcher/help text, and CLI tests.
- [x] Rewire surviving toolchain and LSP behavior to Zig-only defaults: no `.cel` / `.zig-bootstrap` probes, no bootstrap precedence checks, Zig fallback via ZVM master or `zig` on `PATH`, ZLS via explicit override or `zls` on `PATH`.
- [x] Scrub live and archived repo text so repo-tracked content no longer presents CEL/bootstrap as supported workflow, including `README.md`, `CLAUDE.md`, `docs/`, `.claude/`, `zig-abi-plugin/`, and `tasks/`.
- [x] Validation + review: run `git diff --check`, `rg` absence sweeps for `.cel` / `.zig-bootstrap` / `bootstrap-zig` / `toolchain` alias references, repo-safe format checks, targeted compile-only probes for touched Zig surfaces, and record Darwin build-runner blockers for any binary-emitting gates that still cannot run locally.
- [x] Review notes (2026-03-10 15:42 EDT): fixed-string absence sweeps found no live `.zig-bootstrap`, `bootstrap-zig`, `cel.toml`, `tools/cel`, or `src/cel` references outside this bookkeeping entry and the intentionally deleted files in this wave.
- [x] Validation evidence (2026-03-10 15:42 EDT): `git diff --check`; `zig fmt --check build.zig build src tools examples`; `zig test src/services/lsp/client.zig -fno-emit-bin`; `zig test src/services/shared/utils/zig_toolchain.zig -fno-emit-bin`; `zig test src/core/config/lsp.zig -fno-emit-bin`; `zig test tools/scripts/check_zig_version_consistency.zig -fno-emit-bin`; `zig test tools/scripts/toolchain_doctor.zig -fno-emit-bin`; `zig test tools/scripts/check_cli_dsl_consistency.zig -fno-emit-bin`; `bash -n tools/scripts/*.sh`.
- [x] Environment blockers (2026-03-10 15:42 EDT): `zig build refresh-cli-registry` and `zig build check-docs` both fail on this macOS host with the known Darwin build-runner undefined-symbol linker error before `build.zig` executes.
- [x] Standalone probe note (2026-03-10 15:42 EDT): isolated `zig test` on `tools/cli/generated/cli_registry_snapshot.zig`, `tools/cli/commands/mod.zig`, and `tools/cli/main.zig` is not meaningful because those files depend on build-graph-provided module wiring.

## Active — Docs Surface Cleanup (2026-03-10 16:10 EDT)

- [x] Consensus status: best-effort tri-CLI wrapper unavailable; `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` is missing locally.
- [x] Reviewed canonical workflow inputs before edits: `AGENTS.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `tasks/lessons.md`, and the ABI review references for docs-coupled paths.
- [ ] Rewrite the top-level docs entrypoints (`README.md`, `docs/README.md`) so install, validation, docs generation, and navigation are easier to follow.
- [ ] Tighten agent/developer Markdown (`docs/FAQ-agents.md`, `docs/guides/cursor_rules.md`, `GEMINI.md`, `.claude/*`, `zig0abi-codebase/SKILL.md`) around the current Zig/tooling workflow and generated-docs expectations.
- [ ] Remove stale `wdbx` named-module guidance from `zig-abi-plugin/` Markdown and align it with the current `src/root.zig` + `abi.features.database` surface.
- [ ] Improve supporting Markdown with clearer repository-specific guidance and less stale transition language around `profiles` / `personas`.
- [ ] Run docs-focused validation (`zig build check-docs` if possible, otherwise alternate evidence) and record blockers.

## Active — Main Integration Cleanup (2026-03-10 14:38 EDT)

- [x] Consensus status: best-effort tri-CLI wrapper unavailable; `/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh` is missing locally.
- [x] Parked untracked `tools/synthetic_pipeline/` outside the merge path in stash entry `park synthetic pipeline before main merge`.
- [x] Verified `origin/main` already contains PR #485 as merge commit `28af94db` on 2026-03-10.
- [x] Reviewed the remaining branch tail against `origin/main`: keep `997f3143` and `8c9b3ee4`; drop junk-only commit `6ffa7483` (`.!94407!test_link`, `full-check-current.txt`, `full-check-output.txt`, `zls.json`).
- [x] Clean replay validation on `codex/main-integration-cleanup`: `zig fmt --check` passed for retained Zig diffs, `git diff --check` passed after removing the stray `.gitignore` blank-line regression, and compile-only `zig build-obj -fno-emit-bin` probes passed for `src/core/mod.zig`, `src/features/database/mod.zig`, `src/features/database/stub.zig`, `src/services/mcp/mod.zig`, `src/core/database_fast_tests_root.zig`, and `src/root.zig`.
- [x] Validation evidence remains blocked in this environment: local `zig build` gates still hit the known Darwin linker failure, and GitHub Actions run `22911876542` could not provide Linux gate results because the account is billing-locked (`Test Suite`, `Examples`, and `Quality Gates` skipped).
- [x] Replayed the clean branch-tail commits onto `main` and pushed `main` to `origin` at commit `3fbb03a5`.
- [x] Deleted `fix/codebase-quality-sweep` locally and on `origin` after `main` contained the cleaned changes.
- [x] Deleted the live merged remote branch `origin/claude/init-project-setup-TcKbR` and pruned stale tracking refs for `origin/codex/agent-a761c502-reviewable` and `origin/feat/agnts-consolidation`; only `origin/main` remains.
- [x] Preserved the parked `tools/synthetic_pipeline/` work outside `main` in local stash entry `park synthetic pipeline before main merge`; no extra local branches remain.
- [x] Post-push workflow status on `main`: GitHub Actions run `22918697812` fired for commit `3fbb03a5` but still failed before running `Test Suite`, `Quality Gates`, or `Examples`, while `pages build and deployment` run `22918697285` succeeded.

## Next Phase — Release & Scale

- [ ] **CI Restoration**: Push to main and verify GitHub Actions pass on Linux
- [ ] **WASM Optimization**: Refine freestanding distance functions for browser-side inference
- [ ] **API Expansion**: Implement missing OpenAI-compatible streaming endpoints
- [ ] **Darwin Validation**: Keep compile-only and Linux CI guidance current for linker-blocked hosts
- [ ] **Plugin Registry**: Push `zig-abi-plugin` to the official Claude Code registry

## Backlog

1. [ ] Finalize automated doc generation for cross-language bindings
2. [ ] Audit `tools/cli/commands/` for Windows compatibility
3. [ ] Implement distributed WAL for WDBX clusters
4. [ ] MCP server hardening (WDBX + ZLS integration)
5. [ ] Comprehensive test suite run on Linux CI to verify all waves

---

## Archive

### Codebase Quality Sweep (PR #485) — COMPLETE

All 5 waves committed on branch `fix/codebase-quality-sweep`:

- [x] Wave 1: Fix 33 corrupted files, create database/stub.zig, repair migration artifacts
- [x] Wave 2: Repair build system — broken test manifest, stale wdbx test root, dead persona tests
- [x] Wave 3: Repair coordination module — broken personas/ and database/ imports
- [x] Wave 4: Deep corruption sweep — 118+ additional truncated string literals across 66 files
- [x] Wave 5: AI integration bridges, mod/stub parity, doc updates, validation matrix fixes
- [x] Commit `68dcf34c` — 1081 files changed, +5157/-10927 lines
- [x] PR #485 updated with full change list
- [x] PR #485 merged to `origin/main` as merge commit `28af94db` on 2026-03-10.

#### AI Integration (Wave 5)
- [x] `feedback/learning_bridge.zig` — FeedbackSystem → SelfLearningSystem closed loop
- [x] `agents/agent.zig` — AdvancedCognition + BackendMetrics
- [x] `multi_agent/runner.zig` — blackboard → experience buffer
- [x] `ralph/skills_store.zig` — skill quality tracking (execution_count, success_count, avg_quality)
- [x] `database/mod.zig` — expanded to 91-line API (parity with stub.zig)
- [x] `gpu/stub.zig` — added 3 missing sub-module stubs

### Post-Sweep Cleanup — COMPLETE

- [x] Database boundary refactor (single change set): removed the public `wdbx` package surface, standardized on `abi.features.database`, rewired build/test roots away from the named `wdbx` module, and migrated in-tree callers/docs/parity checks together.
- [x] `zig fmt` applied to touched Zig files; compile-only checks passed via `zig build-obj -fno-emit-bin` for `src/root.zig`, `src/features/database/mod.zig`, `src/features/database/stub.zig`, `src/generated_feature_tests.zig`, `src/services/mcp/mod.zig`, and `src/core/database_fast_tests_root.zig`.
- [x] Residual risk: `zig build validate-flags` and `zig build feature-tests --summary all` both failed in this environment with the known Darwin linker undefined-symbol issue; `full-check` / `verify-all` remain pending on a host with a working Zig link path.
- [x] Fix stale README.md references (`abi.personas` → `abi.features.ai.profiles`)
- [x] Fix stale docs/api/v1.md references
- [x] Fix coordination mod/stub — inline CoordinationContext and InteractionCoordinator (removed dependency on deleted MultiPersonaSystem)
- [x] Consensus status: best-effort tri-CLI wrapper unavailable; recorded as proceeding without wrapper output.
