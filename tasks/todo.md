# Task Tracker

This file is the canonical execution tracker for active ABI work.
Active and in-progress plans stay at the top. Completed plans move into the
archive section with their evidence preserved.

**How to use:** Work from the Active Queue top-down. Mark items `[x]` when done;
move completed plans to Archive with evidence. Backlog items can be promoted to
a new plan when starting a wave.

---

## Active Queue

### In Progress - Wave 1 Toolchain and Validation Baseline (2026-03-10)

#### Objective
Make ABI validation trustworthy again before broader codebase work by
canonicalizing the Zig pin, hardening the Darwin/CEL bootstrap path, and
expanding default CI coverage.

#### Plan
- [x] Canonicalize toolchain-facing scripts and docs around `.zigversion` / `0.16.0-dev.1503+738d2be9d`.
- [x] Update `.cel/build.sh`, `cel_migrate.sh`, `use_cel.sh`, `cel_doctor.zig`, and `build/cel.zig` to classify stock Zig mismatch, Darwin build-runner failure, bootstrap-host readiness, and `.cel` readiness.
- [x] Extract shared shell helpers into `.cel/lib.sh` (DRY refactor across build.sh, use_cel.sh, cel_migrate.sh).
- [x] Add ZLS commit-pin support (`ZLS_UPSTREAM_COMMIT` in config.sh, pinned fetch in build.sh).
- [x] Fix `use_cel.sh` sourcing bug (`set -euo pipefail` breaking caller shells).
- [x] Deduplicate `build/cel.zig` build steps via `addCelShellStep` helper.
- [x] Extend default CI coverage with `zig build check-cli-registry` and `zig build check-docs`.
- [ ] Restore GitHub Actions billing and rerun the existing failed `main` CI workflow.
- [ ] Run `zig build verify-all` and `zig build benchmarks` on a working Linux/CEL host after CI is green.

#### Notes
- The AGENTS-required tri-CLI consensus wrapper is absent locally (`/Users/donaldfilimon/.codex/skills/multi-cli-communication-expert/scripts/run_tricli_consensus.sh`); proceeding best-effort and recording the blocker explicitly.
- Local Darwin `zig build` remains blocked at build-runner link time (`__availability_version_check`, `_arc4random_buf`, etc.) even when `zig` is available on PATH.
- External GitHub Actions execution is still blocked by repository billing; the rerun attempt for run `22876292804` failed before any code-level job started.

#### Review Notes
- `bash -n` passed for `.cel/build.sh`, `tools/scripts/cel_migrate.sh`, and `tools/scripts/use_cel.sh`.
- `zig test -fno-emit-bin` passed for `build/cel.zig` and `tools/scripts/cel_doctor.zig`.
- `./.cel/build.sh --status` now starts cleanly, reports the stock Zig version mismatch, classifies the Darwin linker failure, and recommends `abi toolchain bootstrap` as the next action on this host.
- `./tools/scripts/cel_migrate.sh --check` now reports the stock Zig mismatch, flags the blocked build runner, and converges on the same bootstrap next step.
- `./tools/scripts/use_cel.sh` fails fast with the repo pin, stock-Zig mismatch, and a deterministic next action instead of a generic missing-toolchain error.

---

## Next steps (actionable)

*Wave 1 is repo-side complete but externally blocked. Hosted CI and post-baseline gates still require billing recovery plus a working validation host.*

1. [ ] **Restore Actions billing**: Unblock hosted CI for repository `donaldfilimon/abi`.
2. [ ] **Rerun current `main` CI**: Re-run workflow `22876292804` after billing recovery and confirm `Format Check`, `Test Suite`, `Quality Gates`, and `Examples` all start and pass.
3. [ ] **Hosted validation wave**: Confirm `check-cli-registry` and `check-docs` now pass in default CI.
4. [ ] **Post-baseline gates**: Run `zig build verify-all` and `zig build benchmarks` on a working Linux or CEL-capable host.
5. [ ] **Wave 2 planning**: Open the next improvement wave only after validation is green.

### Completed - Code Quality Improvements (2026-03-09)
- [x] Fix CLAUDE.md/README.md/SKILL.md feature count: 19→27 modules (across 19 directories)
- [x] Fix CLAUDE.md flag combo count: 34→38
- [x] Stub audit: confirmed no sub-module stubs needed (parent gating covers all)
- [x] Test manifest expansion: +12 mod.zig entries (compute, documents, desktop, AI sub-modules, database sub-module)
- [x] Test manifest expansion: +20 dedicated test files (database, network, AI training, personas, observability)

---

## Backlog (expanded tasks)

*Promote to Active Queue when starting a new plan.*

### WDBX / Distributed
- [ ] **MCP Server hardening**: Validate combined WDBX+ZLS MCP server end-to-end; add integration tests for `services/mcp/mod.zig`.

### TUI / CLI
- [ ] **CLI registry**: Run `zig build refresh-cli-registry` and `zig build check-cli-registry` after any command changes; keep docs in sync.

### CEL Toolchain Hardening
- [ ] **Pin ZLS commit**: Determine compatible ZLS commit for Zig 0.16.0-dev.1503 and set `ZLS_UPSTREAM_COMMIT` in `.cel/config.sh`.
- [ ] **CI shell lint**: Add `bash -n` checks for `.cel/*.sh` and `tools/scripts/*.sh` in CI.
- [ ] **Build trap handler**: Add SIGINT/SIGTERM trap in `.cel/build.sh` to clean partial builds.
- [ ] **cel-doctor ZLS awareness**: Report ZLS pin status in `tools/scripts/cel_doctor.zig` diagnostics.

### Build / Toolchain
- [ ] **Run full-check on working host**: Execute `zig build full-check` and `zig build verify-all` on Linux/CI or once Darwin linker is fixed.
- [ ] **Baseline update**: After test/bench changes, run `zig build update-baseline` if the project uses baseline comparison.
- [ ] **Feature-flag validation**: Run `zig build validate-flags` when toolchain allows; fix any mod/stub drift.

### Docs / Consistency
- [x] **CLAUDE.md**: Add a one-line note on WDBX dist (heartbeat + RPC codec) and where to find Coordinator/RPC usage. *(Key Modules: dist, dist.rpc, dist.replication.)*
- [ ] **lessons.md**: After any production bug or correction, append a short lesson and prevention rule.
- [ ] **check-docs**: Run `zig build check-docs` when build succeeds; fix broken or stale references.

## Archive

### Completed - ABI Codex Skill Bootstrap (2026-03-09)

#### Objective
Create a reusable Codex skill named `abi` that captures the project's canonical workflow, validation gates, and handoff expectations in a concise, trigger-ready format.

#### Evidence
- Added `.codex/skills/abi/SKILL.md` with concise metadata and ABI workflow contract guidance.
- Consensus wrapper script path is unavailable in this environment; task continued under best-effort rule from AGENTS contract.
- Validation evidence: file-presence and focused diff checks passed.


### Completed - Codebase Formatting and AST Validation (2026-03-09)

#### Objective
Validate the codebase using formatting and static AST checks (`zig ast-check`) without invoking the broken Darwin linker.

#### Evidence
- Ran `zig fmt .` across the repository; validated compliance.
- Ran `zig ast-check` on all `.zig` files and fixed discovered syntax errors:
  - `tools/cli/terminal/brain_panel.zig`: Fixed duplicate `renderPanel` struct member name (renamed internal renderer to `renderBox`).
  - `tools/cli/commands/dev/toolchain.zig`: Fixed pointless discard of `allocator`.
  - `benchmarks/system/framework.zig`: Fixed unreachable code error by moving discard before return.
- Zero remaining syntax/AST errors in the codebase.

#### Residual Risk
- The logic is currently validated by syntax and structural integrity; full behavioral and compilation validation requires a working linker.

### Completed - Core Engine Optimization and Feature Hardening (2026-03-06 → 2026-03-09)

#### Evidence
- All checklist items completed except benchmarks and full-check (blocked by Darwin linker).
- CEL toolchain integration landed with ZLS support. F32 training pipeline hardened.
- Version pin wave: `0.16.0-dev.1503+738d2be9d` aligned across canonical repo files.
- LSP client implementation with CEL-first resolution added (`src/services/lsp/client.zig`).
- Compile-only tests pass. Format clean.

#### Residual Risk
- Benchmarks and `full-check`/`verify-all` not yet run (Darwin linker blocked). Deferred to CI or CEL host.

### Completed - Do all (this host) 2026-03-08

- **Test manifest**: Confirmed `build/test_discovery.zig` includes `wdbx/dist/mod.zig`, `rpc.zig`, `replication.zig`.
- **Format**: `zig fmt --check build.zig build/ src/ tools/` — pass.
- **Compile-only tests**: `zig test … -fno-emit-bin` pass for `wdbx/dist/mod.zig`, `rpc.zig`, `replication.zig`, `graph/mod.zig`, `wdbx/core/alloc.zig`, `features/network/protocol.zig`.
- **Build**: `zig build cel-status` fails at link (`__availability_version_check` etc.) as documented.


### Completed - CEL Toolchain Finish & Migration Integration (2026-03-08)

#### Objective
Aggressively finish the .cel (Custom Environment Linker) toolchain infrastructure and migrate the Zig build system to be CEL-aware, making CEL the primary path for macOS 26+ Darwin hosts.

#### Evidence
- **`build/cel.zig`**: New build module with `detectCelStatus()`, `addCelCheckStep()`, `addCelBuildStep()`, `addCelStatusStep()`, `addCelVerifyStep()`, `emitCelSuggestion()`. Compiles clean.
- **`tools/scripts/cel_doctor.zig`**: Comprehensive diagnostics — platform detection, directory structure, binary check, patch inventory, version consistency, stock zig status, build prerequisites, actionable remediation. Compiles clean.
- **`tools/scripts/cel_migrate.sh`**: Guided migration script with `--check`, `--activate`, `--clean` modes. Syntax valid.
- **`build.zig` integration**: CEL module imported; `cel-check`, `cel-build`, `cel-status`, `cel-verify`, `cel-doctor` build steps registered; blocked Darwin feature-disable now uses `cel.emitCelSuggestion()` for context-aware guidance.
- **`toolchain_doctor.zig`**: Updated with CEL binary detection, version matching, and CEL-first remediation on blocked Darwin.
- **`check_zig_version_consistency.zig`**: Added `.cel/config.sh` ZIG_VERSION consistency check.
- **`.cel/config.sh`**: Enhanced with migration metadata, build configuration, platform requirements, exports.
- **`.cel/README.md`**: Comprehensive docs with build system integration, patch table, version consistency contract, module documentation.
- **`.cel/patches/003-macho-segment-ordering.patch`**: Placeholder for Mach-O segment fix (upstream #25521).
- **`tools/scripts/use_cel.sh`**: Enhanced with better error messages and CEL migration guidance.
- **`CLAUDE.md`**: Updated to document CEL as primary path with all build steps.
- **Format**: `zig fmt --check build.zig build/ src/ tools/` — pass.
- **Compile-only**: All 4 modified/new Zig files pass `zig test -fno-emit-bin`.

#### Residual Risk
- Build runner linking remains blocked on macOS 26+ until CEL toolchain is built from source.
- `003-macho-segment-ordering.patch` is placeholder; needs concrete upstream fix.

### Completed - Do all (this host) 2026-03-06

- **Test manifest**: Confirmed `build/test_discovery.zig` includes `wdbx/dist/mod.zig`, `rpc.zig`, `replication.zig`.
- **Format**: `zig fmt --check build.zig build/ src/ tools/` — pass.
- **Compile-only tests**: `zig test … -fno-emit-bin` pass for `wdbx/dist/mod.zig`, `rpc.zig`, `replication.zig`, `graph/mod.zig`, `wdbx/core/alloc.zig`, `features/network/protocol.zig`.
- **Build**: `zig build test` fails at link (undefined symbol `__availability_version_check` etc.) as documented; run full-check/verify-all on a host where the toolchain links.
- **Stress tests**: Added `Coordinator: many nodes and shards (stress)` (20 nodes, 50 shards, unassign) and `GraphStore: many edges (stress)` (64-node chain, bfs, remove middle edge).

### Completed - Codebase Review (Plan Execution)

#### Objective
Execute the Codebase Review Plan: architecture/conventions, mod/stub parity, test manifest, full gate, docs/registry, and deliverables.

#### Evidence
- **Prep**: Read `tasks/lessons.md`. `zig build check-imports` and `zig build validate-flags` fail on Darwin due to known linker issue (documented in CLAUDE.md); import rules verified via grep (no `@import("abi")` in feature code, only in comments).
- **Mod/stub audit**: Network mod.zig and stub.zig (including heartbeat/rpc_protocol) have matching exports. Database distributed stub was missing cluster types; added `ClusterManager`, `ClusterConfig`, `ClusterStatus`, `NodeRole`, `NodeState`, `TransportType`, `ClusterMessage`, `MessageType`, `PeerAddress`, `ClusterError` to `src/features/database/stubs/misc.zig` distributed struct.
- **Test manifest**: `build/test_discovery.zig` includes network/heartbeat.zig and rpc_protocol.zig. TUI panels (e.g. memory_panel) are covered by tui-tests; no change to feature_test_manifest for tools/ paths.
- **Full gate**: Not run on this host (Darwin linker blocks binary build). Use `zig build full-check` and `zig build verify-all` when toolchain is available.
- **Docs/registry**: CLI registry updated for new `create-subagent` command; run `zig build check-docs` and `zig build check-cli-registry` when build succeeds.
- **Deliverables**: Zig syntax reviewer subagent (`.cursor/agents/zig-syntax-reviewer.md`), `abi create-subagent` CLI command, Create Subagent TUI panel (F10), and ZVM helper script (`tools/scripts/use_zvm_master.sh`) for Darwin.

#### Residual Risk
- Binary-emitting steps (install, full-check, feature-tests) remain blocked on this arch until upstream Zig or SDK fix.

### Completed - Zig 0.16 Refactor and Organize

#### Objective
Refactor and organize the codebase for Zig 0.16: formatting, build API compliance, and structure.

#### Evidence
- **Formatting**: Ran `zig fmt` on `src/`, `build/`, `tools/`. Fixed `BrainDashboardPanel` in `tools/cli/terminal/brain_panel.zig` (moved `_internal_data` field with other struct fields so declarations are not between container fields). All `zig fmt --check` now passes.
- **Build API**: Audited build.zig and build/*.zig; all use Zig 0.16 pattern: `createModule(.{ .root_source_file = b.path(...) })` and `addTest`/`addExecutable` with `.root_module`. No deprecated `.path` on LazyPath. Added build-root doc comment noting 0.16 API.
- **Organization**: Documented WDBX module layout in `src/wdbx/wdbx.zig` doc comment.

### Completed - TUI and CLI Improvements (2026-03-06)

#### Objective
Modernize and stabilize the ABI CLI and TUI architecture for Zig 0.16, focusing on modular component extraction, UX consistency, and robust integration testing.

#### Evidence
- Migrated `GpuMonitor` and `BrainDashboardPanel` to directly implement the `Panel` vtable interface.
- Removed deprecated adapter layers in `tools/cli/terminal/panels/` to simplify the TUI hierarchy.
- Refactored `AsyncLoop` to use `std.posix.poll` for true non-blocking, event-driven terminal input handling.
- Added the `abi doctor` command for system diagnostics, fully registered in the generated registry.
- Established the `.integration-tests/` hierarchical artifact and golden file testing pattern.

#### Residual Risk
- Full end-to-end visual verification is pending the completion of the background Zig bootstrap.

### Completed - WDBX and Abbey Architecture (2026-03-06)

#### Objective
Implement the WDBX semantic memory fabric and Abbey cognition layers in Zig 0.16.

#### Evidence
- Fully implemented the `StoredBlock` binary codec, SHA-256 checksumming, and compression strategies.
- Created a functional in-memory `BlockStore` with payload lifecycle management.
- Replaced all raw `unreachable` stubs with functional logic or descriptive architectural layouts for Distributed Coordination, Graph Relationships, Memory Management, and Tracing.
- Implemented `TraceLog` with `generateLineageGraph` (dot format) and `exportAuditLog` capabilities.
- Added `TrackingAllocator` for memory telemetry.
- Updated `src/wdbx/wdbx.zig` to unify and export all internal subsystems.

#### Residual Risk
- The logic is currently validated by syntax and structural integrity; full behavioral validation requires the bootstrapped native toolchain.

### Completed - Darwin Toolchain Unblock And Branch Stabilization (2026-03-06)

#### Objective
Unblock local Darwin Zig build execution under
`[$zig-master](/Users/donaldfilimon/.codex/skills/zig-master/SKILL.md)` without
starting the next roadmap wave, by stabilizing the current branch state,
repairing or isolating the Apple/Xcode toolchain path, and tightening repo-side
diagnostics so future failures are classified quickly.

#### Evidence
- `tools/scripts/toolchain_doctor.zig` updated to point to `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer` as the local known-good.
- Confirmed the Darwin linker failure is a systemic issue external to the repository codebase. Even simple C/Zig programs linking against `libc` using the current Zig 0.16-dev master branch fail with `undefined symbol: __availability_version_check`, `_abort`, `_arc4random_buf`, etc.
- Exhaustive target clamping (`native-macos.14`) and explicit `SDKROOT` overrides (`MacOSX15.4.sdk`, `MacOSX26.4.sdk`, etc.) using Xcode-beta fail to resolve the issue. This isolated the blocker to an incompatibility between the latest Zig 0.16 master branch linker and the specific futuristic macOS/Xcode-beta environment present on this machine (which reports `26.4.0` native version).

#### Residual Risk
- Linker failures will continue to block any `zig build` target that outputs a binary until the upstream Zig linker resolves the `libSystem` SDK compatibility issue or the emergency bootstrap completes.

### Completed - Canonical Command Registry And Runtime Consolidation (2026-03-06)

#### Objective
Land the first cohesive slice of the approved Zig 0.16 UX-first consolidation
roadmap by making the command registry authoritative across CLI/docs/smoke
coverage, removing duplicate editor runtime logic, tightening `abi ui`
shell/view behavior, and adding a focused fast WDBX validation seam that can
run independently of the blocked full Darwin close-out.

#### Evidence
- The command-registry/runtime slice is now wired so: `tools/cli/commands/dev/editor.zig` remains a thin shared-engine wrapper, `tools/cli/commands/core/ui/mod.zig` accepts `dashboard` as the canonical shared-shell alias.
- `tools/gendocs/source_cli.zig` now consumes the canonical CLI registry via an injected `cli_root` module import.
- Focused WDBX validation is now rooted at `src/wdbx_fast_tests_root.zig`.

### Completed - Fix Review Regressions And Harden AI CLI Backends (2026-03-06)

#### Objective
Land the requested fix wave for the reported regressions in AI config/root exports,
database compatibility, WDBX token datasets, and C bindings, while also tightening
`os-agent`/backend CLI behavior so the new backend routing changes remain
compatible and explicit.

#### Evidence
- Repaired AI config/reasoning integration and restored valid feature gating.
- Fixed WDBX token dataset persistence semantics and C API dimension handling.
- Tightened `os-agent`/backend parsing/help behavior while preserving current aliases.

### Completed - Docs + Assistant Canonical Sync Around `zig-master` (2026-03-06)

#### Objective
Align the repo workflow contract, Zig validation policy, assistant-facing docs,
todo/status markdown, and generated docs around one canonical model.

#### Evidence
- `rg -n '\\.claude/rules/zig\\.md'` returns no matches in the repository.
- Docs generation setup updated to respect the new canonical hierarchy.

### Completed - ABI Zig 0.16 Breaking Cleanup (2026-03-06)

#### Objective
Execute the approved breaking cleanup wave for the ABI Zig 0.16 codebase.

#### Evidence
- Refactored `full-check` and `verify-all` so they compose only from leaf steps.
- Removed legacy build flag aliases, compatibility namespaces, and fallback paths.
- Simplified baseline and consistency checks.

### Completed - Canonicalize WDBX + Persona Architecture (2026-03-06)
(Archived entries continue below...)
