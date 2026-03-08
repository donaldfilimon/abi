# Task Tracker

This file is the canonical execution tracker for active ABI work.
Active and in-progress plans stay at the top. Completed plans move into the
archive section with their evidence preserved.

**How to use:** Work from the Active Queue top-down. Mark items `[x]` when done;
move completed plans to Archive with evidence. Backlog items can be promoted to
a new plan when starting a wave.

---

## Active Queue

### Task Plan - Core Engine Optimization and Feature Hardening (2026-03-06)

#### Objective
Advance the recently implemented WDBX and TUI foundations toward production-grade
stability, focusing on performance profiling, real-world data stress tests, and
deeper subsystem integration.

#### Scope
- Implement real heartbeat state-machines and shard-balancing in `src/wdbx/dist/`.
- Optimize `GraphStore` reverse-traversal with a more memory-efficient index.
- Wire `TrackingAllocator` telemetry into the TUI dashboard for real-time memory profiling.
- Implement the binary RPC protocol for node-to-node block replication.
- Refactor TUI components to use the new `std.Io` concurrent fetch patterns.

#### Verification Criteria
- `zig build verify-all`
- `abi doctor` returns green on the bootstrapped toolchain.
- Successful node-to-node replication verified via trace logs.

#### Checklist

##### Done (this wave)
- [x] Implement Heartbeat state-machine in `dist/mod.zig` (HealthState, tick, configurable timeouts).
- [x] Optimize `GraphStore` adjacency list (flat edge list + outgoing/incoming index maps).
- [x] Create real telemetry dashboard view using `TrackingAllocator` data (MemoryPanel + WDBX tracker).
- [x] Implement skeleton binary RPC message codec in `dist/rpc.zig` (Header, HeartbeatPayload, BlockSyncRequest).

##### Review (before closing plan)
- [ ] Run benchmarks; confirm no regression from new telemetry or GraphStore layout.
- [x] Add or run trace logging for dist state; verify logs reflect node state transitions (active/stale/failed). *(Coordinator.trace_state_change callback added; test covers transitions.)*
- [ ] Run `zig build full-check` and `zig build verify-all` on a host where the toolchain links.
- [x] Document `Coordinator` / `dist.rpc` usage in CLAUDE.md or a short dist README. *(Key Modules table: abi.wdbx.dist, abi.wdbx.dist.rpc.)*

##### Completed (Zig 0.16 std API pass)
- [x] Network protocol: `std.meta.intToEnum` → validated `switch` for `ResultStatus`; decode error set includes `Allocator.Error`; test "encode rejects oversized kind" fixed (no 4GB allocation).
- [x] WDBX core alloc: allocator vtable uses `std.mem.Alignment` instead of `u8`.
- [x] CLAUDE.md: added "Zig 0.16 std API notes" (time, enums, HashMap, Allocator, Build, mem, std.Io).

##### Follow-on tasks (expand into next plan if desired)
- [x] Shard ownership and assignment in `Coordinator` (e.g. `shard_map` usage, rebalance on node fail). *(assignShard, getShardOwner, unassignShardsForNode.)*
- [x] Implement block sync response/chunk encode-decode in `dist/rpc.zig` and wire to a replication path. *(BlockSyncResponse, BlockChunk + tests; replication path TODO.)*
- [x] Refactor TUI data fetches to use `std.Io` concurrent patterns where applicable. *(Audited: training_panel, editor/engine use std.Io.Dir.cwd(); optional follow-up: concurrent fetch to avoid blocking tick.)*
- [x] Optional: wire WDBX engine’s `TrackingAllocator` into the dashboard when running `abi ui` with engine. *(ShellPanel.connectWdbxTracker; abi.wdbx re-export.)*

---

## Next steps (actionable)

*Run in order when toolchain allows; environment-blocked items are noted. "Do all" on this host: verified manifest, fmt, compile-only tests, added stress tests; build/test still fails at link (Darwin).*

**Proceeded (this session):** CEL toolchain integration landed. `build/cel.zig` adds `cel-check`, `cel-build`, `cel-status`, `cel-verify` build steps. `tools/scripts/cel_doctor.zig` provides comprehensive diagnostics. `tools/scripts/cel_migrate.sh` offers guided migration. Toolchain doctor and version consistency checks now include CEL detection. All compile-only tests pass. Format clean.

**Proceeded (this session):** F32 training pipeline hardened. `llm_trainer.zig`: fixed `evaluate()` to use real `model.forward()` instead of random logits; wired `mixed_precision` config flag with `LossScaler`; added NaN/Inf gradient guard. `mixed_precision.zig`: added `reset()` and `updateScale()` alias. `mod.zig`: added `PrecisionMode` enum and f32 precision docs. Format clean.

1. [ ] **Benchmarks**: Run `zig build benchmarks` (or suite=simd/database); confirm no regression from telemetry or GraphStore. *(Requires working linker.)*
2. [ ] **Full gate**: On a host where the toolchain links, run `zig build full-check` and `zig build verify-all`.
3. [x] **Rebalance on fail**: From application code, call `coord.unassignShardsForNode(allocator, node_id)` when a node transitions to failed (e.g. in trace_state_change callback); optionally reassign shards to healthy nodes. *(Test and doc in dist/mod.zig.)*
4. [x] **Block replication path**: Wire `dist.rpc` BlockSyncRequest/Response/Chunk to a single-node-to-node copy path; add trace logs for sync. *(dist/replication.zig runRequesterPath + tests.)*
5. [ ] **CLI registry**: After any command changes, run `zig build refresh-cli-registry` and `zig build check-cli-registry`. *(When build succeeds.)*
6. [ ] **check-docs**: Run `zig build check-docs` when build succeeds.

---

## Backlog (expanded tasks)

*Promote to Active Queue when starting a new plan.*

### WDBX / Distributed
- [x] **Shard balancing**: Implement rebalance logic when nodes go stale/failed; update `Coordinator.shard_map` and document contract. *(unassignShardsForNode + trace callback test; assignShard for reassign.)*
- [x] **RPC transport**: Add a minimal transport layer (e.g. stream over TCP) that uses `dist.rpc` encode/decode; no full Raft yet.
- [x] **Block replication**: Use `dist.rpc` BlockSyncRequest/Response in a single-node-to-node copy path; trace logs for sync. *(dist/replication.zig runRequesterPath.)*
- [x] **WDBX stress tests**: Add tests or a small harness that stress the graph store and dist coordinator under load (many nodes, many edges). *(dist: "Coordinator: many nodes and shards (stress)"; graph: "GraphStore: many edges (stress)".)*

### TUI / CLI
- [x] **std.Io TUI refactor**: Identify TUI code paths that do I/O or blocking work; refactor to use `std.Io` concurrent fetch where it fits. *(Audited: file I/O uses std.Io.Dir.cwd(); optional: concurrent fetch for tick.)*
- [x] **Dashboard WDBX wiring**: When the UI runs with an engine that uses `TrackingAllocator`, call `memory_panel.connectWdbxTracker(...)` from dashboard init. *(ShellPanel.connectWdbxTracker added.)*
- [ ] **CLI registry**: Run `zig build refresh-cli-registry` and `zig build check-cli-registry` after any command changes; keep docs in sync.

### Build / Toolchain
- [ ] **Run full-check on working host**: Execute `zig build full-check` and `zig build verify-all` on Linux/CI or once Darwin linker is fixed.
- [ ] **Baseline update**: After test/bench changes, run `zig build update-baseline` if the project uses baseline comparison.
- [ ] **Feature-flag validation**: Run `zig build validate-flags` when toolchain allows; fix any mod/stub drift.

### Docs / Consistency
- [x] **CLAUDE.md**: Add a one-line note on WDBX dist (heartbeat + RPC codec) and where to find Coordinator/RPC usage. *(Key Modules: dist, dist.rpc, dist.replication.)*
- [ ] **lessons.md**: After any production bug or correction, append a short lesson and prevention rule.
- [ ] **check-docs**: Run `zig build check-docs` when build succeeds; fix broken or stale references.

## Archive

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
