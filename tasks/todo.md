# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Objective

Advance the ABI Zig 0.16 framework toward production maturity: complete the Phase 15 roadmap items (semantic store persistence, mobile native bridges, bare metal examples), resolve remaining Darwin 25+ toolchain blockers, and maintain full verification gate compliance.

## Scope

**In scope:**
- Phase 15 roadmap items (semantic store, mobile bridges, embedded examples)
- Darwin host-built Zig bootstrap completion
- Verification gate maintenance and docs synchronization
- Integration gate validation (Stage 2, COMPLETE via run_build.sh — CI remains authoritative)

**Out of scope:**
- Phases 1–14 (completed, archived below)
- Zig version repinning beyond `0.16.0-dev.2962+08416b44f`
- New feature modules not on the Phase 15 roadmap

## Verification Criteria

- `zig build full-check --summary all` passes (or `./tools/scripts/run_build.sh full-check --summary all` on Darwin 25+)
- `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` clean
- All 56 feature flag combinations validated
- Test baseline consistent with `tools/scripts/baseline.zig`
- Documentation generator deterministic (`check-docs` passes)

## Checklist

### Phase 15 — Active Roadmap

- [x] **Plugin Ecosystem**: Formalize the plugin API for external C/WASM modules to register with `abi.registry`.
- [x] **Distributed Raft V2**: Harden the `abi.network` Raft implementation for production-grade partition tolerance.
- [x] **Semantic Store Persistence**: Optimize the WDBX block-chain layout for multi-terabyte vector indices. Implemented real disk I/O for SegmentLog, WAL flush/recover, RLE compression, compaction, and HNSW graph persistence (commit `3c10080e`).
- [x] **Mobile Native Bridges**: Implement native Swift/Kotlin bridges for iOS and Android high-level UI integration. Added 9 `abi_mobile_*` C exports, Swift Package (`lang/swift/`), Kotlin/JNI bridge (`lang/kotlin/`) (commit `ea5e75e2`).
- [x] **Bare Metal Examples**: Add `examples/embedded/` showing deployment to RISC-V 32 and Thumb bare-metal boards. Files already exist: `bare_metal_riscv32.zig`, `bare_metal_thumb.zig`, `mobile.zig`
- [x] **Import-Rule Guardrail Hardening**: Restored bare `build_options` named imports, normalized AI imports, strengthened `check-imports`.
- [x] **Import Boundary and Skill Repair Wave**: Repaired Zig 0.16 import cleanup, aligned skills with module boundaries.
- [x] **Pinned macOS Host-Zig Bootstrap Wave**: Added deterministic host-built Zig bootstrap/discovery path for Darwin.
  - [ ] Runtime validation pending: `bootstrap_host_zig.sh` builds zig1/zig2 but stage3 self-build fails on Darwin 26.4 (Apple LLD incompatibility with Zig's self-hosted build). Workaround: `run_build.sh` successfully relinks build runners for full gate coverage.

### Stage Status

- **Stage 1** (push + PR triage): COMPLETE — 2 commits pushed, 8 PRs (488-495) closed as superseded
- **Stage 2** (integration gates): COMPLETE via `run_build.sh full-check` on Darwin 26.4 (all 56 flag combos validated; CI remains authoritative)
- **Stage 3** (close plans): COMPLETE — 5 plans marked complete/superseded, Integration Gates v1 unblocked
- **Stage 4** (Phase 15 roadmap): COMPLETE — all 8 items done (semantic store, mobile bridges, bare metal, plugin, raft, imports, bootstrap)
- **Stage 5** (housekeeping): COMPLETE — worktrees cleaned, PRs closed, lessons updated

## Review

### Current State (2026-03-18)

**Phase 15 COMPLETE + Post-Phase improvements** (2026-03-18). Delivered in 17+ commits:

Phase 15 deliverables (`e19ee037..ea5e75e2`):
- Semantic store persistence: real disk I/O for SegmentLog, WAL, RLE compression, compaction, HNSW graph block
- Mobile native bridges: 9 `abi_mobile_*` C exports, Swift Package, Kotlin/JNI bridge (1,087 new lines)
- Integration gates unblocked: matrix manifest export, timeout enforcement, enhanced preflight
- 5 execution plans closed, PRs #512-515 created and merged
- 20/20 feature parity, efficiency fixes (JWT, rate limiter, inference), security fixes (shell injection)

Post-Phase 15 improvements (PRs #516-521):
- [x] Persistence benchmarks: SegmentLog, WAL, compression throughput suites (#519)
- [x] Mobile binding validation: 17 tests covering lifecycle, sensors, permissions, notifications (#516)
- [x] DiskANN mmap persistence: sector-aligned save + zero-copy mmap load (#517)
- [x] BM25 search index persistence: save/load with SRCH binary format (#520)
- [x] Real LZ4 block compression: replaced RLE placeholder with standard LZ4 block format (#518)
- [x] Inference safety: KV cache panic fix, memory leak fix, secure wipe (#521)
- [x] API docs: 30 doc comments on root.zig public exports
- [x] Use-after-free fix in block_chain traverseBackward (defer→errdefer)

Gate: 227/227. All 56 flag combinations validated. Darwin stage3 pending upstream Zig fix.

### Codebase-Wide Review + Advancement (2026-03-19)

Review pass (14 agents, 17 files):
- [x] Fixed 4 ungated cross-feature imports in AI sub-modules (metrics, coordination, training, memory)
- [x] Added missing `feat_lsp`/`feat_mcp` to CLI build_options_stub.zig
- [x] C bindings: `abi_is_feature_enabled` coverage 6→27 features, `abi_enabled_feature_count` 8→17 fields
- [x] Documentation: lang/ Swift/Kotlin, example count 35→36, feat_mobile default, AGENTS.md dedup
- [x] Plugin: fixed hooks wdbx advice, ghost build steps, step counts, test manifest target

Advancement pass:
- [x] Added `types.zig` to 8 feature directories (compute, database, desktop, documents, gpu, network, observability, web) — now 19/19 complete
- [x] Removed legacy `src/abi.zig` tombstone and all doc references
- [x] Wired 5 orphaned GPU test files into test discovery manifest
- [x] Plugin: CEL skill marked aspirational, new-feature command Step 10 added, import hook broadened to all src/
- [x] LSP/MCP: confirmed internally gated via module switcher pattern — no root.zig changes needed

Wave 5A + docs sync pass:
- [x] Simplified redundant AI dual-gating in llm, training, explore isEnabled()
- [x] Fixed auth verifyToken page_allocator leak (added allocator parameter)
- [x] Consolidated auth module's 15 repeated @import calls
- [x] Fixed network stub types namespace divergence (shared_types → types)
- [x] Added observability/mod.zig to feature test manifest
- [x] Removed dead AI sub-module framework lifecycle code (initAiSubModules, 4 facade files, 2 routing files — 974 lines deleted)
- [x] Gated cross-feature imports in brain_export, export, abbey_train, retriever
- [x] Corrected test manifest flags for eval/rag/constitution (feat_ai → feat_reasoning)
- [x] Fixed training stub parity: LoraModel init arity, LoraConfig.TargetModules, 6 missing methods
- [x] Fixed stale docs: README phantom abi.zig, example count 35→36, feature count, feat-mobile exception

### Completed (Archived)

Historical phases of the master-branch structure redesign and normalization (completed March 2026):

- **Phases 1-3**: Logical graph normalization, physical relayout, and feature module consolidation (AI contexts, shared primitives).
- **Phases 4-11**: Post-restructure cleanup, governance drift sweeps, deep contract hardening, and AI import hygiene.
- **Phases 12 (Core), 13 & 15**: Gemini Wave logic consolidation, Darwin/macOS stability refinements, and documentation gate recovery.
- **Phase 14**: Final cleanup and universal support — inference promotion, compat bridge removal, terminology alignment, AI consolidation, docs refresh, Darwin guidance, CI integration, freestanding audit, mobile refinement, runtime SIMD probing.
- **Key Outcomes**: Single `abi` module, canonical `abi.<domain>` API, 56 validated flag combinations, and full 165-step validation success.
