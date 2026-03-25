# Full Codebase Improvement — Design Spec

**Date:** 2026-03-24
**Approach:** Risk-Layered (zero-risk through structural refactoring)
**Baseline:** 3266 tests passed, 4 skipped, 0 failed

## Context

The ABI Zig framework (1,127 .zig files, 20 feature directories, 30 features) is architecturally sound but has accumulated inconsistencies across five areas: git hygiene, protocol gating, AI sub-feature stub parity, monolithic file sizes, and documentation gaps. This spec addresses all of them in risk-ascending order so each phase can be validated before escalating.

## Phase 1 — Zero-Risk (No Behavior Change)

### 1a. Git Hygiene
- `git add src/features/gpu/policy/target_contract.zig` (referenced in build.zig but untracked)
- Remove root-level stray artifacts: `test.db`, `target_contract.o`, `libcontext_init.a`
- Update `.gitignore`: explicit rules for `test.db`, root-level `*.o`, `libcontext_init.a`

### 1b. Doc Comments
- `src/features/gpu/policy/target_contract.zig`: add module-level doc explaining comptime contract validation purpose
- `src/core/database/persistence.zig`: add `//! @internal: Re-exported via parent module`
- `src/tasks/persistence.zig`: same

### 1c. Error Handling Convention
Add to CLAUDE.md under "Key Conventions":
- `@compileError` — compile-time contract violations (policy enforcement only)
- `@panic` — unrecoverable invariant violations (never in library code; CLI/test only)
- `unreachable` — provably impossible branches (comptime-verified)
- Error unions — all runtime failure paths in library code

### 1d. Doc Consistency
- Verify GEMINI.md feature counts match CLAUDE.md (20 dirs, 30 features)
- Update any stale counts found
- **Note:** Phase 2 may add `acp`/`ha` to the feature catalog, changing counts. If so, defer doc count updates to after Phase 2 merges, or use "30+ features" phrasing.

### 1e. Build Step Descriptions
Verify alias steps in build.zig have differentiated descriptions. If already differentiated (e.g., "Run CLI tests"), skip. Only update if descriptions are generic/identical.

**Validation:** `./build.sh lint`, `git status` clean, no test changes needed.

---

## Phase 2 — Low-Risk (Additive Only)

### 2a. Add `feat_acp` and `feat_ha` Feature Gates

**Problem:** ACP and HA protocols have stub.zig files but are unconditionally imported in root.zig, unlike MCP/LSP which are comptime-gated.

**Files to modify:**
- `build.zig` — multiple edit sites:
  - Add `feat-acp` (default true) and `feat-ha` (default true) bool options
  - Add fields to `FeatureFlags` struct so `addAllBuildOptions` emits them
  - Add entries to every `cross_opts` block in the cross-compilation section (otherwise `zig build cross-check` fails)
  - Add display lines to the `doctor` step
- `src/root.zig` (lines 47, 49): Change to comptime gates:
  ```zig
  pub const acp = if (build_options.feat_acp) @import("protocols/acp/mod.zig") else @import("protocols/acp/stub.zig");
  pub const ha = if (build_options.feat_ha) @import("protocols/ha/mod.zig") else @import("protocols/ha/stub.zig");
  ```
- `src/core/feature_catalog.zig`: Add `acp` and `ha` variants if missing. **Note:** this changes the total feature count — update docs accordingly in Phase 1d or after this phase merges.
- `src/feature_parity_tests.zig`: Add parity checks for ACP and HA
- Verify `src/protocols/acp/stub.zig` and `src/protocols/ha/stub.zig` match mod surfaces

**Scope note:** `inference` is also unconditionally imported in root.zig (line 51). This is intentional — inference is a core subsystem always needed by the framework, unlike ACP/HA which are optional protocols. Do not gate inference.

**Validation:** `./build.sh -Dfeat-acp=false -Dfeat-ha=false test --summary all` passes. `./build.sh check-parity` clean. `./build.sh cross-check` passes (verifies cross_opts updated).

---

## Phase 3 — Medium-Risk (Parity Fixes)

### 3a. Fix 10 AI Sub-Feature Stub Parity Mismatches

**Problem:** AI sub-modules have divergent export counts between mod.zig and stub.zig.

All paths below are under `src/features/ai/` (AI sub-modules), NOT the top-level `src/features/database/` or `src/features/documents/`.

| Sub-module (src/features/ai/...) | mod exports | stub exports | Direction |
|----------------------------------|-------------|--------------|-----------|
| ai/agents | 9 | 34 | stub has extras |
| ai/vision | 54 | 37 | mod has extras |
| ai/explore | 53 | 70 | stub has extras |
| ai/llm | 50 | 65 | stub has extras |
| ai/embeddings | 16 | 7 | mod has extras |
| ai/training | 117 | 124 | stub has extras |
| ai/profile | 20 | 14 | mod has extras |
| ai/database | 14 | 10 | mod has extras |
| ai/streaming | 68 | 71 | stub has extras |
| ai/documents | 17 | 15 | mod has extras |

**Note:** Re-verify counts with `zig build check-parity` before starting — counts are approximate from exploration and may have shifted.

**Approach for each:**
1. Run `zig build check-parity` to get exact delta
2. Determine canonical API (mod.zig is authoritative)
3. If stub has extras: remove declarations not in mod.zig (unless they're from types.zig and intentionally broader)
4. If mod has extras: add missing no-op stubs to stub.zig

**Priority order:** By delta magnitude: agents (25), vision (17), explore (17), llm (15), embeddings (9), training (7), profile (6), database (4), streaming (3), documents (2).

**Validation:** `./build.sh check-parity` reports zero mismatches. `./build.sh test --summary all` passes.

---

## Phase 4 — Higher-Risk (Structural Refactoring)

### 4a. Decompose Monolithic Files

Strategy: "thin re-export facade" — original file keeps public API, delegates to new sub-files. No consumer code changes.

| File | Lines | Decomposition |
|------|-------|---------------|
| `core/database/diskann.zig` | 1669 | `diskann/codebook.zig` (PQCodebook), `diskann/graph.zig` (VamanaGraph), `diskann/index.zig` (DiskANNIndex) |
| `core/database/hnsw/mod.zig` | 1423 | `hnsw/search.zig`, `hnsw/insert.zig` (extract algorithms) |
| `features/gpu/ai_ops.zig` | 1355 | `ai_ops/vtable.zig`, `ai_ops/adapters.zig` |
| `features/gpu/backends/metal/mps.zig` | 1267 | Split by operation category |
| `core/database/scann.zig` | 1238 | `scann/codebook.zig`, `scann/partitioning.zig` |
| `features/gpu/dsl/codegen/generic.zig` | 1221 | `generic/arithmetic.zig`, `generic/memory.zig`, `generic/control.zig` |
| `features/gpu/execution_coordinator.zig` | 1174 | `execution/scheduler.zig` extraction |
| `features/gpu/backends/vulkan.zig` | 1173 | `vulkan/pipeline.zig`, `vulkan/descriptors.zig` |
| `features/ai/constitution/enforcement.zig` | 1160 | Per-principle validator files |

### 4b. Reduce Over-Centralized Facades

Introduce intermediate sub-namespace re-exports. **Important:** preserve all existing flat exports as aliases for backwards compatibility. The new sub-namespaces are additive — old paths like `abi.network.SomeType` continue to work alongside `abi.network.http.SomeType`.

| Module | Exports | Strategy |
|--------|---------|----------|
| `network/mod.zig` | 191 | Group into `network.http`, `network.dns`, `network.socket` (keep flat aliases) |
| `abbey/mod.zig` | 140 | Group into `abbey.neural`, `abbey.behavior`, `abbey.memory` (keep flat aliases) |
| `training/mod.zig` | 123 | Group into `training.distributed`, `training.mixed_precision`, `training.core` (keep flat aliases) |
| `gpu/mod.zig` | 119 | Tighten `gpu.backends`, `gpu.memory`, `gpu.kernels` (keep flat aliases) |

### 4c. AI Namespace Documentation

Add conceptual grouping doc comment to `src/features/ai/mod.zig`:
- Inference: llm, embeddings, vision, models, streaming
- Reasoning: abbey, aviva, abi, constitution, eval, reasoning
- Agents: agents, tools, multi_agent, coordination, orchestration
- Learning: training, memory, federated
- Support: templates, prompts, documents, profiles, context_engine

Do NOT restructure directories (too much import churn for the benefit).

**Validation after each sub-phase:** `./build.sh test --summary all` + `./build.sh check-parity` + `./build.sh cross-check`. Run `./build.sh fix` after creating new files to ensure formatting compliance.

---

## Phase 5 — Final Validation

1. Full test suite: `./build.sh test --summary all` — expect 3266+ passed, 4 skipped, 0 failed
2. Parity: `./build.sh check-parity` — zero mismatches
3. Cross-compilation: `./build.sh cross-check` — all 4 targets pass
4. Lint: `./build.sh lint` — clean
5. Full gate: `./build.sh check` — passes
6. Baseline sync: update `.claude/skills/baseline-sync/SKILL.md` with new counts

---

## Implementation Notes

- Each phase gets its own feature branch and commit(s)
- **Phase ordering constraint:** Phase 1d (doc consistency) depends on Phase 2 (feature catalog changes). Either run Phase 2 first, or defer Phase 1d's feature count updates to after Phase 2 merges.
- Phases 1 (excluding 1d counts) and 3 can run in parallel worktrees
- Phase 2 must complete before Phase 1d finalizes feature counts
- Phase 4 must be sequential within a domain, but Phase 4a decompositions across different domains (database vs gpu vs ai) are independent and parallelizable
- All phases use `./build.sh check` as the gate before merge (not bare `zig build` on macOS 26.4+)
- All validation commands must use `./build.sh` wrapper on macOS 26.4+ per CLAUDE.md

## Files Inventory

**Phase 1 (8 files):** `.gitignore`, `CLAUDE.md`, `GEMINI.md`, `build.zig`, `target_contract.zig`, `persistence.zig` (x2)
**Phase 2 (6+ files):** `build.zig` (multiple edit sites: options, FeatureFlags struct, cross_opts blocks, doctor step), `src/root.zig`, `feature_catalog.zig`, `feature_parity_tests.zig`, protocol stubs
**Phase 3 (20 files):** 10 mod.zig + 10 stub.zig under `src/features/ai/`
**Phase 4 (30+ files):** 9 decompositions + 4 facade refactors + AI mod.zig doc + new sub-files
