# Codebase Improvement — Remaining Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining 20% of the full codebase improvement plan (2026-03-24) — two file decompositions, sub-namespace facades for large modules, and baseline sync.

**Architecture:** The original 5-phase plan is ~80% complete. Phases 1-3 are fully done. Phase 4 has two remaining decompositions (ai_ops.zig at 669 lines, scann.zig at 385 lines) and three remaining sub-namespace facade additions (network, training, gpu). Phase 5 needs a baseline sync update (3675 passed vs 3266 original baseline).

**Tech Stack:** Zig 0.16, `./build.sh` wrapper (macOS 26.4+), `zig build check-parity`

**Spec:** `docs/superpowers/specs/2026-03-24-full-codebase-improvement-design.md`

**Prior plan:** `docs/superpowers/plans/2026-03-24-full-codebase-improvement.md`

---

## Completion Status (as of 2026-03-27)

| Original Phase | Status | Notes |
|---|---|---|
| Phase 1: Zero-Risk | **DONE** | Git hygiene, doc comments, error convention, GEMINI.md, build step descriptions |
| Phase 2: Protocol Gates | **DONE** | ACP/HA comptime-gated, docs say 35 features |
| Phase 3: Parity Fixes | **DONE** | `check-parity` passes, exit 0 |
| Phase 4: Structural Refactoring | **80% done** | Most decompositions done. Remaining: ai_ops (669L), scann (385L), 3 facades |
| Phase 5: Final Validation | **Pending** | Baseline needs update: 3675 passed (was 3266) |

**Current baseline:** 3675 passed, 4 skipped, 0 failed. Parity clean. Cross-check clean.

---

## Task 1: ~~Decompose `src/features/gpu/ai_ops.zig`~~ SKIP — already decomposed

> **Status:** SKIP. The file is already a thin facade (3 sub-modules exist: `cpu_fallback.zig`, `adapters.zig`, `reexports.zig`). The 669 lines are interface types (must stay due to circular deps) + 220 lines of tests. No further decomposition needed.

**Files:**
- Read: `src/features/gpu/ai_ops.zig`
- Create: `src/features/gpu/ai_ops/vtable.zig`
- Create: `src/features/gpu/ai_ops/adapters.zig`
- Create: `src/features/gpu/ai_ops/operations.zig`
- Modify: `src/features/gpu/ai_ops.zig` → thin re-export facade

The spec calls for splitting into vtable (interface definitions) and adapters (backend wrappers). Read the file first to identify the actual struct boundaries.

- [x] **Step 1: Read ai_ops.zig and map struct boundaries**

```bash
grep -n "^pub const\|^const\|^pub fn\|^pub const.*= struct" src/features/gpu/ai_ops.zig | head -40
```

Record which structs/functions exist and their line ranges. The file likely contains:
- `AiOps` VTable interface type
- Backend adapter implementations
- Helper/utility functions

- [x] **Step 2: Create the `ai_ops/` sub-directory**

```bash
mkdir -p src/features/gpu/ai_ops
```

- [x] **Step 3: Extract VTable interface into `ai_ops/vtable.zig`**

Move the `AiOps` interface struct (the VTable definition and its associated types) into `vtable.zig`. Add a `//!` doc comment at the top:

```zig
//! AiOps VTable — backend-agnostic interface for GPU AI operations.
```

Adjust imports: replace relative `@import("../../` paths with `@import("../../../` (one level deeper).

- [x] **Step 4: Extract adapter implementations into `ai_ops/adapters.zig`**

Move backend adapter structs (the concrete implementations that satisfy the VTable) into `adapters.zig`. Add doc comment:

```zig
//! Backend adapters — concrete GPU AI operation implementations.
```

- [x] **Step 5: Extract remaining operations into `ai_ops/operations.zig`**

Move standalone functions and helper operations. Add doc comment:

```zig
//! GPU AI operation implementations and helpers.
```

- [x] **Step 6: Convert `ai_ops.zig` to thin re-export facade**

Replace file contents with re-exports preserving the exact public API:

```zig
//! GPU AI Operations — re-export facade
//!
//! VTable-based backend-agnostic interface for AI operations on GPU.
//! Decomposed into vtable (interface), adapters (backends), operations (helpers).

pub const vtable = @import("ai_ops/vtable.zig");
pub const adapters = @import("ai_ops/adapters.zig");
pub const operations = @import("ai_ops/operations.zig");

// Re-export all original public declarations for backwards compatibility
pub const AiOps = vtable.AiOps;
// ... (all other pub const/fn from original file)
```

**Critical:** Every `pub` declaration from the original file must appear in the facade. Compare `grep "^pub " ai_ops.zig` before and after.

- [x] **Step 7: Run format fix**

```bash
./build.sh fix
```

- [x] **Step 8: Run tests**

```bash
./build.sh gpu-tests --summary all 2>&1 | tail -5
```

Expected: passes, exit 0.

- [x] **Step 9: Run parity check**

```bash
./build.sh check-parity 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 10: Commit**

```bash
git add src/features/gpu/ai_ops.zig src/features/gpu/ai_ops/
git commit -m "refactor: decompose gpu/ai_ops.zig into focused sub-modules"
```

---

## Task 2: ~~Decompose `src/core/database/scann.zig`~~ SKIP — already decomposed

> **Status:** SKIP. The file is already a facade (37 lines of re-exports + 348 lines of integration tests). Sub-dir exists with `types.zig`, `codebook.zig`, `index.zig`.

**Files:**
- Read: `src/core/database/scann.zig`
- Create: `src/core/database/scann/codebook.zig`
- Create: `src/core/database/scann/partitioning.zig`
- Create: `src/core/database/scann/index.zig`
- Modify: `src/core/database/scann.zig` → thin re-export facade

At 385 lines this is only moderately over the 300-line threshold. Only split if there are clear struct boundaries. If the file is a single cohesive struct, skip this task.

- [x] **Step 1: Read scann.zig and assess decomposition value**

```bash
grep -n "^pub const\|^pub fn\|= struct" src/core/database/scann.zig | head -20
```

If the file is one struct with a few methods, document "scann.zig is 385 lines but cohesive — skipping decomposition" and move to Task 3.

If there are 3+ distinct public structs (codebook, partitioning, index), proceed with decomposition.

- [x] **Step 2: Create `scann/` directory and extract sub-files**

Same "thin facade" pattern as Task 1:
```bash
mkdir -p src/core/database/scann
```

Extract each struct into its own file. Convert `scann.zig` to facade.

- [x] **Step 3: Run format fix**

```bash
./build.sh fix
```

- [x] **Step 4: Run tests**

```bash
./build.sh database-tests --summary all 2>&1 | tail -5
```

Expected: passes, exit 0.

- [x] **Step 5: Commit**

```bash
git add src/core/database/scann.zig src/core/database/scann/
git commit -m "refactor: decompose scann.zig into codebook/partitioning/index sub-modules"
```

---

## Task 3: Add Sub-Namespace Facade to `src/features/network/mod.zig` (532 lines, ~191 exports)

**Files:**
- Read: `src/features/network/mod.zig`
- Read: `src/features/network/stub.zig`
- Modify: `src/features/network/mod.zig` (add sub-namespace imports, keep flat aliases)

The network module has ~191 flat exports. Group into logical sub-namespaces **additively** — all existing `pub const Foo = ...` lines stay. New sub-namespace imports are added alongside.

- [x] **Step 1: Read mod.zig and categorize exports**

```bash
grep "^pub const\|^pub fn\|^pub var" src/features/network/mod.zig | wc -l
```

Then read the file and categorize exports into groups (e.g., HTTP-related, DNS-related, socket-related, cluster/consensus, RPC).

- [x] **Step 2: Identify which sub-module imports already exist**

The file already imports sub-modules like `cluster.zig`, `consensus.zig`, `rpc.zig`, etc. The sub-namespaces should group these existing imports, not create new files.

For example, if the file has:
```zig
pub const cluster = @import("cluster.zig");
pub const consensus = @import("consensus.zig");
pub const rpc = @import("rpc.zig");
```

Then flat re-exports like `pub const ClusterNode = cluster.ClusterNode;` can be documented with a `// ── Cluster ──` section header for readability.

- [x] **Step 3: Add section headers to organize flat exports**

Insert ASCII section headers (matching the codebase style) to group related exports:

```zig
// ── Cluster & Consensus ─────────────────────────────────────────────
pub const cluster = @import("cluster.zig");
pub const consensus = @import("consensus.zig");
// ... existing flat re-exports for this group

// ── RPC & Transport ────────────────────────────────────────────────
pub const rpc = @import("rpc.zig");
// ... existing flat re-exports for this group

// ── Discovery & Load Balancing ──────────────────────────────────────
pub const discovery = @import("discovery.zig");
pub const loadbalancer = @import("loadbalancer.zig");
// ... existing flat re-exports for this group
```

- [x] **Step 4: Verify stub.zig doesn't need updates**

Sub-namespace grouping in mod.zig is purely organizational — the stub's public API surface doesn't change. Verify:

```bash
./build.sh check-parity 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 5: Run tests**

```bash
./build.sh network-tests --summary all 2>&1 | tail -5
```

Expected: passes, exit 0.

- [x] **Step 6: Commit**

```bash
git add src/features/network/mod.zig
git commit -m "refactor: organize network/mod.zig exports with section headers"
```

---

## Task 4: Add Sub-Namespace Facade to `src/features/ai/training/mod.zig` (525 lines)

**Files:**
- Read: `src/features/ai/training/mod.zig`
- Modify: `src/features/ai/training/mod.zig` (add section headers)

Same approach as Task 3. The file already imports sub-modules (`core_training.zig`, `models.zig`, `data.zig`, `checkpointing.zig`, `specialized.zig`). Add section headers to organize the flat re-exports.

- [x] **Step 1: Read mod.zig and identify groupings**

```bash
grep "^pub const.*= @import" src/features/ai/training/mod.zig
```

Group the re-exports under sections: Core Training, Models, Data, Checkpointing, Specialized.

- [x] **Step 2: Add section headers**

Same ASCII section header style as Task 3.

- [x] **Step 3: Run tests**

```bash
./build.sh test --summary all 2>&1 | tail -5
```

Expected: passes, exit 0.

- [x] **Step 4: Commit**

```bash
git add src/features/ai/training/mod.zig
git commit -m "refactor: organize training/mod.zig exports with section headers"
```

---

## Task 5: Add Sub-Namespace Facade to `src/features/gpu/mod.zig` (564 lines)

**Files:**
- Read: `src/features/gpu/mod.zig`
- Modify: `src/features/gpu/mod.zig` (add section headers)

Same approach. The file already imports sub-modules (`core_gpu.zig`, `execution.zig`, `memory_ns.zig`, `advanced.zig`, `backend.zig`, `runtime_kernels.zig`, etc.). Add section headers.

- [x] **Step 1: Read mod.zig and identify groupings**

```bash
grep "^pub const.*= @import" src/features/gpu/mod.zig
```

Group under: Core GPU, Execution, Memory, Backends, Kernels, Advanced/Profiling.

- [x] **Step 2: Add section headers**

Same ASCII section header style.

- [x] **Step 3: Run parity check (GPU stubs are sensitive)**

```bash
./build.sh check-parity 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 4: Run tests**

```bash
./build.sh gpu-tests --summary all 2>&1 | tail -5
```

Expected: passes, exit 0.

- [x] **Step 5: Commit**

```bash
git add src/features/gpu/mod.zig
git commit -m "refactor: organize gpu/mod.zig exports with section headers"
```

---

## Task 6: Final Validation and Baseline Sync

- [x] **Step 1: Run full test suite**

```bash
./build.sh test --summary all 2>&1 | tail -10
```

Expected: 3675+ passed, 4 skipped, 0 failed.

- [x] **Step 2: Run parity check**

```bash
./build.sh check-parity 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 3: Run cross-compilation check**

```bash
./build.sh cross-check 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 4: Run lint**

```bash
./build.sh lint 2>&1; echo "EXIT: $?"
```

Expected: exit 0.

- [x] **Step 5: Update baseline in SKILL.md**

Edit `.claude/skills/baseline-sync/SKILL.md` line 68-71. Replace the old baseline with:

    ## Current Baseline (2026-03-27)

    3675 passed, 4 skipped, 0 failed (exit 0) | zig: 0.16.0-dev.2984+cb7d2b056
    Build Summary: 6/6 steps succeeded
    Note: macOS 26.4+ requires ./build.sh wrapper

- [x] **Step 6: Commit baseline update**

```bash
git add .claude/skills/baseline-sync/SKILL.md
git commit -m "chore: sync test baseline — 3675 passed, 4 skipped (was 3266)"
```

- [x] **Step 7: Mark original plan as complete**

Add a completion note at the top of `docs/superpowers/plans/2026-03-24-full-codebase-improvement.md`:

```markdown
> **Status (2026-03-27):** COMPLETE. Phases 1-3 done by 2026-03-26. Phase 4 decompositions completed 2026-03-27.
> Revised plan: `docs/superpowers/plans/2026-03-27-codebase-improvement-remaining.md`
```

- [x] **Step 8: Final commit**

```bash
git add docs/superpowers/plans/
git commit -m "docs: mark full codebase improvement plan as complete"
```
