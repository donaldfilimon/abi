# Full Codebase Improvement Implementation Plan

> **Status (2026-04-05):** COMPLETE. Phases 1-3 done by 2026-03-26. Phase 4 decompositions already done via prior PRs. Section header conversions completed 2026-03-27. Baseline synced to 3720 passed.
> Remaining work plan: `docs/superpowers/plans/2026-03-27-codebase-improvement-remaining.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the entire ABI codebase across git hygiene, protocol gating, stub parity, file decomposition, and documentation.

**Architecture:** Risk-layered execution — zero-risk changes first, then additive feature gates, then parity fixes, then structural refactoring. Each phase is independently mergeable with its own validation gate.

**Tech Stack:** Zig 0.17, `./build.sh` wrapper (macOS 26.4+), `zig build check-parity`, `zig build cross-check`

**Spec:** `docs/superpowers/specs/2026-03-24-full-codebase-improvement-design.md`

---

## Phase 1: Zero-Risk (No Behavior Change)

### Task 1: Git Hygiene — Track and Clean Files

**Files:**
- Modify: `.gitignore`
- Track: `src/features/gpu/policy/target_contract.zig`
- Delete: `test.db`, `target_contract.o`, `libcontext_init.a` (root-level stray artifacts)

- [ ] **Step 1: Remove stray root-level artifacts**

```bash
rm -f test.db target_contract.o libcontext_init.a
```

- [ ] **Step 2: Track the GPU policy contract file**

```bash
git add src/features/gpu/policy/target_contract.zig
```

- [ ] **Step 3: Update .gitignore**

Add these lines near the existing build artifacts section (around line 8):
```gitignore
# Root-level build artifacts (stray from manual builds)
/test.db
/target_contract.o
/libcontext_init.a
```

- [ ] **Step 4: Verify clean status**

```bash
git status
```

Expected: `target_contract.zig` staged, stray files gone, `.gitignore` modified.

- [ ] **Step 5: Commit**

```bash
git add .gitignore src/features/gpu/policy/target_contract.zig
git commit -m "chore: track target_contract.zig, clean stray artifacts, update gitignore"
```

---

### Task 2: Add Doc Comments to Internal Files

**Files:**
- Modify: `src/features/gpu/policy/target_contract.zig` (line 1 — no doc comment exists)

Note: Spec 1b also mentions `src/core/database/persistence.zig` and `src/tasks/persistence.zig`, but both already have good module-level doc comments ("WDBX Binary Persistence" and "Task Persistence" respectively). Skipping — no changes needed.

- [ ] **Step 1: Add module-level doc to target_contract.zig**

Insert at line 1 (before `const std = @import("std");`):

```zig
//! GPU Backend Policy Contract Validator
//!
//! Compile-time contract that validates GPU backend resolution for the build
//! target. Uses @compileError to catch misconfigurations before runtime.
//! Included in the `typecheck` build step as a standalone object build.

```

- [ ] **Step 2: Verify it compiles**

```bash
./build.sh typecheck
```

Expected: passes (doc comments don't affect compilation).

- [ ] **Step 3: Commit**

```bash
git add src/features/gpu/policy/target_contract.zig
git commit -m "docs: add module-level doc comment to GPU policy contract validator"
```

---

### Task 3: Add Error Handling Convention to CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (after line 226, end of "Key Conventions" section)

- [ ] **Step 1: Add the convention section**

Insert after the existing Key Conventions bullet points (after line 226 — the integration test accessor rule) and before "## Zig 0.17 Gotchas":

```markdown

### Error Handling Convention

- `@compileError` — compile-time contract violations only (e.g., `target_contract.zig` policy enforcement)
- `@panic` — unrecoverable invariant violations; never in library code (`src/`), only in CLI entry points (`src/main.zig`) and tests
- `unreachable` — provably impossible branches where the compiler can verify exhaustiveness at comptime
- Error unions — all runtime failure paths in library code; prefer `error.FeatureDisabled` in stubs
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add error handling convention to CLAUDE.md"
```

---

### Task 4: Doc Consistency — Verify GEMINI.md

**Files:**
- Modify: `GEMINI.md` (line 47 — feature count mention)

- [ ] **Step 1: Check GEMINI.md feature counts**

Read `GEMINI.md` line 47. It should say "21 feature directories (60 features total including AI sub-features and protocols)".

If it matches CLAUDE.md, skip to Step 3. If stale, proceed to Step 2.

- [ ] **Step 2: Fix any stale counts**

Update to match CLAUDE.md. The current repository already uses the updated 21-dir / 60-feature counts, so keep this section aligned with those values.

- [ ] **Step 3: Commit (if changes made)**

```bash
git add GEMINI.md
git commit -m "docs: sync GEMINI.md feature counts with CLAUDE.md"
```

---

### Task 5: Verify Build Step Descriptions (Spec 1e)

**Files:**
- Check: `build.zig` (alias steps around lines 518-520)

- [ ] **Step 1: Read the alias step descriptions in build.zig**

Search for `"cli-tests"`, `"tui-tests"`, `"dashboard-smoke"`, `"validate-flags"` in `build.zig`. Check if their description strings are already differentiated.

- [ ] **Step 2: If descriptions are generic/identical, differentiate them**

Only modify if needed. Expected good descriptions:
- `cli-tests` → "Run CLI tests"
- `tui-tests` → "Run TUI tests"
- `dashboard-smoke` → "Run dashboard smoke tests"
- `validate-flags` → "Validate feature flags"

If they already have distinct descriptions, skip to Task 6.

- [ ] **Step 3: Commit (if changes made)**

```bash
git add build.zig
git commit -m "docs: differentiate alias build step descriptions"
```

---

### Task 6: Phase 1 Validation

- [ ] **Step 1: Run lint**

```bash
./build.sh lint
```

Expected: clean (no formatting changes from doc comments).

- [ ] **Step 2: Run tests to confirm no regressions**

```bash
./build.sh test --summary all 2>&1 | grep -E "passed|skipped|failed"
```

Expected: 3266+ passed, 4 skipped, 0 failed.

---

## Phase 2: Verify Protocol Feature Gates (Already Implemented)

> **Note:** Phase 2 from the spec (adding `feat_acp`/`feat_ha` gates) is already fully implemented in the current codebase. The build options, comptime gates in `root.zig`, feature catalog entries, and parity tests all exist. This phase verifies correctness and updates documentation.

### Task 7: Verify ACP/HA Feature Gates Work Correctly

- [ ] **Step 1: Verify disabled-protocol build passes**

```bash
./build.sh -Dfeat-acp=false -Dfeat-ha=false test --summary all 2>&1 | tail -5
```

Expected: passes with 0 failures (may have additional skipped tests).

- [ ] **Step 2: Verify parity check passes**

```bash
./build.sh check-parity 2>&1 | grep -i "acp\|ha"
```

Expected: no parity mismatches for ACP or HA.

- [ ] **Step 3: Verify cross-check passes**

```bash
./build.sh cross-check
```

Expected: all 4 targets pass.

---

### Task 8: Update Doc Counts for ACP/HA

**Files:**
- Modify: `CLAUDE.md`, `AGENTS.md`, `README.md`, `GEMINI.md` (feature counts)

The feature catalog now has 60 features (21 feature directories under `src/features/` plus nested AI sub-features and protocol surfaces). Update all docs.

- [ ] **Step 1: Check current feature catalog count**

```bash
grep -c "^\s\+\w\+,$" src/core/feature_catalog.zig
```

Record the actual count.

- [ ] **Step 2: Update CLAUDE.md**

Find "60 features" and update to the actual count. Also update "21 feature directories" if it changed.

- [ ] **Step 3: Update AGENTS.md, README.md, GEMINI.md**

Same updates for each file — search for "60 features" or "60 features total" and update.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md AGENTS.md README.md GEMINI.md
git commit -m "docs: update feature counts to include ACP and HA protocols"
```

---

## Phase 3: Medium-Risk (AI Sub-Feature Parity Fixes)

### Task 9: Audit Current Parity State

- [ ] **Step 1: Run parity check and capture output**

```bash
./build.sh check-parity 2>&1
```

Record exact mismatches. The spec's counts are approximate — use the actual output.

- [ ] **Step 2: Create a tracking list**

For each mismatch reported, note:
- Sub-module path (e.g., `src/features/ai/agents/`)
- Direction (mod has extras vs stub has extras)
- Exact missing declarations

---

### Task 10: Fix Parity — Largest Deltas First (agents, vision, explore)

**Files:**
- Modify: `src/features/ai/agents/stub.zig` (remove extras not in mod.zig)
- Modify: `src/features/ai/vision/stub.zig` (add missing declarations from mod.zig)
- Modify: `src/features/ai/explore/stub.zig` (remove extras not in mod.zig)

For each sub-module:

- [ ] **Step 1: Read both mod.zig and stub.zig, identify exact delta**
- [ ] **Step 2: If stub has extras — remove declarations not present in mod.zig (unless they come from types.zig and are part of the intended public API)**
- [ ] **Step 3: If mod has extras — add matching no-op stub declarations to stub.zig using the pattern from `src/core/stub_helpers.zig` (StubFeature, StubContext)**
- [ ] **Step 4: Run parity check for this sub-module**

```bash
./build.sh check-parity 2>&1 | grep -i "agents\|vision\|explore"
```

- [ ] **Step 5: Commit each sub-module fix separately**

```bash
git add src/features/ai/agents/stub.zig
git commit -m "fix(parity): align ai/agents stub with mod exports"
```

---

### Task 11: Fix Parity — Medium Deltas (llm, embeddings, training, profile)

**Files:**
- Modify: `src/features/ai/llm/stub.zig`
- Modify: `src/features/ai/embeddings/stub.zig`
- Modify: `src/features/ai/training/stub.zig`
- Modify: `src/features/ai/profile/stub.zig`

Same approach as Task 10. For each:

- [ ] **Step 1: Read mod.zig and stub.zig, identify delta**
- [ ] **Step 2: Add or remove declarations to align**
- [ ] **Step 3: Run parity check**
- [ ] **Step 4: Commit each fix**

---

### Task 12: Fix Parity — Small Deltas (database, streaming, documents)

**Files:**
- Modify: `src/features/ai/database/stub.zig`
- Modify: `src/features/ai/streaming/stub.zig`
- Modify: `src/features/ai/documents/stub.zig`

Same approach. These are 2-4 declaration deltas — smallest fixes.

- [ ] **Step 1: Read, identify, fix, verify for each**
- [ ] **Step 2: Commit each fix**

---

### Task 13: Phase 3 Final Validation

- [ ] **Step 1: Run full parity check**

```bash
./build.sh check-parity
```

Expected: zero mismatches across all sub-modules.

- [ ] **Step 2: Run full test suite**

```bash
./build.sh test --summary all 2>&1 | tail -5
```

Expected: 3266+ passed, 0 failed.

- [ ] **Step 3: Commit any remaining fixes**

---

## Phase 4: Higher-Risk (Structural Refactoring)

### Task 14: Decompose Database Domain (diskann, hnsw, scann)

**Files:**
- Decompose: `src/core/database/diskann.zig` (1669 lines) → `src/core/database/diskann/` directory
- Decompose: `src/core/database/hnsw/mod.zig` (1423 lines) → extract `hnsw/search.zig`, `hnsw/insert.zig`
- Decompose: `src/core/database/scann.zig` (1238 lines) → `src/core/database/scann/` directory

Strategy for each file:

- [ ] **Step 1: Read the file and identify logical struct/function boundaries**
- [ ] **Step 2: Create sub-directory if needed (e.g., `diskann/`)**
- [ ] **Step 3: Extract structs/functions into focused sub-files**
- [ ] **Step 4: Convert original file into thin re-export facade**

Example for diskann.zig — the original becomes:
```zig
//! DiskANN Index — re-export facade
pub const PQCodebook = @import("diskann/codebook.zig").PQCodebook;
pub const VamanaGraph = @import("diskann/graph.zig").VamanaGraph;
pub const DiskANNIndex = @import("diskann/index.zig").DiskANNIndex;
// ... all original pub declarations re-exported
```

- [ ] **Step 5: Run tests after each file decomposition**

```bash
./build.sh test --summary all 2>&1 | tail -5
```

- [ ] **Step 6: Run format fix**

```bash
./build.sh fix
```

- [ ] **Step 7: Commit each decomposition**

```bash
git commit -m "refactor: decompose diskann.zig into focused sub-modules"
```

---

### Task 15: Decompose GPU Domain (ai_ops, mps, vulkan, generic, coordinator)

**Files:**
- Decompose: `src/features/gpu/ai_ops.zig` (1355 lines)
- Decompose: `src/features/gpu/backends/metal/mps.zig` (1267 lines)
- Decompose: `src/features/gpu/backends/vulkan.zig` (1173 lines)
- Decompose: `src/features/gpu/dsl/codegen/generic.zig` (1221 lines)
- Decompose: `src/features/gpu/execution_coordinator.zig` (1174 lines)

Same "thin re-export facade" strategy as Task 15.

- [ ] **Step 1: Read each file, identify struct/function boundaries**
- [ ] **Step 2: Extract into sub-files per the spec's decomposition table**
- [ ] **Step 3: Convert originals to facades**
- [ ] **Step 4: Run tests + parity after each**
- [ ] **Step 5: Run `./build.sh fix` for formatting**
- [ ] **Step 6: Commit each decomposition separately**

---

### Task 16: Decompose AI Domain (constitution/enforcement)

**Files:**
- Decompose: `src/features/ai/constitution/enforcement.zig` (1160 lines)

- [ ] **Step 1: Read and identify the 6 principle validators**
- [ ] **Step 2: Extract each principle into its own file under `enforcement/`**
- [ ] **Step 3: Convert enforcement.zig to re-export facade**
- [ ] **Step 4: Run tests + parity**
- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: decompose constitution enforcement into per-principle validators"
```

---

### Task 17: Add Sub-Namespace Facades (network, abbey, training, gpu)

**Files:**
- Modify: `src/features/network/mod.zig` (191 exports)
- Modify: `src/features/ai/abbey/mod.zig` (140 exports)
- Modify: `src/features/ai/training/mod.zig` (123 exports)
- Modify: `src/features/gpu/mod.zig` (119 exports)

**Critical:** All existing flat exports must be preserved as aliases. New sub-namespaces are additive.

For each:

- [ ] **Step 1: Read mod.zig and categorize exports into logical groups**
- [ ] **Step 2: Create intermediate sub-namespace files (e.g., `network/http.zig`)**
- [ ] **Step 3: Add sub-namespace imports to mod.zig alongside existing flat exports**

Example for network/mod.zig:
```zig
// New sub-namespace re-exports (additive)
pub const http = @import("http.zig");
pub const dns = @import("dns.zig");
pub const socket = @import("socket.zig");

// Existing flat exports preserved (backwards compat)
pub const HttpClient = http.HttpClient;
pub const DnsResolver = dns.DnsResolver;
// ... all existing pub declarations unchanged
```

- [ ] **Step 4: Run tests + parity**
- [ ] **Step 5: Commit each facade separately**

---

### Task 18: Add AI Namespace Documentation

**Files:**
- Modify: `src/features/ai/mod.zig` (add conceptual grouping doc comment)

- [ ] **Step 1: Add module doc comment at top of ai/mod.zig**

```zig
//! AI Feature Module
//!
//! This module contains 33 sub-directories organized into conceptual groups:
//!
//! **Inference:** llm, embeddings, vision, models, streaming
//! **Reasoning:** abbey, aviva, abi, constitution, eval, reasoning
//! **Agents:** agents, tools, multi_agent, coordination, orchestration
//! **Learning:** training, memory, federated
//! **Support:** templates, prompts, documents, profiles, context_engine
//!
//! Directory structure is flat (not reorganized by group) to minimize import churn.
```

- [ ] **Step 2: Commit**

```bash
git add src/features/ai/mod.zig
git commit -m "docs: add conceptual grouping documentation to AI module"
```

---

## Phase 5: Final Validation

### Task 19: Full Gate Validation and Baseline Sync

- [ ] **Step 1: Full test suite**

```bash
./build.sh test --summary all 2>&1 | grep -E "passed|skipped|failed"
```

Expected: 3266+ passed, 4 skipped, 0 failed.

- [ ] **Step 2: Parity check**

```bash
./build.sh check-parity
```

Expected: zero mismatches.

- [ ] **Step 3: Cross-compilation**

```bash
./build.sh cross-check
```

Expected: all 4 targets pass.

- [ ] **Step 4: Lint**

```bash
./build.sh lint
```

Expected: clean.

- [ ] **Step 5: Full gate**

```bash
./build.sh check
```

Expected: passes (lint + test + parity).

- [ ] **Step 6: Update baseline**

Update `.claude/skills/baseline-sync/SKILL.md` with new test counts and date.

- [ ] **Step 7: Final commit**

```bash
git add .claude/skills/baseline-sync/SKILL.md
git commit -m "chore: sync test baseline after full codebase improvement"
```
