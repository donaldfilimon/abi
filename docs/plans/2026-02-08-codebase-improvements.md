# Codebase Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the split-large-files refactor (wire SIMD module), integrate all new files into the build, and fix code quality issues across the codebase.

**Architecture:** Three parallel workstreams: (A) complete SIMD split, (B) wire new files into parent modules, (C) fix code quality issues. Streams A and B are independent. Stream C depends on A and B.

**Tech Stack:** Zig 0.16.0-dev.2535+b5bd49460, test baseline 980 pass / 5 skip

---

## Stream A: Complete SIMD Module Split (CRITICAL)

### Task 1: Create `simd/mod.zig` and wire into build

**Files:**
- Create: `src/services/shared/simd/mod.zig`
- Modify: `src/services/shared/mod.zig:75`
- Remove: `src/services/shared/simd.zig` (after verification)

**Context:** Five submodule files already exist in `src/services/shared/simd/` (`vector_ops.zig`, `activations.zig`, `distances.zig`, `integer_ops.zig`, `extras.zig`) but there's no `mod.zig` to re-export them, and `shared/mod.zig` still imports `simd.zig`.

**Step 1: Read the original `simd.zig` to catalog all public symbols**

Read: `src/services/shared/simd.zig` (full file, ~1993 lines)
Catalog every `pub fn` and `pub const` — these must ALL appear in `mod.zig`.

**Step 2: Read each subdirectory file to verify symbol coverage**

Read all 5 files in `src/services/shared/simd/` and cross-reference against the catalog from Step 1.

**Step 3: Create `src/services/shared/simd/mod.zig`**

```zig
//! SIMD vector operations
//!
//! Re-exports from focused submodules. Every public symbol from the
//! original monolithic simd.zig is available here.

pub const vector_ops = @import("vector_ops.zig");
pub const activations = @import("activations.zig");
pub const distances = @import("distances.zig");
pub const integer_ops = @import("integer_ops.zig");
pub const extras = @import("extras.zig");

// Re-export all public functions so callers don't need to know the submodule structure.
// From vector_ops.zig:
pub const vectorAdd = vector_ops.vectorAdd;
pub const vectorDot = vector_ops.vectorDot;
// ... (enumerate ALL pub fns from each submodule)

// Include tests in test builds
comptime {
    if (@import("builtin").is_test) {
        _ = vector_ops;
        _ = activations;
        _ = distances;
        _ = integer_ops;
        _ = extras;
    }
}
```

Every public symbol from the original `simd.zig` MUST be re-exported. Missing any will break downstream callers.

**Step 4: Update `src/services/shared/mod.zig`**

Change line 75 from:
```zig
pub const simd = @import("simd.zig");
```
to:
```zig
pub const simd = @import("simd/mod.zig");
```

**Step 5: Run tests**

```bash
zig fmt . && zig build test --summary all
```

Expected: 980 pass, 5 skip. If tests fail, check which symbols are missing from `mod.zig`.

**Step 6: Delete old `simd.zig`**

Only after tests pass:
```bash
git rm src/services/shared/simd.zig
```

**Step 7: Run tests again to confirm**

```bash
zig build test --summary all
```

**Step 8: Commit**

```bash
git add src/services/shared/simd/ src/services/shared/mod.zig
git commit -m "refactor: complete simd.zig split into subdirectory modules"
```

---

## Stream B: Wire New Files Into Parent Modules

### Task 2: Wire AI training extracted files

**Files:**
- Verify: `src/features/ai/training/self_learning.zig` (re-exports from extracted modules)
- Verify: `src/features/ai/training/mod.zig` (imports self_learning)
- Verify test discovery for: `self_learning_test.zig`, `trainable_model_test.zig`

**Step 1: Read `self_learning.zig` to verify re-exports**

Check that it re-exports from `learning_types.zig`, `experience_buffer.zig`, `reward_policy.zig`, `dpo_optimizer.zig`.

**Step 2: Read `training/mod.zig` to verify test discovery**

Check the `comptime { if (@import("builtin").is_test) { ... } }` block includes the new test files.

**Step 3: Add test discovery if missing**

If the test files aren't in the comptime block, add them:
```zig
_ = @import("self_learning_test.zig");
_ = @import("trainable_model_test.zig");
```

**Step 4: Run tests**

```bash
zig build test --summary all
```

**Step 5: Commit**

```bash
git add src/features/ai/training/
git commit -m "refactor: wire training submodules and add test discovery"
```

---

### Task 3: Wire GPU extracted files

**Files:**
- Verify: `src/features/gpu/dispatcher.zig` (re-exports dispatch_types, batched_dispatch)
- Verify: `src/features/gpu/multi_device.zig` (re-exports device_group, gpu_cluster, gradient_sync)
- Verify: `src/features/gpu/mod.zig` (test discovery for new test files)
- Verify: `src/features/gpu/backends/vulkan.zig` (re-exports vulkan_types)
- Verify: `src/features/gpu/backends/metal.zig` (re-exports metal_types)

**Step 1: Read `dispatcher.zig` — verify it re-exports from `dispatch_types.zig` and `batched_dispatch.zig`**

**Step 2: Read `multi_device.zig` — verify it re-exports from `device_group.zig`, `gpu_cluster.zig`, `gradient_sync.zig`**

**Step 3: Read `gpu/mod.zig` — verify test discovery includes `dispatcher_test.zig`, `multi_device_test.zig`**

**Step 4: Read `vulkan.zig` and `metal.zig` — verify type re-exports**

**Step 5: Add any missing re-exports or test discovery entries**

**Step 6: Run tests**

```bash
zig build test --summary all
```

**Step 7: Commit**

```bash
git add src/features/gpu/
git commit -m "refactor: wire GPU submodules and add test discovery"
```

---

### Task 4: Wire database extracted files

**Files:**
- Verify: `src/features/database/hnsw.zig` (re-exports search_state, distance_cache)
- Verify: Test discovery for `hnsw_test.zig`

**Step 1: Read `hnsw.zig` — verify re-exports**

**Step 2: Check test discovery in `database/mod.zig`**

**Step 3: Add missing entries, run tests, commit**

```bash
zig build test --summary all
git add src/features/database/
git commit -m "refactor: wire database submodules and add test discovery"
```

---

### Task 5: Wire streaming extracted files

**Files:**
- Verify: `src/features/ai/streaming/server.zig` (re-exports request_types)
- Verify: Test discovery for `server_test.zig`

**Step 1: Read `server.zig` — verify re-export of AbiStreamRequest**

**Step 2: Check test discovery in streaming module**

**Step 3: Add missing entries, run tests, commit**

```bash
zig build test --summary all
git add src/features/ai/streaming/
git commit -m "refactor: wire streaming submodules and add test discovery"
```

---

## Stream C: Code Quality Fixes

### Task 6: Fix `std.debug.print` in library code

**Files:**
- Modify: `src/services/shared/logging.zig`

**Step 1: Read `logging.zig`**

Lines 19-21, 59 use `std.debug.print`. This is a logging wrapper — it should use `std.io.getStdErr().writer()` or `std.log.*` internally.

**Step 2: Assess if this is intentional**

If `logging.zig` is a low-level log backend that writes to stderr, `std.debug.print` may be intentional (it writes to stderr). If so, add a comment explaining why. If not, convert to `std.log.*`.

**Step 3: Run tests, commit**

---

### Task 7: Replace `catch unreachable` in non-test code with proper error handling

**Files (prioritized by risk):**
- `src/features/network/discovery.zig` (4 instances, buffer formatting)
- `src/features/network/discovery_types.zig` (1 instance)
- `src/features/network/linking/internet.zig` (2 instances)
- `src/features/web/middleware/auth.zig` (1 instance)
- `src/services/shared/utils/json/mod.zig` (3 instances)

**Step 1: Read each file and assess each `catch unreachable`**

Most are `bufPrint catch unreachable` — formatting into a fixed-size buffer. If the buffer is provably large enough, add a `// SAFETY:` comment explaining why. If it could overflow, replace with error propagation.

**Step 2: For each instance, either:**
- Add `// SAFETY: buffer is large enough because [reason]` comment if provably safe
- Or replace `catch unreachable` with `catch |e| return e` or `catch return error.FormatError`

**Step 3: Run tests, commit**

```bash
zig build test --summary all
git commit -m "fix: add safety comments for catch unreachable patterns"
```

---

### Task 8: Run full validation

**Step 1: Full check**

```bash
zig build full-check
```

This runs: format check + tests + flag validation (16 combos) + CLI smoke tests.

**Step 2: Verify baseline**

Must be: 980 pass, 5 skip (or higher pass count if new tests were added).

**Step 3: Final commit if needed**

---

## Parallel Execution Strategy

Tasks 1, 2-5, and 6-7 can run in parallel:
- **Agent A**: Task 1 (SIMD split — most critical)
- **Agent B**: Tasks 2-5 (wire new files — independent per module)
- **Agent C**: Tasks 6-7 (code quality — independent of wiring)

Task 8 runs last (depends on all others).

---

## Final Verification

After all tasks:

```bash
zig build test --summary all    # Must be 980 pass, 5 skip (or more)
zig build validate-flags        # Must pass all 16 flag combos
zig fmt .                       # Must be clean
```
