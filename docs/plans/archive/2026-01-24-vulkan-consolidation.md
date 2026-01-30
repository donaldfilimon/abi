# Vulkan Backend Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge four separate Vulkan backend files into a single unified `vulkan.zig` module for improved maintainability.

**Architecture:** Consolidate `vulkan_cache.zig`, `vulkan_command_pool.zig`, and `vulkan_vtable.zig` into the existing `vulkan.zig`. Update all imports to reference the unified module. Remove the deprecated files.

**Tech Stack:** Zig 0.16, Vulkan API bindings

---

## Current State Analysis

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `vulkan.zig` | ~600 | Main implementation | Keep, extend |
| `vulkan_vtable.zig` | ~30 | VTable factory wrapper | Merge into vulkan.zig |
| `vulkan_cache.zig` | 6 | Empty ShaderCache stub | Merge into vulkan.zig |
| `vulkan_command_pool.zig` | 7 | Empty CommandPool stub | Merge into vulkan.zig |

---

## Task 1: Add Stub Types to vulkan.zig

**Files:**
- Modify: `src/gpu/backends/vulkan.zig` (add at end of file)

**Step 1: Read current vulkan.zig to find insertion point**

Read the end of `src/gpu/backends/vulkan.zig` to understand structure.

**Step 2: Add cache and command pool stubs**

Add at the end of `vulkan.zig`, before final closing brace if any:

```zig
// ============================================================================
// Shader Cache (stub)
// ============================================================================

/// Shader cache stub for future caching implementation.
/// Currently a placeholder to satisfy imports.
pub const ShaderCache = struct {};

// ============================================================================
// Command Pool (stub)
// ============================================================================

/// Command pool stub for future pooling implementation.
/// Currently a placeholder to satisfy imports.
pub const CommandPool = struct {};
```

**Step 3: Verify build**

Run: `zig build 2>&1 | head -10`
Expected: Build succeeds (no errors related to vulkan)

**Step 4: Commit**

```bash
git add src/gpu/backends/vulkan.zig
git commit -m "feat(gpu): add ShaderCache and CommandPool stubs to vulkan.zig"
```

---

## Task 2: Merge VTable Factory Function

**Files:**
- Modify: `src/gpu/backends/vulkan.zig` (add at end)
- Read: `src/gpu/backends/vulkan_vtable.zig` (for reference)

**Step 1: Add VTable imports and factory function**

Add to `vulkan.zig` after the stub types:

```zig
// ============================================================================
// VTable Factory
// ============================================================================

const interface = @import("../interface.zig");

/// Creates a Vulkan backend instance wrapped in the VTable interface.
///
/// Returns BackendError.NotAvailable if Vulkan is disabled at compile time
/// or the Vulkan driver cannot be loaded.
/// Returns BackendError.InitFailed if Vulkan initialization fails.
pub fn createVulkanVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    // Check if Vulkan is enabled at compile time
    if (comptime !build_options.gpu_vulkan) {
        return interface.BackendError.NotAvailable;
    }

    // Use existing implementation from vulkan_vtable
    return vulkan_vtable.createVulkanVTable(allocator);
}
```

**Step 2: Verify build**

Run: `zig build 2>&1 | head -10`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/gpu/backends/vulkan.zig
git commit -m "feat(gpu): add createVulkanVTable factory to vulkan.zig"
```

---

## Task 3: Update External Imports

**Files:**
- Modify: `src/gpu/backends/mod.zig` or any file importing vulkan_vtable/vulkan_cache

**Step 1: Search for imports of deprecated files**

Run: `grep -r "vulkan_vtable\|vulkan_cache\|vulkan_command_pool" src/gpu --include="*.zig" | grep -v "vulkan_vtable.zig\|vulkan_cache.zig\|vulkan_command_pool.zig"`

**Step 2: Update each import**

For each file found, change:
- `@import("vulkan_vtable.zig")` → `@import("vulkan.zig")`
- `@import("vulkan_cache.zig")` → `@import("vulkan.zig")`
- `@import("vulkan_command_pool.zig")` → `@import("vulkan.zig")`

**Step 3: Verify build**

Run: `zig build 2>&1 | head -10`
Expected: Build succeeds

**Step 4: Run tests**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: All tests pass (194/198)

**Step 5: Commit**

```bash
git add src/gpu/
git commit -m "refactor(gpu): update imports to use consolidated vulkan.zig"
```

---

## Task 4: Remove Deprecated Files

**Files:**
- Delete: `src/gpu/backends/vulkan_vtable.zig`
- Delete: `src/gpu/backends/vulkan_cache.zig`
- Delete: `src/gpu/backends/vulkan_command_pool.zig`

**Step 1: Remove the files**

```bash
git rm src/gpu/backends/vulkan_vtable.zig
git rm src/gpu/backends/vulkan_cache.zig
git rm src/gpu/backends/vulkan_command_pool.zig
```

**Step 2: Verify build**

Run: `zig build 2>&1 | head -10`
Expected: Build succeeds

**Step 3: Run tests**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: All tests pass (194/198)

**Step 4: Commit**

```bash
git commit -m "refactor(gpu): remove deprecated vulkan split files"
```

---

## Task 5: Update Documentation

**Files:**
- Modify: `ROADMAP.md` (mark task complete)
- Delete: `docs/plans/2026-01-23-vulkan-combine.md` reference (if exists)

**Step 1: Update ROADMAP.md**

Change:
```markdown
- [ ] Vulkan backend consolidation
  - [ ] Merge four Vulkan source files into a single module (`vulkan.zig`)
```

To:
```markdown
- [x] Vulkan backend consolidation
  - [x] Merge four Vulkan source files into a single module (`vulkan.zig`)
```

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: mark Vulkan consolidation complete in roadmap"
```

---

## Task 6: Final Verification

**Step 1: Full build**

Run: `zig build 2>&1`
Expected: Build succeeds with no errors

**Step 2: Full test suite**

Run: `zig build test --summary all 2>&1 | tail -10`
Expected: 194/198 tests pass

**Step 3: GPU backend verification**

Run: `zig build run -- gpu backends 2>&1`
Expected: Shows Vulkan as enabled with 1 device

---

## Summary

| Before | After |
|--------|-------|
| 4 files: vulkan.zig, vulkan_vtable.zig, vulkan_cache.zig, vulkan_command_pool.zig | 1 file: vulkan.zig |
| ~650 total lines | ~660 lines (consolidated) |
| Scattered imports | Single import point |

**Total estimated tasks:** 6
**Commits:** 5
