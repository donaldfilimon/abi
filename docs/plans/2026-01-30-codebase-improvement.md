# ABI Codebase Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Continue improving ABI codebase quality, consistency, and production-readiness.

**Architecture:** Incremental improvements following existing patterns. Focus on code consistency, test coverage, and documentation alignment. Each task is independent and can be completed in isolation.

**Tech Stack:** Zig 0.16, existing ABI patterns (ArrayListUnmanaged, {t} format specifier, std.Io API)

---

## Task 1: Standardize Error Module

**Files:**
- Create: `src/shared/errors.zig`
- Modify: `src/shared/mod.zig`
- Test: `src/tests/error_handling_test.zig`

**Step 1: Read existing error patterns**

Run: `grep -r "pub const.*Error = error{" src/ | head -20`
Document the patterns used across modules.

**Step 2: Create shared error module**

```zig
//! Shared error types for cross-module error handling.
//!
//! Provides common error sets that multiple modules can use,
//! reducing duplication and enabling consistent error handling.

const std = @import("std");

/// Common resource errors across all modules.
pub const ResourceError = error{
    OutOfMemory,
    ResourceExhausted,
    Timeout,
    Cancelled,
};

/// Common I/O errors for file and network operations.
pub const IoError = error{
    ConnectionRefused,
    ConnectionReset,
    EndOfStream,
    InvalidData,
};

/// Feature availability errors.
pub const FeatureError = error{
    FeatureDisabled,
    NotSupported,
    NotImplemented,
};

/// Combine multiple error sets.
pub fn combineErrors(comptime sets: anytype) type {
    var result = sets[0];
    inline for (sets[1..]) |set| {
        result = result || set;
    }
    return result;
}

test "error combination" {
    const Combined = combineErrors(.{ ResourceError, IoError });
    _ = @as(Combined, error.OutOfMemory);
    _ = @as(Combined, error.ConnectionRefused);
}
```

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: 787+ tests pass

**Step 4: Commit**

```bash
git add src/shared/errors.zig src/shared/mod.zig
git commit -m "feat(shared): add standardized error module

Provides common error sets for cross-module error handling.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Missing Inline Tests

**Files:**
- Modify: `src/config/loader.zig` (add more tests)
- Modify: `src/platform/detection.zig` (add edge case tests)

**Step 1: Identify files with low test coverage**

Run: `find src/ -name "*.zig" -exec grep -L "^test " {} \; | head -10`
List files without inline tests.

**Step 2: Add tests to loader.zig**

```zig
test "ConfigLoader handles missing env vars gracefully" {
    var loader = ConfigLoader.init(std.testing.allocator);
    defer loader.deinit();

    const config = try loader.load();
    // Should use defaults when env vars are not set
    try std.testing.expect(config.gpu != null or !build_options.enable_gpu);
}

test "parseGpuBackend handles all valid backends" {
    const backends = [_][]const u8{ "auto", "cuda", "vulkan", "metal", "webgpu", "opengl", "none" };
    for (backends) |backend| {
        const result = ConfigLoader.parseGpuBackend(backend);
        try std.testing.expect(result != null);
    }
}
```

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: 790+ tests pass

**Step 4: Commit**

```bash
git add src/config/loader.zig src/platform/detection.zig
git commit -m "test: add inline tests for config loader and platform detection

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Sync Documentation Dates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `GEMINI.md`

**Step 1: Check current dates**

Run: `grep -n "2026-01" CLAUDE.md AGENTS.md GEMINI.md | head -20`

**Step 2: Update all dates to current**

Update any outdated dates to 2026-01-30 where appropriate.

**Step 3: Verify no broken links**

Run: `grep -o '\[.*\](.*\.md)' README.md | head -10`

**Step 4: Commit**

```bash
git add CLAUDE.md AGENTS.md GEMINI.md
git commit -m "docs: sync documentation dates to 2026-01-30

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Verify All Feature Flag Combinations

**Files:**
- Test: Build system verification

**Step 1: Test all disabled combinations**

Run each command and verify success:

```bash
zig build -Denable-ai=false
zig build -Denable-gpu=false
zig build -Denable-database=false
zig build -Denable-network=false
zig build -Denable-web=false
zig build -Denable-profiling=false
```

**Step 2: Test combined disabled flags**

```bash
zig build -Denable-ai=false -Denable-gpu=false
zig build -Denable-database=false -Denable-network=false
```

**Step 3: Document any failures**

If any combination fails, create a tracking issue.

**Step 4: Update PLAN.md with verification results**

---

## Task 5: Clean Up Unused Imports

**Files:**
- Various source files identified by linter

**Step 1: Find unused imports**

Run: `zig build 2>&1 | grep -i "unused" | head -20`

**Step 2: Remove unused imports**

Edit files to remove any unused imports identified.

**Step 3: Format code**

Run: `zig fmt .`

**Step 4: Run tests and commit**

```bash
zig build test --summary all
git add -A
git commit -m "refactor: remove unused imports

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add C Bindings Headers

**Files:**
- Create: `bindings/c/abi.h`
- Create: `bindings/c/abi_types.h`
- Create: `bindings/c/README.md`

**Step 1: Create directory structure**

```bash
mkdir -p bindings/c
```

**Step 2: Create main header file**

```c
/* abi.h - C bindings for ABI Framework
 *
 * This header provides C-compatible access to ABI functionality.
 *
 * Usage:
 *   #include <abi.h>
 *
 *   abi_framework_t fw = NULL;
 *   abi_init(&fw);
 *   // ... use framework ...
 *   abi_shutdown(fw);
 */

#ifndef ABI_H
#define ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef void* abi_framework_t;
typedef void* abi_database_t;

/* Error codes */
typedef enum {
    ABI_OK = 0,
    ABI_ERROR_INVALID_PARAM = -1,
    ABI_ERROR_OUT_OF_MEMORY = -2,
    ABI_ERROR_NOT_INITIALIZED = -3,
    ABI_ERROR_FEATURE_DISABLED = -4,
} abi_error_t;

/* Framework lifecycle */
abi_error_t abi_init(abi_framework_t* fw);
abi_error_t abi_shutdown(abi_framework_t fw);
const char* abi_version(void);

/* SIMD operations */
float abi_simd_dot_product(const float* a, const float* b, size_t len);
void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* ABI_H */
```

**Step 3: Build and verify**

```bash
zig build
```

**Step 4: Commit**

```bash
git add bindings/c/
git commit -m "feat(bindings): add C header files for FFI

Provides C-compatible API for framework, SIMD operations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Test Count in README

**Files:**
- Modify: `README.md`

**Step 1: Get current test count**

Run: `zig build test --summary all 2>&1 | grep "tests passed"`

**Step 2: Update README badge**

Update the badge to show current test count (787 passing).

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update test count badge in README

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

After completing all tasks:
- Standardized error handling across modules
- Improved test coverage with inline tests
- Synced documentation dates
- Verified all feature flag combinations
- Cleaned up unused imports
- Added C bindings headers
- Updated README with accurate metrics

**Next Steps:**
- Implement full C bindings with Zig exports
- Add Python bindings
- Add WASM bindings
- Monitor Zig 0.16 API stabilization for native HTTP
