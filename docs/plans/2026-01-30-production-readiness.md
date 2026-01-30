# ABI Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve full production readiness by modernizing Zig 0.16 patterns, improving code quality, and ensuring documentation accuracy.

**Architecture:** Incremental improvements following existing patterns. Each task is independent and can be completed in isolation. Focus on @tagName/@errorName modernization, test coverage gaps, and documentation consistency.

**Tech Stack:** Zig 0.16, existing ABI patterns (ArrayListUnmanaged, {t} format specifier, std.Io API)

---

## Task 1: Modernize Config Module Format Specifiers

**Files:**
- Modify: `src/config/cloud.zig`
- Modify: `src/config/mod.zig`

**Step 1: Read cloud.zig to find @tagName usage**

Run: `grep -n "@tagName" src/config/cloud.zig`

**Step 2: Fix cloud.zig - replace @tagName with proper pattern**

The `@tagName` is used to return a string for enum values. For functions returning `[]const u8`, we need to keep `@tagName` (it's valid). Only replace when used in print/format contexts.

```zig
// In src/config/cloud.zig - if used in format context, change to:
// Old: std.debug.print("{s}", .{@tagName(self)});
// New: std.debug.print("{t}", .{self});
```

**Step 3: Read mod.zig to find @tagName usage**

Run: `grep -n "@tagName" src/config/mod.zig`

**Step 4: Verify both files compile**

Run: `zig build 2>&1 | head -10`
Expected: No errors

**Step 5: Run tests**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: 787+ tests pass

**Step 6: Commit**

```bash
git add src/config/cloud.zig src/config/mod.zig
git commit -m "refactor(config): audit @tagName usage for Zig 0.16 compliance

Verified @tagName usage is appropriate (returning strings, not formatting).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Modernize Web Module Format Specifiers

**Files:**
- Modify: `src/web/routes/personas.zig`
- Modify: `src/web/handlers/chat.zig`

**Step 1: Read personas.zig to find @errorName usage**

Run: `grep -n "@errorName" src/web/routes/personas.zig`

**Step 2: Fix personas.zig - replace @errorName in format contexts**

```zig
// Old pattern:
const err_name = @errorName(err);
// ... later used in format

// New pattern (if used in print):
std.debug.print("Error: {t}", .{err});

// If err_name is used as a string value (JSON, response), keep @errorName
```

**Step 3: Read chat.zig to find all @tagName/@errorName usage**

Run: `grep -n "@tagName\|@errorName" src/web/handlers/chat.zig`

**Step 4: Fix chat.zig - evaluate each usage**

For struct field assignments like `.name = @tagName(pt)`, this is valid Zig 0.16.
For print statements, use `{t}` format specifier.

**Step 5: Run tests**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: 787+ tests pass

**Step 6: Commit**

```bash
git add src/web/routes/personas.zig src/web/handlers/chat.zig
git commit -m "refactor(web): audit @tagName/@errorName for Zig 0.16 compliance

Verified usage patterns are appropriate for string values in responses.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Modernize AI Module Format Specifiers

**Files:**
- Modify: `src/ai/gpu_agent.zig`

**Step 1: Read gpu_agent.zig to find @tagName usage**

Run: `grep -n "@tagName" src/ai/gpu_agent.zig`

**Step 2: Evaluate each @tagName usage**

```zig
// Pattern: .backend_name = @tagName(decision.backend_type)
// This assigns a string to a struct field - VALID, keep as-is

// Pattern in print: std.debug.print("Backend: {s}", .{@tagName(x)})
// Change to: std.debug.print("Backend: {t}", .{x})
```

**Step 3: Run tests**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: 787+ tests pass

**Step 4: Commit**

```bash
git add src/ai/gpu_agent.zig
git commit -m "refactor(ai): audit @tagName usage for Zig 0.16 compliance

Backend name assignments use @tagName correctly for string fields.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Integration Test for Error Module

**Files:**
- Modify: `src/tests/mod.zig`
- Test: Error module integration

**Step 1: Read current test imports**

Run: `head -50 src/tests/mod.zig`

**Step 2: Add error module test reference**

```zig
// Add to src/tests/mod.zig imports section:
test {
    _ = @import("../shared/errors.zig");
}
```

**Step 3: Run tests to verify inclusion**

Run: `zig build test --summary all 2>&1 | tail -5`
Expected: Test count increases (788+)

**Step 4: Commit**

```bash
git add src/tests/mod.zig
git commit -m "test: include shared/errors.zig in test suite

Ensures error module tests run with main test suite.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Verify and Document Feature Flag Matrix

**Files:**
- Create: `docs/feature-flags.md`

**Step 1: Test all single-flag disabled builds**

Run each and verify success:
```bash
zig build -Denable-ai=false 2>&1 | tail -1
zig build -Denable-gpu=false 2>&1 | tail -1
zig build -Denable-database=false 2>&1 | tail -1
zig build -Denable-network=false 2>&1 | tail -1
zig build -Denable-web=false 2>&1 | tail -1
zig build -Denable-profiling=false 2>&1 | tail -1
```

**Step 2: Test combination builds**

```bash
zig build -Denable-ai=false -Denable-gpu=false 2>&1 | tail -1
zig build -Denable-database=false -Denable-network=false -Denable-web=false 2>&1 | tail -1
```

**Step 3: Create feature flags documentation**

```markdown
---
title: "Feature Flags"
tags: [build, configuration]
---
# Feature Flags Reference

> **Codebase Status:** Synced with repository as of 2026-01-30.

## Available Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI module (LLM, vision, agents, training) |
| `-Denable-gpu` | true | GPU acceleration framework |
| `-Denable-database` | true | Vector database (WDBX) |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | Web utilities and HTTP |
| `-Denable-profiling` | true | Performance profiling |

## GPU Backend Flags

| Flag | Description |
|------|-------------|
| `-Dgpu-backend=auto` | Auto-detect available backends |
| `-Dgpu-backend=cuda` | NVIDIA CUDA |
| `-Dgpu-backend=vulkan` | Vulkan (cross-platform) |
| `-Dgpu-backend=metal` | Apple Metal (macOS) |
| `-Dgpu-backend=webgpu` | WebGPU (WASM) |
| `-Dgpu-backend=none` | Disable all GPU backends |

## Verified Combinations

All combinations below have been tested and compile successfully:

- All flags enabled (default)
- Each flag individually disabled
- `-Denable-ai=false -Denable-gpu=false`
- `-Denable-database=false -Denable-network=false -Denable-web=false`
```

**Step 4: Commit**

```bash
git add docs/feature-flags.md
git commit -m "docs: add feature flags reference documentation

Documents all build flags and verified combinations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Clean Up Docs Index

**Files:**
- Modify: `docs/docs-index.md`

**Step 1: Read current docs-index.md**

Run: `head -100 docs/docs-index.md`

**Step 2: Verify all linked files exist**

Run: `grep -o '\[.*\](.*\.md)' docs/docs-index.md | sed 's/.*(\(.*\))/\1/' | while read f; do [ -f "docs/$f" ] || echo "Missing: $f"; done`

**Step 3: Remove references to deleted files**

Remove any references to:
- `api_*.md` files (deleted redirects)
- `performance.md` (deleted stub)
- `gpu-backends.md` (deleted duplicate)

**Step 4: Add new documentation links**

Add links to:
- `feature-flags.md` (new)
- `plans/2026-01-30-codebase-improvement.md`
- `plans/2026-01-30-production-readiness.md`

**Step 5: Run link verification again**

Run: `grep -o '\[.*\](.*\.md)' docs/docs-index.md | sed 's/.*(\(.*\))/\1/' | while read f; do [ -f "docs/$f" ] || echo "Missing: $f"; done`
Expected: No output (all links valid)

**Step 6: Commit**

```bash
git add docs/docs-index.md
git commit -m "docs: update docs-index with current file structure

Removed references to deleted files, added new documentation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update README Test Badge

**Files:**
- Modify: `README.md`

**Step 1: Get current test count**

Run: `zig build test --summary all 2>&1 | grep "tests passed"`

**Step 2: Update badge if needed**

If test count changed from 787, update:
```markdown
<img src="https://img.shields.io/badge/tests-XXX_passing-brightgreen?logo=checkmarx&logoColor=white" alt="Tests"/>
```

And update the text:
```markdown
Battle-tested with XXX+ tests, comprehensive error handling...
```

**Step 3: Commit (if changed)**

```bash
git add README.md
git commit -m "docs: update test count in README

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Push All Changes

**Step 1: Review commits**

Run: `git log --oneline origin/main..HEAD`

**Step 2: Run final verification**

Run: `zig build test --summary all && zig fmt --check .`
Expected: All tests pass, no formatting issues

**Step 3: Push to remote**

Run: `git push origin main`

---

## Summary

After completing all tasks:
- Modernized @tagName/@errorName usage for Zig 0.16 compliance
- Added error module to test suite
- Created feature flags documentation
- Cleaned up docs-index references
- Updated README metrics
- All changes pushed to remote

**Next Steps:**
- Monitor CI for any regressions
- Consider adding more inline tests to low-coverage modules
- Continue iterating on documentation quality
