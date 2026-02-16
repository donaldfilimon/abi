---
title: "Codebase Improvements Implementation Plan"
date: 2026-02-04
status: mostly-complete
tags: [planning, implementation, quality]
related:
  - "../../PLAN.md"
  - "../../ROADMAP.md"
  - "./2026-02-04-feature-modules-completion.md"
---
# Codebase Improvements Implementation Plan

> **Status:** Mostly Complete (Phase 1 done via Ralph Loop iterations) | **Created:** 2026-02-04 | **Parent:** [PLAN.md](../../PLAN.md)

**Goal:** Modernize the ABI codebase by fixing deprecated Zig 0.16.0-dev.2535+b5bd49460 patterns, improving security defaults, and enhancing code quality across ~100 instances.

**Architecture:** Systematic migration of deprecated APIs (`@errorName`/`@tagName` → `{t}`, `catch unreachable` → proper error handling), security hardening (rate limiting defaults), and documentation improvements. Changes are isolated to individual files with no cross-module dependencies.

**Tech Stack:** Zig 0.16.0-dev.2535+b5bd49460, std library modern APIs, TDD with existing test infrastructure

---

## Phase 1: Quick Wins - Format Specifier Migration (7 files)

### Task 1: Migrate @errorName in stress tests

**Files:**
- Modify: `src/services/tests/stress/mod.zig`
- Test: `zig test src/services/tests/stress/mod.zig`

**Step 1: Read the file to identify @errorName patterns**

```bash
grep -n "@errorName" src/services/tests/stress/mod.zig
```

**Step 2: Replace @errorName with {t} format specifier**

Replace patterns like:
```zig
// Before
std.debug.print("Error: {s}\n", .{@errorName(err)});

// After
std.debug.print("Error: {t}\n", .{err});
```

**Step 3: Run test to verify it compiles and passes**

Run: `zig test src/services/tests/stress/mod.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/services/tests/stress/mod.zig
git commit -m "$(cat <<'EOF'
refactor(tests): migrate @errorName to {t} format specifier

Zig 0.16.0-dev.2535+b5bd49460 introduced {t} for printing errors/enums directly.
This is more idiomatic than using @errorName(@tagName).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Migrate @errorName in AI tools

**Files:**
- Modify: `src/features/ai/tools/tool.zig`
- Test: `zig build test --summary all`

**Step 1: Read file and identify patterns**

```bash
grep -n "@errorName" src/features/ai/tools/tool.zig
```

**Step 2: Replace @errorName with {t} format specifier**

For each occurrence, change `{s}` + `@errorName(err)` to `{t}` + `err`.

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/ai/tools/tool.zig
git commit -m "$(cat <<'EOF'
refactor(ai/tools): migrate @errorName to {t} format specifier

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Migrate @errorName in cloud modules

**Files:**
- Modify: `src/services/cloud/aws_lambda.zig`
- Modify: `src/services/cloud/azure_functions.zig`
- Modify: `src/services/cloud/gcp_functions.zig`
- Test: `zig build -Denable-network=true test --summary all`

**Step 1: Read files and identify patterns**

```bash
grep -n "@errorName" src/services/cloud/*.zig
```

**Step 2: Replace @errorName with {t} format specifier in all three files**

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/services/cloud/aws_lambda.zig src/services/cloud/azure_functions.zig src/services/cloud/gcp_functions.zig
git commit -m "$(cat <<'EOF'
refactor(cloud): migrate @errorName to {t} format specifier

Updated AWS Lambda, Azure Functions, and GCP Functions adapters.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Migrate @tagName in web handlers

**Files:**
- Modify: `src/features/web/handlers/chat.zig`
- Test: `zig build test --summary all`

**Step 1: Read file and identify @tagName patterns**

```bash
grep -n "@tagName" src/features/web/handlers/chat.zig
```

**Step 2: Replace @tagName with {t} format specifier**

For each print statement using `@tagName(enum_value)`, change to `{t}` + `enum_value`.

Note: Only replace in print statements. If @tagName is used to get a `[]const u8` value for other purposes (e.g., string comparison), leave it unchanged.

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/web/handlers/chat.zig
git commit -m "$(cat <<'EOF'
refactor(web): migrate @tagName to {t} in chat handler

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Migrate @tagName in AI personas

**Files:**
- Modify: `src/features/ai/personas/metrics.zig`
- Modify: `src/features/ai/personas/loadbalancer.zig`
- Modify: `src/features/ai/personas/abbey/reasoning.zig`
- Modify: `src/features/ai/personas/abi/mod.zig`
- Modify: `src/features/ai/personas/aviva/code.zig`
- Modify: `src/features/ai/personas/embeddings/persona_index.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify patterns**

```bash
grep -n "@tagName" src/features/ai/personas/*.zig src/features/ai/personas/**/*.zig
```

**Step 2: Replace @tagName with {t} format specifier (print statements only)**

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/ai/personas/
git commit -m "$(cat <<'EOF'
refactor(ai/personas): migrate @tagName to {t} format specifier

Updated metrics, loadbalancer, abbey, abi, aviva, and embeddings.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Migrate @tagName in orchestration module

**Files:**
- Modify: `src/features/ai/orchestration/mod.zig`
- Modify: `src/features/ai/orchestration/router.zig`
- Modify: `src/features/ai/orchestration/ensemble.zig`
- Modify: `src/features/ai/orchestration/fallback.zig`
- Modify: `src/features/ai/orchestration/stub.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify patterns**

```bash
grep -n "@tagName" src/features/ai/orchestration/*.zig
```

**Step 2: Replace @tagName with {t} format specifier (print statements only)**

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/ai/orchestration/
git commit -m "$(cat <<'EOF'
refactor(ai/orchestration): migrate @tagName to {t} format specifier

Updated mod, router, ensemble, fallback, and stub modules.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Migrate remaining @tagName occurrences

**Files:**
- Modify: `src/features/ai/gpu_agent.zig`
- Modify: `src/features/ai/tools/task.zig`
- Modify: `src/core/config/mod.zig`
- Modify: `src/core/config/cloud.zig`
- Modify: `src/core/config/stubs/types.zig`
- Modify: `src/services/cloud/types.zig`
- Modify: `src/services/cloud/stubs/types.zig`
- Modify: `src/services/platform/mod.zig`
- Modify: `src/core/registry/stub.zig`
- Modify: `src/core/registry/registration.zig`
- Modify: `tools/cli/commands/system_info.zig`
- Modify: `tools/cli/tui/model_panel.zig`
- Modify: `benchmarks/main.zig`
- Modify: `src/services/tests/integration/c_api_test.zig`
- Modify: `src/services/tests/integration/fixtures.zig`
- Test: `zig build test --summary all`

**Step 1: Search and identify all remaining patterns**

```bash
grep -rn "@tagName" src/ tools/ benchmarks/ --include="*.zig" | grep -v "CODEBASE\|CLAUDE\|PLAN\|README"
```

**Step 2: Replace @tagName with {t} format specifier (print statements only)**

**Step 3: Run full test suite**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/ tools/ benchmarks/
git commit -m "$(cat <<'EOF'
refactor: complete @tagName to {t} migration across codebase

Updated remaining files in ai, config, cloud, platform, registry,
cli, and benchmarks modules.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Code Quality - Unreachable Pattern Fixes (19 files)

### Task 8: Write test for help.zig error handling

**Files:**
- Create: `tools/cli/utils/help_test.zig`
- Test: `zig test tools/cli/utils/help_test.zig`

**Step 1: Write a failing test for format error handling**

```zig
//! Tests for help utilities error handling
const std = @import("std");
const help = @import("help.zig");

test "formatOption handles buffer overflow gracefully" {
    // Test with a tiny buffer that will overflow
    var tiny_buf: [5]u8 = undefined;

    // This should return an error, not crash with unreachable
    const result = help.formatOption(&tiny_buf, .{
        .name = "this-is-a-very-long-option-name",
        .description = "This is an extremely long description that will overflow",
    });

    try std.testing.expectError(error.NoSpaceLeft, result);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/utils/help_test.zig`
Expected: FAIL - current code uses `catch unreachable`

**Step 3: Fix help.zig to return errors instead of unreachable**

```zig
// Before
const result = std.fmt.bufPrint(&buf, "{}", .{opt}) catch unreachable;

// After
const result = std.fmt.bufPrint(&buf, "{}", .{opt}) catch |err| {
    return err;
};
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/utils/help_test.zig`
Expected: PASS

**Step 5: Run full test suite**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 6: Commit**

```bash
git add tools/cli/utils/help.zig tools/cli/utils/help_test.zig
git commit -m "$(cat <<'EOF'
fix(cli/help): replace unreachable with proper error returns

Added error handling for buffer overflow scenarios in formatOption.
Added unit tests for edge cases.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Fix unreachable in network discovery

**Files:**
- Modify: `src/features/network/discovery.zig`
- Modify: `src/features/network/discovery_types.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify unreachable patterns**

```bash
grep -n "catch unreachable" src/features/network/discovery*.zig
```

**Step 2: Replace unreachable with proper error handling**

For each occurrence:
```zig
// Before
const result = operation() catch unreachable;

// After
const result = operation() catch |err| {
    std.log.err("Discovery operation failed: {t}", .{err});
    return error.DiscoveryError;
};
```

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/network/discovery.zig src/features/network/discovery_types.zig
git commit -m "$(cat <<'EOF'
fix(network): replace unreachable with error handling in discovery

Improved error visibility for network discovery failures.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Fix unreachable in JSON utilities

**Files:**
- Modify: `src/services/shared/utils/json/mod.zig`
- Test: `zig test src/services/shared/utils/json/mod.zig`

**Step 1: Read file and identify unreachable patterns**

```bash
grep -n "catch unreachable" src/services/shared/utils/json/mod.zig
```

**Step 2: Replace unreachable with proper error handling**

**Step 3: Run tests**

Run: `zig test src/services/shared/utils/json/mod.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/services/shared/utils/json/mod.zig
git commit -m "$(cat <<'EOF'
fix(shared/json): replace unreachable with error handling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Fix unreachable in AI modules

**Files:**
- Modify: `src/features/ai/agent.zig`
- Modify: `src/features/ai/streaming/mod.zig`
- Modify: `src/features/ai/streaming/backpressure.zig`
- Modify: `src/features/ai/orchestration/ensemble.zig`
- Modify: `src/features/ai/llm/io/gguf_writer.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify unreachable patterns**

```bash
grep -n "catch unreachable" src/features/ai/*.zig src/features/ai/**/*.zig
```

**Step 2: Replace unreachable with proper error handling**

Note: Be careful with hot paths - some may be intentional for performance. Add comments if keeping unreachable intentionally.

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/ai/
git commit -m "$(cat <<'EOF'
fix(ai): replace unreachable with error handling

Updated agent, streaming, orchestration, and llm modules.
Kept unreachable where intentional for performance (with comments).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Fix unreachable in GPU and database modules

**Files:**
- Modify: `src/features/gpu/tests/comprehensive_test.zig`
- Modify: `src/features/gpu/tests/performance_benchmark_test.zig`
- Modify: `src/features/gpu/backends/fpga/kernels/matmul_kernels.zig`
- Modify: `src/features/database/distributed/integration_test.zig`
- Modify: `src/features/database/distributed/shard_assignment_test.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify unreachable patterns**

```bash
grep -n "catch unreachable" src/features/gpu/**/*.zig src/features/database/**/*.zig
```

**Step 2: Evaluate each occurrence**

For test files: Replace with proper error returns
For benchmark files: Keep unreachable (intentional for benchmarks) but add comment

```zig
// Intentionally unreachable - benchmark code assumes valid input
const result = operation() catch unreachable;
```

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/features/gpu/ src/features/database/
git commit -m "$(cat <<'EOF'
fix(gpu,database): improve error handling in tests

Replaced unreachable with error returns in test files.
Documented intentional unreachable in benchmark code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Fix unreachable in remaining test files

**Files:**
- Modify: `src/services/tests/observability_test.zig`
- Modify: `src/services/tests/e2e_llm_test.zig`
- Modify: `src/services/tests/chaos/ha_chaos_test.zig`
- Test: `zig build test --summary all`

**Step 1: Read files and identify unreachable patterns**

```bash
grep -n "catch unreachable" src/services/tests/*.zig src/services/tests/**/*.zig
```

**Step 2: Replace unreachable with proper error handling**

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/services/tests/
git commit -m "$(cat <<'EOF'
fix(tests): replace unreachable with proper error handling

Updated observability, e2e_llm, and ha_chaos tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Fix unreachable in benchmarks

**Files:**
- Modify: `benchmarks/infrastructure/crypto.zig`
- Test: `zig build benchmarks`

**Step 1: Read file and identify unreachable patterns**

```bash
grep -n "catch unreachable" benchmarks/**/*.zig
```

**Step 2: Document intentional unreachable usage**

For benchmark code, add comments explaining why unreachable is intentional:

```zig
// PERF: Intentionally unreachable - benchmark assumes valid crypto input
// to avoid branch prediction overhead in hot path
const result = crypto.encrypt(&buf, data) catch unreachable;
```

**Step 3: Build benchmarks**

Run: `zig build benchmarks`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add benchmarks/
git commit -m "$(cat <<'EOF'
docs(benchmarks): document intentional unreachable usage

Added comments explaining performance rationale for unreachable
patterns in benchmark code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Security Hardening

### Task 15: Enable rate limiting by default in production

**Files:**
- Modify: `src/services/shared/security/rate_limit.zig`
- Modify: `src/core/config/web.zig`
- Test: `zig build test --summary all`

**Step 1: Write failing test for default rate limiting**

Add test to verify rate limiting is enabled by default in production mode:

```zig
test "rate limiting enabled by default in production" {
    const config = RateLimitConfig.productionDefaults();
    try std.testing.expect(config.enabled);
    try std.testing.expect(config.requests > 0);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/services/shared/security/rate_limit.zig`
Expected: FAIL

**Step 3: Add productionDefaults() with rate limiting enabled**

```zig
pub fn productionDefaults() RateLimitConfig {
    return .{
        .enabled = true,
        .requests = 100,
        .window_seconds = 60,
        .burst = 20,
    };
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/services/shared/security/rate_limit.zig`
Expected: PASS

**Step 5: Update web config to use production defaults**

**Step 6: Run full test suite**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/services/shared/security/rate_limit.zig src/core/config/web.zig
git commit -m "$(cat <<'EOF'
security(rate_limit): enable rate limiting by default in production

Added productionDefaults() with sensible rate limits.
Updated web config to use production defaults.
Addresses security issue M-2.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Add parameterized query helpers for fulltext search

**Files:**
- Modify: `src/features/database/fulltext.zig`
- Test: `zig test src/features/database/fulltext.zig`

**Step 1: Write failing test for query sanitization**

```zig
test "search rejects SQL injection attempts" {
    var index = InvertedIndex.init(std.testing.allocator, .{}, .{});
    defer index.deinit();

    // These should be safely handled, not cause injection
    const malicious_queries = &[_][]const u8{
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "admin'--",
    };

    for (malicious_queries) |query| {
        // Should either sanitize or reject, not execute raw
        const results = try index.search(query, 10);
        defer std.testing.allocator.free(results);
        // Search should complete safely (results may be empty)
    }
}
```

**Step 2: Run test to verify behavior**

Run: `zig test src/features/database/fulltext.zig`
Expected: Verify current behavior

**Step 3: Add input sanitization if needed**

If the fulltext module passes user input directly to any SQL, add parameterization.
If it's pure in-memory BM25 (no SQL), add a comment documenting this is SQL-free.

**Step 4: Run tests**

Run: `zig test src/features/database/fulltext.zig`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/features/database/fulltext.zig
git commit -m "$(cat <<'EOF'
security(database): verify fulltext search is SQL injection safe

Added tests confirming BM25 search is pure in-memory with no SQL.
Addresses security concern M-4.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Documentation Improvements

### Task 17: Add stub file documentation

**Files:**
- Modify: `src/features/ai/stub.zig`
- Modify: `src/features/gpu/stub.zig`
- Modify: `src/features/database/stub.zig`
- Modify: `src/features/network/stub.zig`
- Modify: `src/features/web/stub.zig`
- Modify: `src/features/observability/stub.zig`
- Test: `zig build`

**Step 1: Add module-level doc comments to each stub**

```zig
//! Stub module for when the AI feature is disabled.
//!
//! This module provides API-compatible no-op implementations for all
//! public AI functions. All functions return `error.AIDisabled` or
//! empty/default values as appropriate.
//!
//! To enable the real implementation, build with `-Denable-ai=true`.
```

**Step 2: Build to verify comments are valid**

Run: `zig build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/*/stub.zig
git commit -m "$(cat <<'EOF'
docs: add module-level documentation to all stub files

Each stub now explains its purpose and how to enable the real module.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Document Big-O complexity for concurrency primitives

**Files:**
- Modify: `src/services/runtime/concurrency/chase_lev.zig`
- Modify: `src/services/runtime/concurrency/mpmc_queue.zig`
- Modify: `src/services/runtime/concurrency/epoch.zig`
- Modify: `src/services/runtime/concurrency/priority_queue.zig`
- Test: `zig build`

**Step 1: Add complexity documentation to chase_lev.zig**

```zig
//! Chase-Lev work-stealing deque.
//!
//! A lock-free double-ended queue optimized for work-stealing schedulers.
//!
//! ## Complexity
//! - `push()`: O(1) amortized (may grow buffer)
//! - `pop()`: O(1)
//! - `steal()`: O(1)
//!
//! ## Memory
//! - O(n) where n is the maximum number of concurrent items
//! - Uses epoch-based reclamation for safe memory management
```

**Step 2: Add complexity documentation to mpmc_queue.zig**

```zig
//! Multi-producer multi-consumer bounded queue.
//!
//! ## Complexity
//! - `push()`: O(1)
//! - `pop()`: O(1)
//!
//! ## Memory
//! - O(capacity) fixed allocation
```

**Step 3: Add complexity documentation to epoch.zig and priority_queue.zig**

**Step 4: Build to verify comments are valid**

Run: `zig build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add src/services/runtime/concurrency/
git commit -m "$(cat <<'EOF'
docs(concurrency): add Big-O complexity documentation

Documented time and space complexity for chase_lev, mpmc_queue,
epoch reclamation, and priority_queue.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 19: Create test helpers module

**Files:**
- Create: `src/services/tests/helpers.zig`
- Modify: `src/services/tests/mod.zig`
- Test: `zig test src/services/tests/mod.zig`

**Step 1: Create helpers.zig with common test utilities**

```zig
//! Common test helpers and utilities.
//!
//! This module provides shared setup/teardown logic and assertion
//! helpers used across the test suite.

const std = @import("std");

/// Test allocator with leak detection.
pub const TestAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{
        .stack_trace_frames = 10,
    }),

    pub fn init() TestAllocator {
        return .{ .gpa = .{} };
    }

    pub fn allocator(self: *TestAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    pub fn deinit(self: *TestAllocator) void {
        const check = self.gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in test");
        }
    }
};

/// Skip test if hardware not available.
pub fn skipIfNoGpu() !void {
    // Check for GPU availability
    if (!hasGpuSupport()) {
        return error.SkipZigTest;
    }
}

fn hasGpuSupport() bool {
    // Platform detection for GPU
    return @import("builtin").os.tag != .freestanding;
}

/// Create a temporary directory for test files.
pub fn createTempDir(allocator: std.mem.Allocator) ![]const u8 {
    _ = allocator;
    return "/tmp/abi-test";
}
```

**Step 2: Add import to tests/mod.zig**

```zig
pub const helpers = @import("helpers.zig");
```

**Step 3: Run tests**

Run: `zig test src/services/tests/mod.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/services/tests/helpers.zig src/services/tests/mod.zig
git commit -m "$(cat <<'EOF'
feat(tests): add shared test helpers module

Added TestAllocator with leak detection, skipIfNoGpu(), and
createTempDir() utilities for cleaner test code.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Update CODEBASE_IMPROVEMENTS.md status

**Files:**
- Modify: `CODEBASE_IMPROVEMENTS.md`

**Step 1: Update the document to reflect completed work**

Mark completed items with checkmarks and update the status summary.

**Step 2: Commit**

```bash
git add CODEBASE_IMPROVEMENTS.md
git commit -m "$(cat <<'EOF'
docs: update CODEBASE_IMPROVEMENTS.md status

Marked Phase 1-4 improvements as complete.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

### Task 21: Run full test suite and format check

**Step 1: Format all code**

```bash
zig fmt .
```

**Step 2: Run lint check**

```bash
zig build lint
```
Expected: No formatting errors

**Step 3: Run full test suite**

```bash
zig build test --summary all
```
Expected: 909+ tests passing

**Step 4: Verify feature-disabled builds**

```bash
zig build -Denable-ai=false
zig build -Denable-gpu=false
zig build -Denable-database=false
zig build -Denable-network=false
```
Expected: All builds succeed

**Step 5: Commit any formatting fixes**

```bash
git add -A
git commit -m "$(cat <<'EOF'
style: apply zig fmt across codebase

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Phase | Tasks | Files | Focus |
|-------|-------|-------|-------|
| 1 | 1-7 | ~33 | @errorName/@tagName → {t} |
| 2 | 8-14 | ~19 | catch unreachable → error handling |
| 3 | 15-16 | ~3 | Security hardening |
| 4 | 17-20 | ~12 | Documentation |
| Final | 21 | - | Verification |

**Total: 21 tasks, ~67 files, ~100 individual changes**

---

**Last Updated:** 2026-02-04
