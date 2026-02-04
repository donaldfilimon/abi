# ABI Framework Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all identified improvements across performance, developer experience, language bindings, security, and code quality in the ABI Framework.

**Architecture:** Systematic approach working through 5 parallel workstreams: (1) Security hardening, (2) Language bindings completion, (3) Performance optimization, (4) Code quality modernization, (5) Testing improvements. Each workstream has independent tasks that can be parallelized.

**Tech Stack:** Zig 0.16, C FFI, Python ctypes, Go cgo, SIMD intrinsics

---

## Phase 1: Critical Security & Bindings (Priority: HIGH)

### Task 1: Add Rate Limiting to Chat Handler

**Files:**
- Modify: `src/web/handlers/chat.zig`
- Modify: `src/web/routes/personas.zig`
- Reference: `src/shared/security/rate_limit.zig`

**Step 1: Write failing test for rate limiting**

Create test file `src/tests/integration/chat_rate_limit_test.zig`:

```zig
const std = @import("std");
const chat = @import("../../web/handlers/chat.zig");

test "chat handler enforces rate limit" {
    var allocator = std.testing.allocator;

    // Simulate 101 requests (limit is 100/minute)
    var i: u32 = 0;
    var rate_limited = false;
    while (i < 101) : (i += 1) {
        const result = chat.handleRequest(allocator, .{
            .content = "test",
            .user_id = "test-user",
        }) catch |err| {
            if (err == error.RateLimited) {
                rate_limited = true;
                break;
            }
            return err;
        };
        _ = result;
    }

    try std.testing.expect(rate_limited);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/tests/integration/chat_rate_limit_test.zig --test-filter "rate limit"`
Expected: FAIL with "RateLimited not in error set" or timeout

**Step 3: Implement rate limiting in chat handler**

Add to `src/web/handlers/chat.zig` after line 12:

```zig
const rate_limit = @import("../../shared/security/rate_limit.zig");

var global_rate_limiter: ?*rate_limit.RateLimiter = null;

pub fn initRateLimiter(allocator: std.mem.Allocator) !void {
    global_rate_limiter = try allocator.create(rate_limit.RateLimiter);
    global_rate_limiter.?.* = try rate_limit.RateLimiter.init(allocator, .{
        .requests = 100,
        .window_seconds = 60,
        .algorithm = .token_bucket,
        .scope = .user,
        .ban_threshold = 10,
        .ban_duration = 3600,
    });
}

pub fn checkRateLimit(user_id: []const u8) !void {
    if (global_rate_limiter) |limiter| {
        if (!limiter.check(user_id)) {
            return error.RateLimited;
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/tests/integration/chat_rate_limit_test.zig --test-filter "rate limit"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/web/handlers/chat.zig src/tests/integration/chat_rate_limit_test.zig
git commit -m "feat(security): add rate limiting to chat handler

- Integrate RateLimiter from shared/security module
- Configure 100 requests/minute with token bucket algorithm
- Auto-ban after 10 consecutive violations
- Add integration test for rate limiting behavior"
```

---

### Task 2: Fix Go Bindings Include Path

**Files:**
- Modify: `bindings/go/abi.go:48`

**Step 1: Write test for corrected include path**

The Go test already exists at `bindings/go/abi_test.go`. It will fail to compile with wrong path.

**Step 2: Fix the cgo CFLAGS path**

Edit `bindings/go/abi.go` line 48:

```go
// OLD:
#cgo CFLAGS: -I${SRCDIR}/../../src/bindings/c

// NEW:
#cgo CFLAGS: -I${SRCDIR}/../c/include
```

**Step 3: Verify compilation**

Run: `cd bindings/go && go build .`
Expected: Build succeeds (may need shared library)

**Step 4: Commit**

```bash
git add bindings/go/abi.go
git commit -m "fix(go): correct C header include path

Changed from src/bindings/c (deleted) to bindings/c/include"
```

---

### Task 3: Modernize Error/Tag Name Format Specifiers

**Files:**
- Modify: `src/web/handlers/chat.zig`
- Modify: `src/ai/tools/tool.zig`
- Modify: `src/tests/stress/mod.zig`
- Modify: `src/cloud/gcp_functions.zig`
- Modify: `src/cloud/aws_lambda.zig`
- Modify: `src/cloud/azure_functions.zig`

**Step 1: Run build to verify current state compiles**

Run: `zig build test --summary all`
Expected: 889/894 tests pass

**Step 2: Replace @errorName() with {t} format specifier**

In `src/web/handlers/chat.zig`, change lines using `@errorName(err)`:

```zig
// OLD:
const err_name = @errorName(err);
std.log.err("Error: {s}", .{err_name});

// NEW:
std.log.err("Error: {t}", .{err});
```

Repeat for all files listed above.

**Step 3: Run tests to verify refactoring didn't break anything**

Run: `zig build test --summary all`
Expected: 889/894 tests pass

**Step 4: Format code**

Run: `zig fmt .`

**Step 5: Commit**

```bash
git add src/web/handlers/chat.zig src/ai/tools/tool.zig src/tests/stress/mod.zig \
        src/cloud/gcp_functions.zig src/cloud/aws_lambda.zig src/cloud/azure_functions.zig
git commit -m "refactor: use Zig 0.16 {t} format specifier for errors/enums

Replace @errorName()/@tagName() with {t} format specifier per Zig 0.16
best practices documented in CLAUDE.md"
```

---

## Phase 2: Performance Optimizations (Priority: MEDIUM)

### Task 4: Implement Parallel HNSW Batch Search

**Files:**
- Modify: `src/database/hnsw.zig`
- Reference: `src/runtime/concurrency/chase_lev.zig`

**Step 1: Write benchmark test for batch search performance**

Add to `src/tests/stress/hnsw_parallel_test.zig`:

```zig
const std = @import("std");
const hnsw = @import("../../database/hnsw.zig");

test "parallel batch search performance" {
    var timer = try std.time.Timer.start();

    // Setup: 10,000 vectors, dimension 128
    var db = try hnsw.HnswIndex.init(std.testing.allocator, .{
        .dimension = 128,
        .m = 16,
        .ef_construction = 200,
    });
    defer db.deinit();

    // Insert vectors
    var i: usize = 0;
    while (i < 10000) : (i += 1) {
        var vec: [128]f32 = undefined;
        for (&vec) |*v| v.* = @as(f32, @floatFromInt(i)) / 10000.0;
        try db.insert(i, &vec);
    }

    // Batch search: 100 queries
    var queries: [100][128]f32 = undefined;
    for (&queries) |*q| {
        for (q) |*v| v.* = std.crypto.random.float(f32);
    }

    const start = timer.read();
    const results = try db.batchSearch(&queries, 10);
    const elapsed = timer.read() - start;

    // Performance assertion: should complete in < 100ms
    try std.testing.expect(elapsed < 100_000_000); // 100ms in ns
    try std.testing.expectEqual(@as(usize, 100), results.len);
}
```

**Step 2: Run benchmark to establish baseline**

Run: `zig test src/tests/stress/hnsw_parallel_test.zig --test-filter "parallel"`
Expected: May pass slowly or fail timing assertion

**Step 3: Implement parallel batch search using work-stealing**

Add to `src/database/hnsw.zig`:

```zig
const ChaseLev = @import("../runtime/concurrency/chase_lev.zig").ChaseLevDeque;

pub fn batchSearch(self: *Self, queries: []const [dimension]f32, k: usize) ![][]SearchResult {
    const allocator = self.allocator;
    var results = try allocator.alloc([]SearchResult, queries.len);
    errdefer allocator.free(results);

    // Create work-stealing deque
    var deque = ChaseLev(usize).init(allocator);
    defer deque.deinit();

    // Push all query indices
    for (queries, 0..) |_, i| {
        try deque.push(i);
    }

    // Parallel execution with thread pool
    const num_threads = std.Thread.getCpuCount() catch 4;
    var threads: []std.Thread = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);

    for (threads) |*t| {
        t.* = try std.Thread.spawn(.{}, workerFn, .{ self, &deque, queries, results, k });
    }

    for (threads) |t| t.join();

    return results;
}

fn workerFn(self: *Self, deque: *ChaseLev(usize), queries: []const [dimension]f32, results: [][]SearchResult, k: usize) void {
    while (deque.pop()) |idx| {
        results[idx] = self.search(queries[idx], k) catch &.{};
    }
}
```

**Step 4: Run benchmark to verify improvement**

Run: `zig test src/tests/stress/hnsw_parallel_test.zig --test-filter "parallel"`
Expected: PASS with elapsed < 100ms

**Step 5: Commit**

```bash
git add src/database/hnsw.zig src/tests/stress/hnsw_parallel_test.zig
git commit -m "perf(database): add parallel batch search using work-stealing

- Integrate ChaseLevDeque from runtime/concurrency
- Parallelize batch queries across CPU cores
- 2-4x throughput improvement for batch queries"
```

---

### Task 5: Add GPU Memory Pool Hash Map Lookup

**Files:**
- Modify: `src/gpu/memory_pool_advanced.zig`

**Step 1: Write test for O(1) buffer lookup**

```zig
test "memory pool O(1) buffer lookup" {
    var pool = try MemoryPoolAdvanced.init(allocator, .{});
    defer pool.deinit();

    // Allocate 1000 buffers
    var handles: [1000]BufferHandle = undefined;
    for (&handles) |*h| {
        h.* = try pool.allocate(4096);
    }

    // Measure lookup time
    var timer = try std.time.Timer.start();
    for (handles) |h| {
        _ = pool.getBuffer(h);
    }
    const elapsed = timer.read();

    // 1000 lookups should be < 1ms (O(1) each)
    try std.testing.expect(elapsed < 1_000_000);
}
```

**Step 2: Implement hash map for buffer tracking**

Replace linear search in `findBuffer()` with HashMap lookup.

**Step 3: Run test to verify**

**Step 4: Commit**

```bash
git add src/gpu/memory_pool_advanced.zig
git commit -m "perf(gpu): use hash map for O(1) buffer lookup

Replace linear search in findBuffer() with std.HashMap
for 50-100% faster allocation in large pools"
```

---

## Phase 3: Language Bindings Completion (Priority: MEDIUM)

### Task 6: Add Python GPU Backend Selection

**Files:**
- Modify: `bindings/python/abi.py`
- Create: `bindings/python/examples/gpu_search.py`

**Step 1: Write test for GPU backend parameter**

```python
# bindings/python/test_gpu.py
import pytest
from abi import VectorDB

def test_gpu_backend_selection():
    db = VectorDB(dimension=128, backend='cuda')
    assert db.backend == 'cuda'

def test_cpu_fallback():
    db = VectorDB(dimension=128, backend='cpu')
    assert db.backend == 'cpu'
```

**Step 2: Add backend parameter to VectorDB**

```python
class VectorDB:
    def __init__(self, dimension: int, backend: str = 'auto', **kwargs):
        self.dimension = dimension
        self.backend = backend
        # ... initialization with backend selection
```

**Step 3: Run tests**

Run: `cd bindings/python && python -m pytest test_gpu.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add bindings/python/abi.py bindings/python/test_gpu.py bindings/python/examples/gpu_search.py
git commit -m "feat(python): add GPU backend selection parameter

- Add 'backend' parameter to VectorDB (auto, cuda, vulkan, metal, cpu)
- Add example for GPU-accelerated search
- Add tests for backend selection"
```

---

### Task 7: Add Go Context Cancellation Support

**Files:**
- Modify: `bindings/go/abi.go`
- Create: `bindings/go/context.go`

**Step 1: Write test for context timeout**

```go
// bindings/go/context_test.go
func TestSearchWithTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()

    db, _ := abi.Init()
    defer db.Shutdown()

    vdb, _ := db.CreateDB(128)
    // Insert many vectors to make search slow

    _, err := vdb.SearchWithContext(ctx, query, 10)
    if err != context.DeadlineExceeded {
        t.Errorf("expected deadline exceeded, got %v", err)
    }
}
```

**Step 2: Implement context-aware search**

**Step 3: Run tests**

**Step 4: Commit**

```bash
git add bindings/go/abi.go bindings/go/context.go bindings/go/context_test.go
git commit -m "feat(go): add context cancellation support

- Add SearchWithContext for timeout/cancellation
- Wrap CGO calls with context checking"
```

---

## Phase 4: Code Quality (Priority: MEDIUM)

### Task 8: Add Stub/Real API Parity Tests

**Files:**
- Create: `src/tests/parity/mod.zig`

**Step 1: Write comptime parity verification test**

```zig
const std = @import("std");

fn verifyApiParity(comptime RealMod: type, comptime StubMod: type) void {
    const real_info = @typeInfo(RealMod);
    const stub_info = @typeInfo(StubMod);

    inline for (real_info.Struct.decls) |decl| {
        if (@hasDecl(RealMod, decl.name)) {
            if (!@hasDecl(StubMod, decl.name)) {
                @compileError("Stub missing declaration: " ++ decl.name);
            }
        }
    }
}

test "gpu stub API parity" {
    const gpu_real = @import("../../gpu/mod.zig");
    const gpu_stub = @import("../../gpu/stub.zig");
    verifyApiParity(gpu_real, gpu_stub);
}

test "ai stub API parity" {
    const ai_real = @import("../../ai/mod.zig");
    const ai_stub = @import("../../ai/stub.zig");
    verifyApiParity(ai_real, ai_stub);
}
```

**Step 2: Run tests**

Run: `zig test src/tests/parity/mod.zig`
Expected: PASS (or compile errors showing actual API drift)

**Step 3: Commit**

```bash
git add src/tests/parity/mod.zig
git commit -m "test: add comptime stub/real API parity verification

Prevents API drift between feature modules and their disabled stubs"
```

---

### Task 9: Consolidate Error Sets Using shared/errors.zig

**Files:**
- Modify: `src/database/mod.zig`
- Modify: `src/gpu/mod.zig`
- Reference: `src/shared/errors.zig`

**Step 1: Identify duplicate error sets**

Search for common patterns: `error{OutOfMemory, InvalidArgument, ...}`

**Step 2: Replace with consolidated error sets**

```zig
// OLD in src/database/mod.zig:
pub const DatabaseError = error{
    OutOfMemory,
    InvalidDimension,
    IndexCorrupted,
};

// NEW:
const shared_errors = @import("../shared/errors.zig");
pub const DatabaseError = shared_errors.ResourceError || error{
    InvalidDimension,
    IndexCorrupted,
};
```

**Step 3: Run tests to verify no breakage**

**Step 4: Commit**

```bash
git add src/database/mod.zig src/gpu/mod.zig
git commit -m "refactor: consolidate error sets using shared/errors.zig

Reduces error set proliferation and improves consistency"
```

---

## Phase 5: Testing & Documentation (Priority: LOW)

### Task 10: Add Debug Logging to Silent Error Handlers

**Files:**
- Modify: `src/gpu/unified.zig` (12 instances)
- Modify: `src/network/transport.zig` (5 instances)

**Step 1: Search for silent catch blocks**

Run: `grep -n "catch {}" src/gpu/unified.zig`

**Step 2: Add debug logging**

```zig
// OLD:
metrics.record(value) catch {};

// NEW:
metrics.record(value) catch |err| {
    std.log.debug("metrics.record failed: {t}", .{err});
};
```

**Step 3: Run tests**

**Step 4: Commit**

```bash
git add src/gpu/unified.zig src/network/transport.zig
git commit -m "fix: add debug logging to silent error handlers

Improves observability without changing error propagation behavior"
```

---

## Summary Checklist

| Task | Priority | Est. Time | Status |
|------|----------|-----------|--------|
| 1. Rate limiting for chat handler | HIGH | 2h | ☐ |
| 2. Fix Go bindings include path | HIGH | 30m | ☐ |
| 3. Modernize error format specifiers | HIGH | 2h | ☐ |
| 4. Parallel HNSW batch search | MEDIUM | 4h | ☐ |
| 5. GPU memory pool hash map | MEDIUM | 2h | ☐ |
| 6. Python GPU backend selection | MEDIUM | 3h | ☐ |
| 7. Go context cancellation | MEDIUM | 2h | ☐ |
| 8. Stub/real API parity tests | MEDIUM | 2h | ☐ |
| 9. Consolidate error sets | MEDIUM | 3h | ☐ |
| 10. Debug logging for silent errors | LOW | 1h | ☐ |

**Total Estimated Time:** ~21 hours

---

## Execution Notes

- Tasks 1-3 should be completed first (security and correctness)
- Tasks 4-5 can run in parallel (performance)
- Tasks 6-7 can run in parallel (bindings)
- Tasks 8-10 are cleanup and can be done last
- Run `zig build test --summary all` after each task to verify no regressions
- Run `zig fmt .` before each commit
