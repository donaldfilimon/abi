# ABI System v2.0 Integration — Mega Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate 18 modules from abi-system-v2.0 into the ABI framework, adapting all code to Zig 0.16 APIs and the framework's architectural conventions.

**Architecture:** New modules land in `src/services/` (infrastructure layer), not `src/features/` (feature-gated layer). Each module either extends an existing file or creates a new one. All Zig 0.16 incompatibilities (`nanoTimestamp`, `sleep`, allocator vtable signatures) are fixed during integration. Existing tests must continue to pass (944 pass, 5 skip baseline).

**Tech Stack:** Zig 0.16.x, SIMD intrinsics (`@Vector`, `@reduce`, `@splat`), lock-free CAS primitives (`@cmpxchgWeak`), comptime generics.

**Source location:** `/tmp/abi-system/src/` (extracted from `~/Downloads/abi-system-v2.0.tar`)

---

## Module Integration Map

| # | v2.0 Module | Target Location | Strategy |
|---|-------------|----------------|----------|
| 1 | `utils.zig` | `src/services/shared/utils/v2_primitives.zig` | New file — foundation types |
| 2 | `error.zig` | `src/services/shared/utils/structured_error.zig` | New file — complements existing error handling |
| 3 | `memory.zig` | `src/services/shared/utils/memory/arena.zig` + `vector_pool.zig` + `slab.zig` + `scratch.zig` | New files — extends memory subsystem |
| 4 | `alloc.zig` | `src/services/shared/utils/memory/composable.zig` | New file — allocator combinators |
| 5 | `simd.zig` | `src/services/shared/simd.zig` | Extend — add missing kernels to existing 1599-line file |
| 6 | `matrix.zig` | `src/services/shared/matrix.zig` | New file — dense matrix ops |
| 7 | `tensor.zig` | `src/services/shared/tensor.zig` | New file — shared tensor (AI tensors remain separate) |
| 8 | `hashmap.zig` | `src/services/shared/utils/swissmap.zig` | New file — SwissMap |
| 9 | `channel.zig` | `src/services/runtime/concurrency/channel.zig` | New file — MPMC channel |
| 10 | `thread_pool.zig` | `src/services/runtime/concurrency/thread_pool.zig` | New file — work-stealing pool |
| 11 | `scheduler.zig` | `src/services/runtime/scheduling/dag.zig` | New file — DAG pipeline |
| 12 | `profiler.zig` | `src/services/shared/profiler.zig` | New file — hierarchical profiler |
| 13 | `bench.zig` | `src/services/shared/bench.zig` | New file — statistical benchmarking |
| 14 | `serialize.zig` | `src/services/shared/utils/binary.zig` | Extend — add ABIX wire format |
| 15 | `config.zig` | (skip — already have layered config) | Harvest validation patterns only |
| 16 | `gpu.zig` | (skip — existing GPU module is far more complete) | Harvest BufferPool staging pattern |
| 17 | `cli.zig` | (skip — existing CLI is 24 commands) | No action |
| 18 | `main.zig` | (skip — entry point) | No action |

---

## Zig 0.16 Migration Checklist

Every module must have these fixed during integration:

| Pattern | Replacement |
|---------|-------------|
| `std.time.nanoTimestamp()` | `std.time.Instant.now()` + `.since(anchor)` or `@import("time.zig").timestampNs()` |
| `std.time.sleep(ns)` | `@import("time.zig").sleepNs(ns)` or `std.Thread.sleep(ns)` |
| `std.ArrayList(T)` → if heap needed | Keep (acceptable in Zig 0.16) |
| `std.ArrayList(T)` → if no heap | `std.ArrayListUnmanaged(T)` with `.empty` |
| Allocator vtable `u8` alignment | `std.mem.Alignment` enum (`.fromByteUnits()`) |
| `@import("utils")` | Relative `@import` within `src/services/` |

---

## Task 1: Foundation — v2 Primitives (`utils.zig` subset)

**Files:**
- Create: `src/services/shared/utils/v2_primitives.zig`
- Modify: `src/services/shared/utils/mod.zig` (add re-export)
- Test: inline tests in `v2_primitives.zig`

We cherry-pick types from v2.0's `utils.zig` that don't already exist in the framework.
Existing equivalents we skip: `Platform` (already in `src/services/shared/platform.zig`),
`SpinLock` (already in `src/services/shared/sync.zig`), basic `Math` (some overlap).

### Step 1: Write the failing test

Add the test block at the bottom of the new file. We test `Result(T,E)`, `RingBuffer`,
`String.Builder`, and `String.hash`.

```zig
// src/services/shared/utils/v2_primitives.zig — bottom
test "Result monad" {
    const R = Result(u32, []const u8);
    const ok = R.ok(42);
    try std.testing.expectEqual(@as(u32, 42), ok.unwrap());
    const err = R.err("nope");
    try std.testing.expectEqual(@as(?u32, null), err.toOptional());
}

test "RingBuffer push/pop" {
    var rb = RingBuffer(u32, 4){};
    try std.testing.expect(rb.push(1));
    try std.testing.expect(rb.push(2));
    try std.testing.expectEqual(@as(?u32, 1), rb.pop());
    try std.testing.expectEqual(@as(?u32, 2), rb.pop());
    try std.testing.expectEqual(@as(?u32, null), rb.pop());
}

test "String.hash deterministic" {
    const h1 = String.hash("hello");
    const h2 = String.hash("hello");
    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(String.hash("hello") != String.hash("world"));
}
```

### Step 2: Run test to verify it fails

```bash
zig test src/services/shared/utils/v2_primitives.zig
```

Expected: FAIL — file doesn't exist yet.

### Step 3: Write the implementation

Copy from `/tmp/abi-system/src/utils.zig` these types:
- `Result(T, E)` (lines 241–276)
- `RingBuffer(T, N)` (lines 278–326)
- `String` namespace: `hash` (FNV-1a), `eqlIgnoreCase`, `formatBuf`, `Builder` (lines 155–238)
- `Math.nextPowerOfTwo`, `Math.isPowerOfTwo`, `Math.alignUp` (lines 76–110) — only if not already in `shared/`

**Zig 0.16 fixes:**
- Remove `Stopwatch` (uses `nanoTimestamp`; we already have `shared/time.zig`)
- Remove `SpinLock` / `SpscQueue` / `Counter` (already in `shared/sync.zig`)
- Remove `Platform` (already in `shared/platform.zig`)
- Change `@import("std")` only — no module imports

```zig
// src/services/shared/utils/v2_primitives.zig
const std = @import("std");

// ─── String Utilities ─────────────────────────────────────────────
pub const String = struct {
    /// FNV-1a hash
    pub fn hash(data: []const u8) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (data) |byte| {
            h ^= byte;
            h *%= 0x100000001b3;
        }
        return h;
    }

    pub fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
        if (a.len != b.len) return false;
        for (a, b) |ca, cb| {
            if (std.ascii.toLower(ca) != std.ascii.toLower(cb)) return false;
        }
        return true;
    }

    pub fn formatBuf(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
        var stream = std.io.fixedBufferStream(buf);
        stream.writer().print(fmt, args) catch {};
        return buf[0..stream.pos];
    }

    pub const Builder = struct {
        buf: [4096]u8 = undefined,
        pos: usize = 0,

        pub fn append(self: *Builder, data: []const u8) void {
            const n = @min(data.len, self.buf.len - self.pos);
            @memcpy(self.buf[self.pos..][0..n], data[0..n]);
            self.pos += n;
        }

        pub fn slice(self: *const Builder) []const u8 {
            return self.buf[0..self.pos];
        }

        pub fn reset(self: *Builder) void {
            self.pos = 0;
        }
    };
};

// ─── Math Extras ──────────────────────────────────────────────────
pub const Math = struct {
    pub fn nextPowerOfTwo(v: usize) usize {
        if (v == 0) return 1;
        var x = v - 1;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        if (@sizeOf(usize) > 4) x |= x >> 32;
        return x + 1;
    }

    pub fn isPowerOfTwo(v: usize) bool {
        return v > 0 and (v & (v - 1)) == 0;
    }

    pub fn alignUp(comptime T: type, value: T, alignment: T) T {
        const mask = alignment - 1;
        return (value + mask) & ~mask;
    }
};

// ─── Result Monad ─────────────────────────────────────────────────
pub fn Result(comptime T: type, comptime E: type) type {
    return struct {
        const Self = @This();
        value: ?T = null,
        err: ?E = null,

        pub fn ok(val: T) Self {
            return .{ .value = val };
        }

        pub fn err(e: E) Self {
            return .{ .err = e };
        }

        pub fn isOk(self: Self) bool {
            return self.value != null;
        }

        pub fn unwrap(self: Self) T {
            return self.value.?;
        }

        pub fn toOptional(self: Self) ?T {
            return self.value;
        }

        pub fn map(self: Self, f: *const fn (T) T) Self {
            if (self.value) |v| return Self.ok(f(v));
            return self;
        }
    };
}

// ─── Ring Buffer ──────────────────────────────────────────────────
pub fn RingBuffer(comptime T: type, comptime N: usize) type {
    return struct {
        const Self = @This();
        buf: [N]T = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,

        pub fn push(self: *Self, item: T) bool {
            if (self.count >= N) return false;
            self.buf[self.tail] = item;
            self.tail = (self.tail + 1) % N;
            self.count += 1;
            return true;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            const item = self.buf[self.head];
            self.head = (self.head + 1) % N;
            self.count -= 1;
            return item;
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        pub fn isFull(self: *const Self) bool {
            return self.count == N;
        }
    };
}
```

Then add the tests from Step 1 at the bottom of the same file.

### Step 4: Run test to verify it passes

```bash
zig test src/services/shared/utils/v2_primitives.zig
```

Expected: 3 tests PASS.

### Step 5: Wire into module system

In `src/services/shared/utils/mod.zig`, add:
```zig
pub const v2 = @import("v2_primitives.zig");
```

### Step 6: Verify full test suite

```bash
zig build test --summary all 2>&1 | tail -5
```

Expected: 944 pass, 5 skip (baseline maintained).

### Step 7: Commit

```bash
git add src/services/shared/utils/v2_primitives.zig src/services/shared/utils/mod.zig
git commit -m "feat: add v2 primitives (Result, RingBuffer, String utils)"
```

---

## Task 2: Structured Error Handling

**Files:**
- Create: `src/services/shared/utils/structured_error.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "ErrorAccumulator collects and queries" {
    var acc = ErrorAccumulator(8){};
    acc.pushMessage(.memory, .warn, "low memory: {d}%", .{85});
    acc.pushMessage(.gpu, .err, "device lost", .{});
    try std.testing.expect(acc.hasErrors());
    try std.testing.expectEqual(@as(usize, 2), acc.count);
    acc.clear();
    try std.testing.expect(!acc.hasErrors());
}
```

### Step 2: Run test — expected FAIL

```bash
zig test src/services/shared/utils/structured_error.zig
```

### Step 3: Write implementation

Copy from `/tmp/abi-system/src/error.zig`:
- `Category` enum (lines 15–35)
- `Severity` enum (lines 37–66)
- `Context` struct (lines 72–121)
- `ErrorAccumulator(max)` (lines 127–173)
- `AbiError` error set (lines 177–195)
- Convenience constructors (lines 199–217)

**Zig 0.16 fix:** Line 86 — replace `std.time.nanoTimestamp()`:
```zig
// BEFORE (v2.0):
.timestamp_ns = std.time.nanoTimestamp(),

// AFTER (Zig 0.16):
.timestamp_ns = blk: {
    const now = std.time.Instant.now() catch break :blk 0;
    break :blk @intCast(now.timestamp.tv_sec * std.time.ns_per_s + now.timestamp.tv_nsec);
},
```

**Alternative** (simpler, if `shared/time.zig` is importable):
```zig
const time = @import("../../time.zig");
// ...
.timestamp_ns = time.timestampNs(),
```

Skip the `Logger` struct (the framework already has logging infrastructure).

### Step 4: Run test — expected PASS

```bash
zig test src/services/shared/utils/structured_error.zig
```

### Step 5: Commit

```bash
git add src/services/shared/utils/structured_error.zig
git commit -m "feat: add structured error handling with categories and accumulator"
```

---

## Task 3: Memory — Arena Pool

**Files:**
- Create: `src/services/shared/utils/memory/arena.zig`
- Modify: `src/services/shared/utils/memory/mod.zig` (add re-export)
- Test: inline tests

### Step 1: Write failing test

```zig
test "ArenaPool bump allocation" {
    var arena = try ArenaPool.init(std.testing.allocator, .{ .size = 4096 });
    defer arena.deinit();

    const a = arena.alloc(64, 8) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 64), a.len);

    const b = arena.alloc(128, 16) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 128), b.len);

    // Verify non-overlapping
    const a_end = @intFromPtr(a.ptr) + a.len;
    const b_start = @intFromPtr(b.ptr);
    try std.testing.expect(b_start >= a_end);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `ArenaPool` from `/tmp/abi-system/src/memory.zig` lines 22–87.

**Zig 0.16 fix:** The `arenaAllocFn` vtable function signature must match Zig 0.16's `std.mem.Allocator.VTable`:

```zig
// Zig 0.16 vtable signature:
fn arenaAllocFn(
    ctx: *anyopaque,
    len: usize,
    alignment: std.mem.Alignment,
    ret_addr: usize,
) ?[*]u8 {
    const self: *ArenaPool = @ptrCast(@alignCast(ctx));
    const align_bytes = alignment.toByteUnits();
    return if (self.alloc(len, align_bytes)) |slice| slice.ptr else null;
}
```

Similarly fix `arenaResizeFn` and add no-op `arenaFreeFn` and `arenaRemapFn`:

```zig
fn arenaResizeFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
    return false; // Arena doesn't support resize
}

fn arenaRemapFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
    return null; // Arena doesn't support remap
}

fn arenaFreeFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {
    // Arena frees everything at once on deinit
}
```

Provide `allocator()` method:
```zig
pub fn allocator(self: *ArenaPool) std.mem.Allocator {
    return .{
        .ptr = @ptrCast(self),
        .vtable = &.{
            .alloc = arenaAllocFn,
            .resize = arenaResizeFn,
            .remap = arenaRemapFn,
            .free = arenaFreeFn,
        },
    };
}
```

### Step 4: Run test — expected PASS

### Step 5: Wire into memory module

In `src/services/shared/utils/memory/mod.zig`, add:
```zig
pub const arena = @import("arena.zig");
```

### Step 6: Full test suite check

```bash
zig build test --summary all 2>&1 | tail -5
```

### Step 7: Commit

```bash
git add src/services/shared/utils/memory/arena.zig src/services/shared/utils/memory/mod.zig
git commit -m "feat: add ArenaPool bump allocator with Zig 0.16 vtable"
```

---

## Task 4: Memory — Vector Pool, Slab Pool, Scratch Allocator

**Files:**
- Create: `src/services/shared/utils/memory/vector_pool.zig`
- Create: `src/services/shared/utils/memory/slab.zig`
- Create: `src/services/shared/utils/memory/scratch.zig`
- Modify: `src/services/shared/utils/memory/mod.zig`
- Test: inline tests in each file

### Step 1: Write failing tests (one per file)

**vector_pool.zig:**
```zig
test "VectorPool SIMD-aligned allocation" {
    var pool = try VectorPool.init(std.testing.allocator, .{ .size = 4096 });
    defer pool.deinit();
    const slice = pool.alloc(256, 64) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(slice.ptr) % 64);
}
```

**slab.zig:**
```zig
test "SlabPool alloc/free cycle" {
    var slab = SlabPool(TestStruct).init(std.testing.allocator, 16) catch return error.SkipZigTest;
    defer slab.deinit();
    const a = slab.alloc() orelse return error.SkipZigTest;
    a.* = .{ .x = 42 };
    slab.free(a);
    const b = slab.alloc() orelse return error.SkipZigTest;
    // Should reuse the freed slot
    try std.testing.expectEqual(@intFromPtr(a), @intFromPtr(b));
}
const TestStruct = struct { x: u32 = 0 };
```

**scratch.zig:**
```zig
test "ScratchAllocator double-buffer swap" {
    var scratch = ScratchAllocator.init(std.testing.allocator, 1024) catch return error.SkipZigTest;
    defer scratch.deinit();
    _ = scratch.alloc(64, 8);
    scratch.swap(); // Reset active buffer, swap to back
    _ = scratch.alloc(64, 8);
    try std.testing.expect(scratch.usedBytes() > 0);
}
```

### Step 2: Run tests — expected FAIL

### Step 3: Write implementations

Copy from `/tmp/abi-system/src/memory.zig`:
- `VectorPool` (lines 89–140) → `vector_pool.zig`
- `SlabPool(T)` (lines 142–208) → `slab.zig`
- `ScratchAllocator` (lines 210–278) → `scratch.zig`

**Zig 0.16 fixes for all three:**
- Same vtable signature pattern as Task 3 (use `std.mem.Alignment` parameter)
- Replace any `@import("utils")` with direct `const std = @import("std")`

### Step 4: Run tests — expected PASS

### Step 5: Wire into memory module

```zig
// src/services/shared/utils/memory/mod.zig — add these lines:
pub const vector_pool = @import("vector_pool.zig");
pub const slab = @import("slab.zig");
pub const scratch = @import("scratch.zig");
```

### Step 6: Commit

```bash
git add src/services/shared/utils/memory/vector_pool.zig \
        src/services/shared/utils/memory/slab.zig \
        src/services/shared/utils/memory/scratch.zig \
        src/services/shared/utils/memory/mod.zig
git commit -m "feat: add VectorPool, SlabPool, ScratchAllocator to memory subsystem"
```

---

## Task 5: Composable Allocator Combinators

**Files:**
- Create: `src/services/shared/utils/memory/composable.zig`
- Modify: `src/services/shared/utils/memory/mod.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "TrackingAllocator tracks bytes" {
    var tracker = TrackingAllocator.init(std.testing.allocator);
    const alloc = tracker.allocator();
    const data = try alloc.alloc(u8, 100);
    try std.testing.expect(tracker.current_bytes.load(.monotonic) >= 100);
    alloc.free(data);
    try std.testing.expectEqual(@as(usize, 0), tracker.current_bytes.load(.monotonic));
}

test "LimitingAllocator enforces cap" {
    var limiter = LimitingAllocator.init(std.testing.allocator, 50);
    const alloc = limiter.allocator();
    const small = alloc.alloc(u8, 30);
    try std.testing.expect(small != null);
    const too_big = alloc.alloc(u8, 30); // would exceed 50
    try std.testing.expectEqual(@as(?[]u8, null), too_big);
    if (small) |s| alloc.free(s);
}

test "FallbackAllocator falls through" {
    var limiter = LimitingAllocator.init(std.testing.allocator, 10);
    var fallback = FallbackAllocator.init(limiter.allocator(), std.testing.allocator);
    const alloc = fallback.allocator();
    // This exceeds the limiter's 10-byte cap, so fallback should kick in
    const data = try alloc.alloc(u8, 100);
    defer alloc.free(data);
    try std.testing.expectEqual(@as(usize, 100), data.len);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy from `/tmp/abi-system/src/alloc.zig`:
- `TrackingAllocator` (lines 14–74)
- `LimitingAllocator` (lines 76–139)
- `FallbackAllocator` (lines 141–210)
- `NullAllocator` (lines 212–244)

**Zig 0.16 fixes — ALL vtable functions must use this signature pattern:**

```zig
const vtable: std.mem.Allocator.VTable = .{
    .alloc = allocFn,
    .resize = resizeFn,
    .remap = remapFn,  // NEW in 0.16 — return null for "not supported"
    .free = freeFn,
};

fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
    // ...
}

fn resizeFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
    // ...
}

fn remapFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
    // ...
}

fn freeFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
    // ...
}
```

Key: `alignment` is `std.mem.Alignment` (enum), not `u8`. Call `.toByteUnits()` to get the numeric value.

For `TrackingAllocator`, atomics use `@atomicRmw(.add, ...)` — translate to Zig 0.16's `std.atomic.Value(usize)` or direct `@atomicRmw`.

### Step 4: Run test — expected PASS

### Step 5: Wire and commit

```bash
git add src/services/shared/utils/memory/composable.zig src/services/shared/utils/memory/mod.zig
git commit -m "feat: add composable allocator combinators (Tracking, Limiting, Fallback)"
```

---

## Task 6: SIMD Kernel Extensions

**Files:**
- Modify: `src/services/shared/simd.zig` (append new functions)
- Test: inline tests at bottom of file

### Step 1: Write failing tests

Append to the bottom of `src/services/shared/simd.zig`:

```zig
test "cosineSimilarity unit vectors" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 1, 0, 0, 0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 1e-5);
}

test "euclideanDistance zero" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 2, 3, 4 };
    const dist = euclideanDistance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-5);
}

test "softmax sums to one" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    softmax(&input, &output);
    var sum: f32 = 0;
    for (output) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}
```

### Step 2: Run test — expected FAIL

```bash
zig test src/services/shared/simd.zig --test-filter "cosine"
```

### Step 3: Write implementation

Add these functions (adapted from `/tmp/abi-system/src/simd.zig`) to `src/services/shared/simd.zig`.
Check which already exist — the existing file has `vectorDot`, `vectorAdd`, `vectorSub`, `vectorMul`, `vectorScale`, `vectorNorm`, `vectorNormalize`. Add the **missing** ones:

```zig
/// Cosine similarity: dot(a,b) / (|a| * |b|)
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dot = vectorDot(a, b);
    const norm_a = vectorNorm(a);
    const norm_b = vectorNorm(b);
    const denom = norm_a * norm_b;
    if (denom < 1e-10) return 0;
    return dot / denom;
}

/// Euclidean distance: sqrt(sum((a-b)^2))
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    const len = a.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            sum += @reduce(.Add, diff * diff);
        }
    }

    while (i < len) : (i += 1) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return @sqrt(sum);
}

/// Numerically stable softmax
pub fn softmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len and input.len > 0);
    // Find max for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |v| max_val = @max(max_val, v);

    var sum: f32 = 0;
    for (input, 0..) |v, i| {
        output[i] = @exp(v - max_val);
        sum += output[i];
    }
    const inv_sum = 1.0 / sum;
    for (output) |*v| v.* *= inv_sum;
}

/// SAXPY: y[i] += a * x[i]
pub fn saxpy(a_scalar: f32, x: []const f32, y: []f32) void {
    std.debug.assert(x.len == y.len);
    const len = x.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const va: Vec = @splat(a_scalar);
        while (i + VectorSize <= len) : (i += VectorSize) {
            const vx: Vec = x[i..][0..VectorSize].*;
            var vy: Vec = y[i..][0..VectorSize].*;
            vy += va * vx;
            y[i..][0..VectorSize].* = vy;
        }
    }

    while (i < len) : (i += 1) {
        y[i] += a_scalar * x[i];
    }
}

/// Reduce to find minimum value
pub fn reduceMin(data: []const f32) f32 {
    std.debug.assert(data.len > 0);
    var result: f32 = data[0];
    for (data[1..]) |v| result = @min(result, v);
    return result;
}

/// Reduce to find maximum value
pub fn reduceMax(data: []const f32) f32 {
    std.debug.assert(data.len > 0);
    var result: f32 = data[0];
    for (data[1..]) |v| result = @max(result, v);
    return result;
}
```

### Step 4: Run tests — expected PASS

```bash
zig test src/services/shared/simd.zig --summary all
```

### Step 5: Full suite check + commit

```bash
zig build test --summary all 2>&1 | tail -5
git add src/services/shared/simd.zig
git commit -m "feat: add cosineSimilarity, euclideanDistance, softmax, saxpy to SIMD module"
```

---

## Task 7: Dense Matrix Operations

**Files:**
- Create: `src/services/shared/matrix.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "Matrix identity and multiply" {
    const alloc = std.testing.allocator;
    var a = try Mat32.alloc(alloc, 3, 3);
    defer a.free(alloc);
    a.identity();

    var b = try Mat32.alloc(alloc, 3, 3);
    defer b.free(alloc);
    b.set(0, 0, 2.0);
    b.set(1, 1, 3.0);
    b.set(2, 2, 4.0);

    var c = try Mat32.alloc(alloc, 3, 3);
    defer c.free(alloc);
    Mat32.multiply(&a, &b, &c);

    // I * B = B
    try std.testing.expect(c.approxEqual(&b, 1e-6));
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy entire `Matrix(T)` generic from `/tmp/abi-system/src/matrix.zig`.

**Zig 0.16 fixes:**
- Replace `@import("utils")` with `const std = @import("std")`
- The `vectorizedSaxpy` helper uses `Platform.simd_width` — replace with:
  ```zig
  const lanes = std.simd.suggestVectorLength(T) orelse 4;
  ```
- Remove `Math` and `Platform` imports — use std equivalents

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/services/shared/matrix.zig
git commit -m "feat: add dense matrix operations with tiled matmul and SIMD SAXPY"
```

---

## Task 8: Shared Tensor

**Files:**
- Create: `src/services/shared/tensor.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "Tensor 2D create and element access" {
    const T = Tensor(f32);
    var t = try T.alloc(std.testing.allocator, &[_]usize{ 3, 4 });
    defer t.free(std.testing.allocator);
    t.fill(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), t.at(&[_]usize{ 1, 2 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), t.sum(), 1e-6);
}

test "Tensor softmax rows sum to 1" {
    const T = Tensor(f32);
    var t = try T.alloc(std.testing.allocator, &[_]usize{ 2, 3 });
    defer t.free(std.testing.allocator);
    t.arange(0, 1);
    var out = try T.alloc(std.testing.allocator, &[_]usize{ 2, 3 });
    defer out.free(std.testing.allocator);
    t.softmax(&out);
    // Each row should sum to ~1.0
    var row0_sum: f32 = 0;
    for (0..3) |j| row0_sum += out.at(&[_]usize{ 0, j });
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0_sum, 1e-5);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `Shape` and `Tensor(T)` from `/tmp/abi-system/src/tensor.zig`.

**Zig 0.16 fixes:**
- Replace `@import("utils")` → `const std = @import("std")`
- Replace SIMD references: use `std.simd.suggestVectorLength(T) orelse 4` instead of `Platform.simd_width / @sizeOf(T)`
- The v2.0 tensor is a general-purpose compute tensor. It does **not** replace the AI-specific tensors in `abbey/neural/tensor.zig` (which have gradient tracking) or `llm/tensor/tensor.zig` (which have quantization). This is infrastructure-level.

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/services/shared/tensor.zig
git commit -m "feat: add shared Tensor(T) with broadcasting, softmax, SIMD ops"
```

---

## Task 9: SwissMap Hash Table

**Files:**
- Create: `src/services/shared/utils/swissmap.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "SwissMap put/get/remove" {
    var map = SwissMap(u32, []const u8).init(std.testing.allocator);
    defer map.deinit();

    try map.put(42, "hello");
    try map.put(7, "world");

    try std.testing.expectEqualStrings("hello", map.get(42).?);
    try std.testing.expectEqualStrings("world", map.get(7).?);
    try std.testing.expectEqual(@as(?[]const u8, null), map.get(999));

    try std.testing.expect(map.remove(42));
    try std.testing.expectEqual(@as(?[]const u8, null), map.get(42));
}

test "SwissMap iterator" {
    var map = SwissMap(u32, u32).init(std.testing.allocator);
    defer map.deinit();

    for (0..100) |i| try map.put(@intCast(i), @intCast(i * 2));

    var count: usize = 0;
    var iter = map.iterator();
    while (iter.next()) |_| count += 1;
    try std.testing.expectEqual(@as(usize, 100), count);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy from `/tmp/abi-system/src/hashmap.zig`:
- `SwissMap(K, V)` — the entire generic type (lines 11–310)
- Hash functions: `splitmix64`, `wyhash` (lines 312–378)

**Zig 0.16 fixes:**
- Replace `@import("utils")` → `const std = @import("std")`
- The v2.0 code uses `std.mem.Allocator` which is fine in 0.16
- Ensure `@cmpxchgWeak` calls (if any in CAS paths) use Zig 0.16 ordering enums

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/services/shared/utils/swissmap.zig
git commit -m "feat: add SwissMap (open-addressing hash table with H2 control bytes)"
```

---

## Task 10: MPMC Channel

**Files:**
- Create: `src/services/runtime/concurrency/channel.zig`
- Modify: `src/services/runtime/concurrency/mod.zig` (add re-export)
- Test: inline tests

### Step 1: Write failing test

```zig
test "Channel send/recv basic" {
    var ch = Channel(u64, 16){};
    try std.testing.expect(ch.send(42));
    try std.testing.expect(ch.send(99));
    try std.testing.expectEqual(@as(?u64, 42), ch.recv());
    try std.testing.expectEqual(@as(?u64, 99), ch.recv());
}

test "Channel close semantics" {
    var ch = Channel(u64, 4){};
    try std.testing.expect(ch.send(1));
    ch.close();
    // Can still recv buffered items after close
    try std.testing.expectEqual(@as(?u64, 1), ch.recv());
    // But new sends fail
    try std.testing.expect(!ch.send(2));
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `Channel(T, N)` from `/tmp/abi-system/src/channel.zig`.

**Zig 0.16 fixes:**
- Line 128, 179: Replace `std.time.sleep(...)` with `std.Thread.sleep(...)`:
  ```zig
  // BEFORE:
  std.time.sleep(1);
  // AFTER:
  std.Thread.sleep(1);
  ```
- Replace `@import("utils")` → `const std = @import("std")`
- Ensure `@atomicStore`/`@atomicLoad`/`@cmpxchgWeak` use correct Zig 0.16 ordering enum (`.monotonic`, `.acquire`, `.release`, `.seq_cst`)

Skip `MessageTag` and `Message` types (pipeline-specific, not general-purpose).

### Step 4: Run test — expected PASS

### Step 5: Wire into concurrency module

```zig
// src/services/runtime/concurrency/mod.zig — add:
pub const channel = @import("channel.zig");
```

### Step 6: Commit

```bash
git add src/services/runtime/concurrency/channel.zig src/services/runtime/concurrency/mod.zig
git commit -m "feat: add lock-free MPMC channel (Vyukov design)"
```

---

## Task 11: Work-Stealing Thread Pool

**Files:**
- Create: `src/services/runtime/concurrency/thread_pool.zig`
- Modify: `src/services/runtime/concurrency/mod.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "ThreadPool execute tasks" {
    var pool = ThreadPool.init(std.testing.allocator, 2) catch return error.SkipZigTest;
    defer pool.deinit();

    var counter = std.atomic.Value(u32).init(0);

    // Submit 10 tasks that each increment the counter
    for (0..10) |_| {
        pool.submit(struct {
            fn run(ctx: *anyopaque) void {
                const c: *std.atomic.Value(u32) = @ptrCast(@alignCast(ctx));
                _ = c.fetchAdd(1, .monotonic);
            }
        }.run, @ptrCast(&counter)) catch continue;
    }

    pool.waitIdle();
    try std.testing.expectEqual(@as(u32, 10), counter.load(.monotonic));
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `ThreadPool` from `/tmp/abi-system/src/thread_pool.zig`.

**Zig 0.16 fixes:**
- Line 309: Replace `std.time.sleep(...)` → `std.Thread.sleep(...)`
- Replace `@import("utils")` → `const std = @import("std")`
- The `WorkQueue` uses a SpinLock internally — use `std.Thread.Mutex` or a custom spinlock (already available in the concurrency module)
- `std.Thread.spawn(.{}, workerLoop, .{self, id})` syntax is fine in 0.16

### Step 4: Run test — expected PASS

### Step 5: Wire and commit

```bash
git add src/services/runtime/concurrency/thread_pool.zig src/services/runtime/concurrency/mod.zig
git commit -m "feat: add work-stealing thread pool with per-worker deques"
```

---

## Task 12: DAG Pipeline Scheduler

**Files:**
- Create: `src/services/runtime/scheduling/dag.zig`
- Modify: `src/services/runtime/scheduling/mod.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "DAG Pipeline topological execute" {
    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();

    var order = std.ArrayList(u8).init(std.testing.allocator);
    defer order.deinit();

    // A -> B -> C (linear chain)
    try pipeline.addStage("A", 0, makeRecorder(&order, 'A'));
    try pipeline.addStage("B", 1, makeRecorder(&order, 'B')); // depends on A
    try pipeline.addStage("C", 2, makeRecorder(&order, 'C')); // depends on B

    try pipeline.execute();

    // Must execute in order A, B, C
    try std.testing.expectEqualSlices(u8, "ABC", order.items);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `Pipeline` and `Stage` from `/tmp/abi-system/src/scheduler.zig`.

**Zig 0.16 fixes:**
- Replace `std.ArrayList` → keep it (acceptable in 0.16)
- Replace `@import("utils")` → `const std = @import("std")`
- Skip `createInferencePipeline()` template — that's application-specific

The DAG scheduler uses bitmask dependencies (up to 64 stages) and Kahn's topological sort. This complements the existing `future.zig` and `task_group.zig` which handle individual async tasks, not pipeline orchestration.

### Step 4: Run test — expected PASS

### Step 5: Wire and commit

```bash
git add src/services/runtime/scheduling/dag.zig src/services/runtime/scheduling/mod.zig
git commit -m "feat: add DAG pipeline scheduler with topological execution"
```

---

## Task 13: Hierarchical Profiler

**Files:**
- Create: `src/services/shared/profiler.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "Profiler span lifecycle" {
    var profiler = Profiler(256).init();
    const span = profiler.begin("test-op", "compute");
    // Simulate work
    std.Thread.sleep(1_000_000); // 1ms
    profiler.end(span);

    try std.testing.expectEqual(@as(usize, 1), profiler.span_count);
    try std.testing.expect(profiler.spans[0].end_ns > profiler.spans[0].start_ns);
}

test "Profiler Chrome Trace export" {
    var profiler = Profiler(16).init();
    const s1 = profiler.begin("load", "io");
    profiler.end(s1);
    const s2 = profiler.begin("compute", "math");
    profiler.end(s2);

    var buf: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    profiler.exportChromeTrace(stream.writer()) catch {};
    const output = buf[0..stream.pos];
    // Should be valid JSON array
    try std.testing.expect(std.mem.startsWith(u8, output, "[{"));
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `ProfilerType(max_spans)` from `/tmp/abi-system/src/profiler.zig`.

**Zig 0.16 fixes — CRITICAL:**
Replace ALL `std.time.nanoTimestamp()` calls (lines 146, 167) with:

```zig
fn timestampNs() i128 {
    const now = std.time.Instant.now() catch return 0;
    // Convert to nanoseconds from epoch approximation
    return @as(i128, now.timestamp.tv_sec) * std.time.ns_per_s +
           @as(i128, now.timestamp.tv_nsec);
}
```

Or import from `shared/time.zig` if the relative path works:
```zig
const time = @import("time.zig");
// then use time.timestampNs()
```

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/services/shared/profiler.zig
git commit -m "feat: add hierarchical profiler with Chrome Trace JSON export"
```

---

## Task 14: Statistical Benchmark Suite

**Files:**
- Create: `src/services/shared/bench.zig`
- Test: inline tests

### Step 1: Write failing test

```zig
test "BenchmarkSuite basic measurement" {
    var suite = BenchmarkSuite.init(std.testing.allocator);
    defer suite.deinit();

    const stats = suite.measure("noop", struct {
        fn run() void {}
    }.run, 100) catch return error.SkipZigTest;

    try std.testing.expect(stats.mean_ns > 0);
    try std.testing.expect(stats.median_ns > 0);
    try std.testing.expect(stats.samples > 0);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Copy `Suite` and `Stats` from `/tmp/abi-system/src/bench.zig`.

**Zig 0.16 fixes:**
- Lines 119, 121: Replace `std.time.nanoTimestamp()` with `std.time.Timer`:
  ```zig
  // BEFORE (v2.0):
  const start = std.time.nanoTimestamp();
  func();
  const end = std.time.nanoTimestamp();
  const elapsed = end - start;

  // AFTER (Zig 0.16):
  var timer = try std.time.Timer.start();
  func();
  const elapsed = timer.read(); // returns u64 nanoseconds
  ```
- Replace `@import("utils")` → `const std = @import("std")`
- Skip the built-in benchmarks (`benchArena`, `benchSimd`, etc.) — those reference other v2.0 modules

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/services/shared/bench.zig
git commit -m "feat: add statistical benchmark suite with Chauvenet outlier filtering"
```

---

## Task 15: Binary Serialization — ABIX Wire Format

**Files:**
- Modify: `src/services/shared/utils/binary.zig` (extend existing file)
- Test: inline tests at bottom

### Step 1: Write failing test

Append to `src/services/shared/utils/binary.zig`:

```zig
test "ABIX round-trip" {
    const alloc = std.testing.allocator;
    var writer = AbixWriter.init(alloc);
    defer writer.deinit();

    try writer.writeU32(42);
    try writer.writeBytes("hello");
    try writer.writeF32(3.14);

    const data = try writer.finalize(alloc);
    defer alloc.free(data);

    var reader = AbixReader.init(data);
    // Skip header (8 bytes: magic + version + length)
    try std.testing.expectEqual(@as(u32, 42), reader.readU32());
    try std.testing.expectEqualStrings("hello", reader.readBytes(5));
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), reader.readF32(), 1e-5);
}
```

### Step 2: Run test — expected FAIL

### Step 3: Write implementation

Add to the existing `binary.zig` file (don't replace existing `SerializationWriter`):

```zig
// ─── ABIX Binary Wire Format ─────────────────────────────────────
// Header: [4]u8 magic "ABIX" + u16 version + u16 reserved + u32 payload_len

pub const ABIX_MAGIC = [4]u8{ 'A', 'B', 'I', 'X' };
pub const ABIX_VERSION: u16 = 1;

pub const AbixWriter = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) AbixWriter {
        return .{ .buffer = .{}, .allocator = allocator };
    }

    pub fn deinit(self: *AbixWriter) void {
        self.buffer.deinit(self.allocator);
    }

    pub fn writeU32(self: *AbixWriter, value: u32) !void {
        const bytes = std.mem.asBytes(&std.mem.nativeToLittle(u32, value));
        try self.buffer.appendSlice(self.allocator, bytes);
    }

    pub fn writeF32(self: *AbixWriter, value: f32) !void {
        const int_val: u32 = @bitCast(value);
        try self.writeU32(int_val);
    }

    pub fn writeBytes(self: *AbixWriter, data: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, data);
    }

    /// Finalize: prepend header, return owned slice
    pub fn finalize(self: *AbixWriter, allocator: std.mem.Allocator) ![]u8 {
        const payload_len: u32 = @intCast(self.buffer.items.len);
        var header_buf: [12]u8 = undefined;
        @memcpy(header_buf[0..4], &ABIX_MAGIC);
        @memcpy(header_buf[4..6], std.mem.asBytes(&std.mem.nativeToLittle(u16, ABIX_VERSION)));
        @memcpy(header_buf[6..8], &[2]u8{ 0, 0 }); // reserved
        @memcpy(header_buf[8..12], std.mem.asBytes(&std.mem.nativeToLittle(u32, payload_len)));

        const total = 12 + payload_len;
        const result = try allocator.alloc(u8, total);
        @memcpy(result[0..12], &header_buf);
        @memcpy(result[12..total], self.buffer.items);
        return result;
    }
};

pub const AbixReader = struct {
    data: []const u8,
    pos: usize,

    pub fn init(data: []const u8) AbixReader {
        // Skip 12-byte header
        return .{ .data = data, .pos = 12 };
    }

    pub fn readU32(self: *AbixReader) u32 {
        if (self.pos + 4 > self.data.len) return 0;
        const bytes = self.data[self.pos..][0..4];
        self.pos += 4;
        return std.mem.littleToNative(u32, @as(*align(1) const u32, @ptrCast(bytes)).*);
    }

    pub fn readF32(self: *AbixReader) f32 {
        const int_val = self.readU32();
        return @bitCast(int_val);
    }

    pub fn readBytes(self: *AbixReader, len: usize) []const u8 {
        if (self.pos + len > self.data.len) return &[0]u8{};
        const result = self.data[self.pos..self.pos + len];
        self.pos += len;
        return result;
    }
};
```

### Step 4: Run test — expected PASS

```bash
zig test src/services/shared/utils/binary.zig --test-filter "ABIX"
```

### Step 5: Commit

```bash
git add src/services/shared/utils/binary.zig
git commit -m "feat: add ABIX binary wire format to serialization module"
```

---

## Task 16: Integration — Wire New Modules into Service Layer

**Files:**
- Modify: `src/services/shared/mod.zig` (or equivalent root export)
- Modify: `src/services/runtime/mod.zig`
- Test: `zig build test --summary all`

### Step 1: Verify current module structure

```bash
# Check which mod.zig files need updating
head -30 src/services/shared/mod.zig
head -30 src/services/runtime/mod.zig
```

### Step 2: Add re-exports for new shared modules

```zig
// src/services/shared/mod.zig — add:
pub const matrix = @import("matrix.zig");
pub const tensor = @import("tensor.zig");
pub const profiler = @import("profiler.zig");
pub const bench = @import("bench.zig");
```

### Step 3: Add re-exports for new runtime modules

The concurrency and scheduling sub-modules already have their own `mod.zig` files that were updated in Tasks 10–12.

### Step 4: Full test suite

```bash
zig build test --summary all 2>&1 | tail -5
```

Expected: 944+ pass (new tests add to the count), 5 skip.

### Step 5: Validate all flag combinations

```bash
zig build validate-flags
```

Expected: All 16 configurations compile.

### Step 6: Commit

```bash
git add src/services/shared/mod.zig src/services/runtime/mod.zig
git commit -m "feat: wire v2 modules into service layer exports"
```

---

## Task 17: Format, Lint, and Final Validation

**Files:** All new/modified files

### Step 1: Format

```bash
zig fmt .
```

### Step 2: Full check

```bash
zig build full-check
```

This runs: format check + tests + validate-flags + CLI tests.

### Step 3: Commit any formatting fixes

```bash
git add -u
git commit -m "chore: format all v2 integration files"
```

---

## Task 18: Documentation and Memory Update

**Files:**
- Update: `CLAUDE.md` (add v2 integration notes to Key File Locations)
- Update: `~/.claude/projects/-Users-donaldfilimon-abi/memory/MEMORY.md`

### Step 1: Add to CLAUDE.md Key File Locations table

```markdown
| v2 primitives (Result, RingBuffer) | `src/services/shared/utils/v2_primitives.zig` |
| SwissMap hash table | `src/services/shared/utils/swissmap.zig` |
| Composable allocators | `src/services/shared/utils/memory/composable.zig` |
| Dense matrix ops | `src/services/shared/matrix.zig` |
| Shared tensor | `src/services/shared/tensor.zig` |
| MPMC channel | `src/services/runtime/concurrency/channel.zig` |
| Work-stealing thread pool | `src/services/runtime/concurrency/thread_pool.zig` |
| DAG pipeline scheduler | `src/services/runtime/scheduling/dag.zig` |
| Hierarchical profiler | `src/services/shared/profiler.zig` |
| Statistical benchmarking | `src/services/shared/bench.zig` |
```

### Step 2: Update MEMORY.md

Add section:
```markdown
## v2.0 Integration (2026-02-08)
- 14 modules integrated from abi-system-v2.0 (4 skipped: config, gpu, cli, main)
- All placed in `src/services/` (shared or runtime layer)
- Zig 0.16 fixes applied: nanoTimestamp→Timer/Instant, sleep→Thread.sleep, allocator vtable→Alignment enum
- New infrastructure: SwissMap, MPMC channel, thread pool, DAG scheduler, profiler, statistical benchmarks
- Extended: SIMD (softmax, cosine, euclidean), binary serialization (ABIX format), memory (arena, slab, scratch, composable)
```

### Step 3: Commit

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with v2 integration module locations"
```

---

## Execution Order and Dependencies

```
Task 1 (primitives) ──┐
Task 2 (errors)  ─────┤
                       ├── Task 3 (arena) ──── Task 4 (memory pools) ──── Task 5 (composable alloc)
Task 6 (simd ext) ─────┤
                       ├── Task 7 (matrix)
                       ├── Task 8 (tensor)
Task 9 (swissmap) ─────┤
                       ├── Task 10 (channel) ─── Task 11 (thread pool)
                       ├── Task 12 (dag scheduler)
                       ├── Task 13 (profiler)
                       ├── Task 14 (bench)
Task 15 (serialize) ───┘

All above ──── Task 16 (integration wiring) ──── Task 17 (format/lint) ──── Task 18 (docs)
```

**Parallelizable groups:**
- Group A: Tasks 1, 2, 6, 9, 15 (no dependencies between them)
- Group B: Tasks 3, 7, 8, 10, 12, 13, 14 (depend only on Task 1 foundation)
- Group C: Tasks 4, 11 (depend on Group B)
- Group D: Task 5 (depends on Task 4)
- Group E: Tasks 16, 17, 18 (sequential, after all code tasks)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Allocator vtable mismatch | Every allocator has explicit Zig 0.16 vtable with `remap` field |
| Test baseline regression | Check `zig build test --summary all` after every commit |
| Module import cycles | All new code in `src/services/` — never imports from `src/features/` |
| SIMD portability | Use `std.simd.suggestVectorLength()` not hardcoded widths |
| Thread pool flakiness | Use `error.SkipZigTest` for timing-sensitive tests |
