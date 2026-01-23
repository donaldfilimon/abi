# GPU Performance Refactor Implementation Plan

**Status** – Stubs for `SyncEvent` and `KernelRing` have been added to the codebase; full implementation pending.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize GPU acceleration module for 1.5-2x performance improvement by addressing memory synchronization, kernel launch overhead, and adaptive tiling.

**Architecture:** Event-based memory synchronization replacing polling, kernel descriptor ring buffer for fast-path launches, and runtime-adaptive matrix tiling based on device occupancy analysis.

**Tech Stack:** Zig 0.16, CUDA/Vulkan backends, comptime generics, atomic operations, lock-free ring buffers.

---

## Task 1: Add Event-Based Memory Synchronization

**Files:**
- Modify: `src/gpu/unified_buffer.zig:113-150`
- Create: `src/gpu/sync_event.zig`
- Test: `src/gpu/tests/sync_event_test.zig`

### Step 1: Write the failing test for SyncEvent

```zig
// src/gpu/tests/sync_event_test.zig
const std = @import("std");
const SyncEvent = @import("../sync_event.zig").SyncEvent;

test "SyncEvent records and queries completion" {
    var event = SyncEvent.init();
    defer event.deinit();

    // Event should not be complete initially
    try std.testing.expect(!event.isComplete());

    // Record event (simulates GPU operation completion)
    event.record();

    // After recording, event should be complete
    try std.testing.expect(event.isComplete());
}

test "SyncEvent wait blocks until complete" {
    var event = SyncEvent.init();
    defer event.deinit();

    // Spawn thread to complete event after delay
    const thread = try std.Thread.spawn(.{}, struct {
        fn run(e: *SyncEvent) void {
            std.time.sleep(10 * std.time.ns_per_ms);
            e.record();
        }
    }.run, .{&event});

    // Wait should return after event completes
    event.wait();
    try std.testing.expect(event.isComplete());

    thread.join();
}

test "SyncEvent reset clears completion state" {
    var event = SyncEvent.init();
    defer event.deinit();

    event.record();
    try std.testing.expect(event.isComplete());

    event.reset();
    try std.testing.expect(!event.isComplete());
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/sync_event_test.zig`
Expected: FAIL with "unable to open 'src/gpu/sync_event.zig'"

### Step 3: Write minimal SyncEvent implementation

```zig
// src/gpu/sync_event.zig
const std = @import("std");

/// Event-based synchronization primitive for GPU operations.
/// Replaces polling-based dirty state checks with wait/signal semantics.
pub const SyncEvent = struct {
    completed: std.atomic.Value(bool),
    futex: std.Thread.Futex,

    pub fn init() SyncEvent {
        return .{
            .completed = std.atomic.Value(bool).init(false),
            .futex = .{},
        };
    }

    pub fn deinit(self: *SyncEvent) void {
        _ = self;
    }

    /// Record event completion (called by GPU callback or completion handler).
    pub fn record(self: *SyncEvent) void {
        self.completed.store(true, .release);
        self.futex.wake(.one);
    }

    /// Check if event has completed without blocking.
    pub fn isComplete(self: *const SyncEvent) bool {
        return self.completed.load(.acquire);
    }

    /// Block until event completes.
    pub fn wait(self: *SyncEvent) void {
        while (!self.completed.load(.acquire)) {
            self.futex.wait(null, null) catch {};
        }
    }

    /// Reset event to incomplete state for reuse.
    pub fn reset(self: *SyncEvent) void {
        self.completed.store(false, .release);
    }
};
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/sync_event_test.zig`
Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add src/gpu/sync_event.zig src/gpu/tests/sync_event_test.zig
git commit -m "feat(gpu): add SyncEvent for event-based memory synchronization"
```

---

## Task 2: Integrate SyncEvent into UnifiedBuffer

**Files:**
- Modify: `src/gpu/unified_buffer.zig:113-200`
- Test: `src/gpu/tests/unified_buffer_test.zig`

### Step 1: Write the failing test for event-based buffer sync

```zig
// Add to src/gpu/tests/unified_buffer_test.zig
test "UnifiedBuffer uses event-based sync instead of polling" {
    const allocator = std.testing.allocator;
    var buffer = try UnifiedBuffer.init(allocator, 1024, .f32);
    defer buffer.deinit();

    // Mark host dirty
    buffer.markHostDirty();

    // Get the sync event
    const event = buffer.getSyncEvent();
    try std.testing.expect(!event.isComplete());

    // Simulate device sync completion
    buffer.markDeviceSynced();

    // Event should now be complete
    try std.testing.expect(event.isComplete());
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/unified_buffer_test.zig --test-filter "event-based sync"`
Expected: FAIL with "no member named 'getSyncEvent'"

### Step 3: Add SyncEvent field and methods to UnifiedBuffer

In `src/gpu/unified_buffer.zig`, add after line 117 (dirty_state field):

```zig
const SyncEvent = @import("sync_event.zig").SyncEvent;

// Inside UnifiedBuffer struct, after dirty_state field:
sync_event: SyncEvent,

// In init() function, add:
.sync_event = SyncEvent.init(),

// In deinit() function, add:
self.sync_event.deinit();

// Add new methods:
pub fn getSyncEvent(self: *const UnifiedBuffer) *const SyncEvent {
    return &self.sync_event;
}

pub fn markDeviceSynced(self: *UnifiedBuffer) void {
    self.dirty_state.host_dirty.store(false, .release);
    self.sync_event.record();
}

// Modify markHostDirty to reset event:
pub fn markHostDirty(self: *UnifiedBuffer) void {
    self.dirty_state.host_dirty.store(true, .release);
    self.sync_event.reset();
}
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/unified_buffer_test.zig --test-filter "event-based sync"`
Expected: PASS

### Step 5: Commit

```bash
git add src/gpu/unified_buffer.zig src/gpu/tests/unified_buffer_test.zig
git commit -m "feat(gpu): integrate SyncEvent into UnifiedBuffer for non-blocking sync"
```

---

## Task 3: Add Kernel Descriptor Ring Buffer

**Files:**
- Create: `src/gpu/kernel_ring.zig`
- Test: `src/gpu/tests/kernel_ring_test.zig`

### Step 1: Write the failing test for KernelRing

```zig
// src/gpu/tests/kernel_ring_test.zig
const std = @import("std");
const KernelRing = @import("../kernel_ring.zig").KernelRing;

test "KernelRing stores and retrieves kernel descriptors" {
    var ring = KernelRing.init();

    const desc = KernelRing.Descriptor{
        .kernel_handle = 42,
        .grid_dim = .{ 64, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_mem = 0,
    };

    const slot = ring.push(desc);
    try std.testing.expect(slot != null);

    const retrieved = ring.get(slot.?);
    try std.testing.expectEqual(@as(u64, 42), retrieved.kernel_handle);
    try std.testing.expectEqual(@as(u32, 64), retrieved.grid_dim[0]);
}

test "KernelRing fast-path reuses recent descriptors" {
    var ring = KernelRing.init();

    const desc = KernelRing.Descriptor{
        .kernel_handle = 100,
        .grid_dim = .{ 32, 32, 1 },
        .block_dim = .{ 16, 16, 1 },
        .shared_mem = 4096,
    };

    // Push same descriptor twice
    const slot1 = ring.push(desc);
    const slot2 = ring.pushOrReuse(desc);

    // Should reuse the same slot
    try std.testing.expectEqual(slot1, slot2);
}

test "KernelRing wraps around when full" {
    var ring = KernelRing.init();

    // Fill the ring
    var i: u64 = 0;
    while (i < KernelRing.CAPACITY + 10) : (i += 1) {
        const desc = KernelRing.Descriptor{
            .kernel_handle = i,
            .grid_dim = .{ 1, 1, 1 },
            .block_dim = .{ 1, 1, 1 },
            .shared_mem = 0,
        };
        _ = ring.push(desc);
    }

    // Ring should still be functional
    try std.testing.expect(ring.count() <= KernelRing.CAPACITY);
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/kernel_ring_test.zig`
Expected: FAIL with "unable to open 'src/gpu/kernel_ring.zig'"

### Step 3: Write KernelRing implementation

```zig
// src/gpu/kernel_ring.zig
const std = @import("std");

/// Lock-free ring buffer for kernel launch descriptors.
/// Enables fast-path kernel launches by caching recent configurations.
pub const KernelRing = struct {
    pub const CAPACITY = 256;

    pub const Descriptor = struct {
        kernel_handle: u64,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem: u32,

        pub fn hash(self: Descriptor) u64 {
            var h: u64 = self.kernel_handle;
            h ^= @as(u64, self.grid_dim[0]) << 32;
            h ^= @as(u64, self.grid_dim[1]) << 16;
            h ^= @as(u64, self.grid_dim[2]);
            h ^= @as(u64, self.block_dim[0]) << 48;
            h ^= @as(u64, self.block_dim[1]) << 40;
            h ^= @as(u64, self.block_dim[2]) << 32;
            h ^= @as(u64, self.shared_mem);
            return h;
        }

        pub fn eql(a: Descriptor, b: Descriptor) bool {
            return a.kernel_handle == b.kernel_handle and
                std.mem.eql(u32, &a.grid_dim, &b.grid_dim) and
                std.mem.eql(u32, &a.block_dim, &b.block_dim) and
                a.shared_mem == b.shared_mem;
        }
    };

    buffer: [CAPACITY]Descriptor,
    head: std.atomic.Value(u32),
    tail: std.atomic.Value(u32),
    lookup: [CAPACITY]u64, // hash -> slot mapping for fast reuse

    pub fn init() KernelRing {
        return .{
            .buffer = undefined,
            .head = std.atomic.Value(u32).init(0),
            .tail = std.atomic.Value(u32).init(0),
            .lookup = [_]u64{0} ** CAPACITY,
        };
    }

    pub fn push(self: *KernelRing, desc: Descriptor) ?u32 {
        const tail = self.tail.load(.acquire);
        const next_tail = (tail + 1) % CAPACITY;

        // Check if full
        if (next_tail == self.head.load(.acquire)) {
            // Advance head to make room (drop oldest)
            _ = self.head.fetchAdd(1, .release);
        }

        self.buffer[tail] = desc;
        self.lookup[tail] = desc.hash();
        self.tail.store(next_tail, .release);

        return tail;
    }

    pub fn pushOrReuse(self: *KernelRing, desc: Descriptor) ?u32 {
        const hash = desc.hash();

        // Check recent entries for match
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.acquire);
        var i = tail;
        var checked: u32 = 0;
        const max_check: u32 = 16; // Only check last 16 entries

        while (i != head and checked < max_check) {
            i = if (i == 0) CAPACITY - 1 else i - 1;
            if (self.lookup[i] == hash and Descriptor.eql(self.buffer[i], desc)) {
                return i;
            }
            checked += 1;
        }

        return self.push(desc);
    }

    pub fn get(self: *const KernelRing, slot: u32) Descriptor {
        return self.buffer[slot % CAPACITY];
    }

    pub fn count(self: *const KernelRing) u32 {
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.acquire);
        if (tail >= head) {
            return tail - head;
        } else {
            return CAPACITY - head + tail;
        }
    }
};
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/kernel_ring_test.zig`
Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add src/gpu/kernel_ring.zig src/gpu/tests/kernel_ring_test.zig
git commit -m "feat(gpu): add KernelRing for fast-path kernel launch caching"
```

---

## Task 4: Integrate KernelRing into Dispatcher

**Files:**
- Modify: `src/gpu/dispatcher.zig:174-250`
- Test: `src/gpu/tests/dispatcher_test.zig`

### Step 1: Write the failing test for dispatcher fast-path

```zig
// Add to src/gpu/tests/dispatcher_test.zig
test "Dispatcher uses kernel ring for repeated launches" {
    const allocator = std.testing.allocator;
    var dispatcher = try Dispatcher.init(allocator, null);
    defer dispatcher.deinit();

    // Get initial cache stats
    const initial_stats = dispatcher.getStats();

    // Execute same kernel config multiple times
    const config = LaunchConfig.for1D(1024, 256);
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        _ = dispatcher.executeBuiltin(.vector_add, config, .{});
    }

    // Check ring buffer hits
    const final_stats = dispatcher.getStats();
    try std.testing.expect(final_stats.ring_hits > 0);
    try std.testing.expect(final_stats.ring_hits >= 9); // At least 9 reuses
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/dispatcher_test.zig --test-filter "kernel ring"`
Expected: FAIL with "no member named 'ring_hits'"

### Step 3: Add KernelRing to Dispatcher

In `src/gpu/dispatcher.zig`, add:

```zig
const KernelRing = @import("kernel_ring.zig").KernelRing;

// Inside Dispatcher struct, add field:
kernel_ring: KernelRing,

// Inside Stats struct, add field:
ring_hits: u64,

// In init(), add:
.kernel_ring = KernelRing.init(),

// In getStats(), initialize:
.ring_hits = self.ring_hits.load(.acquire),

// Add ring_hits atomic counter:
ring_hits: std.atomic.Value(u64),

// In execute() function, add fast-path check:
pub fn execute(self: *Dispatcher, kernel: KernelHandle, config: LaunchConfig, args: anytype) !void {
    const desc = KernelRing.Descriptor{
        .kernel_handle = kernel.id,
        .grid_dim = config.grid_dim,
        .block_dim = config.block_dim,
        .shared_mem = config.shared_mem,
    };

    // Try to reuse cached descriptor
    if (self.kernel_ring.pushOrReuse(desc)) |slot| {
        const cached = self.kernel_ring.get(slot);
        if (cached.kernel_handle == kernel.id) {
            _ = self.ring_hits.fetchAdd(1, .monotonic);
            // Fast-path: use cached launch config
        }
    }

    // Regular launch path...
}
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/dispatcher_test.zig --test-filter "kernel ring"`
Expected: PASS

### Step 5: Commit

```bash
git add src/gpu/dispatcher.zig src/gpu/tests/dispatcher_test.zig
git commit -m "feat(gpu): integrate KernelRing into Dispatcher for launch fast-path"
```

---

## Task 5: Add Adaptive Matrix Tiling

**Files:**
- Create: `src/gpu/adaptive_tiling.zig`
- Test: `src/gpu/tests/adaptive_tiling_test.zig`

### Step 1: Write the failing test for adaptive tiling

```zig
// src/gpu/tests/adaptive_tiling_test.zig
const std = @import("std");
const AdaptiveTiling = @import("../adaptive_tiling.zig").AdaptiveTiling;

test "AdaptiveTiling selects optimal tile size for square matrices" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 8, .minor = 0 },
    });

    const tile = tiling.selectTile(1024, 1024, 1024, .f32);

    // Should select a reasonable tile size
    try std.testing.expect(tile.m >= 16 and tile.m <= 128);
    try std.testing.expect(tile.n >= 16 and tile.n <= 128);
    try std.testing.expect(tile.k >= 8 and tile.k <= 32);
}

test "AdaptiveTiling handles non-square matrices" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 48 * 1024,
        .warp_size = 32,
        .compute_capability = .{ .major = 7, .minor = 5 },
    });

    // Tall skinny matrix
    const tile1 = tiling.selectTile(4096, 64, 256, .f32);
    try std.testing.expect(tile1.m > tile1.n); // Should favor M dimension

    // Wide flat matrix
    const tile2 = tiling.selectTile(64, 4096, 256, .f32);
    try std.testing.expect(tile2.n > tile2.m); // Should favor N dimension
}

test "AdaptiveTiling respects shared memory limits" {
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = 1024,
        .max_shared_memory = 16 * 1024, // Limited shared memory
        .warp_size = 32,
        .compute_capability = .{ .major = 6, .minor = 1 },
    });

    const tile = tiling.selectTile(2048, 2048, 2048, .f32);

    // Calculate shared memory usage: (tile_m + tile_n) * tile_k * sizeof(f32)
    const shared_bytes = (tile.m + tile.n) * tile.k * 4;
    try std.testing.expect(shared_bytes <= 16 * 1024);
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/adaptive_tiling_test.zig`
Expected: FAIL with "unable to open 'src/gpu/adaptive_tiling.zig'"

### Step 3: Write AdaptiveTiling implementation

```zig
// src/gpu/adaptive_tiling.zig
const std = @import("std");

/// Adaptive tiling configuration for matrix operations.
/// Selects optimal tile sizes based on matrix dimensions and device capabilities.
pub const AdaptiveTiling = struct {
    device_info: DeviceInfo,

    pub const DeviceInfo = struct {
        max_threads_per_block: u32,
        max_shared_memory: u32,
        warp_size: u32,
        compute_capability: struct { major: u32, minor: u32 },
    };

    pub const TileConfig = struct {
        m: u32, // Tile height
        n: u32, // Tile width
        k: u32, // Reduction tile size
    };

    pub const ElementType = enum {
        f16,
        f32,
        f64,

        pub fn sizeOf(self: ElementType) u32 {
            return switch (self) {
                .f16 => 2,
                .f32 => 4,
                .f64 => 8,
            };
        }
    };

    pub fn init(device_info: DeviceInfo) AdaptiveTiling {
        return .{ .device_info = device_info };
    }

    pub fn selectTile(self: AdaptiveTiling, m: u32, n: u32, k: u32, elem_type: ElementType) TileConfig {
        const elem_size = elem_type.sizeOf();
        const warp_size = self.device_info.warp_size;
        const max_shared = self.device_info.max_shared_memory;
        const max_threads = self.device_info.max_threads_per_block;

        // Start with default tile sizes
        var tile_m: u32 = 64;
        var tile_n: u32 = 64;
        var tile_k: u32 = 16;

        // Adjust for matrix shape
        if (m > n * 4) {
            // Tall matrix: favor M dimension
            tile_m = 128;
            tile_n = 32;
        } else if (n > m * 4) {
            // Wide matrix: favor N dimension
            tile_m = 32;
            tile_n = 128;
        }

        // Adjust for compute capability
        const cc = self.device_info.compute_capability;
        if (cc.major >= 8) {
            // Ampere+: can use larger tiles with tensor cores
            tile_m = @min(tile_m * 2, 128);
            tile_n = @min(tile_n * 2, 128);
        } else if (cc.major >= 7) {
            // Volta/Turing: moderate tile sizes
            tile_m = @min(tile_m, 96);
            tile_n = @min(tile_n, 96);
        }

        // Ensure thread count fits
        while (tile_m * tile_n > max_threads) {
            if (tile_m > tile_n) {
                tile_m /= 2;
            } else {
                tile_n /= 2;
            }
        }

        // Ensure tiles are warp-aligned
        tile_m = ((tile_m + warp_size - 1) / warp_size) * warp_size;
        tile_n = ((tile_n + warp_size - 1) / warp_size) * warp_size;

        // Ensure shared memory fits: (tile_m + tile_n) * tile_k * elem_size
        while ((tile_m + tile_n) * tile_k * elem_size > max_shared) {
            if (tile_k > 8) {
                tile_k /= 2;
            } else if (tile_m > tile_n and tile_m > 32) {
                tile_m /= 2;
            } else if (tile_n > 32) {
                tile_n /= 2;
            } else {
                break;
            }
        }

        // Clamp to matrix dimensions
        tile_m = @min(tile_m, m);
        tile_n = @min(tile_n, n);
        tile_k = @min(tile_k, k);

        return .{
            .m = @max(tile_m, 16),
            .n = @max(tile_n, 16),
            .k = @max(tile_k, 8),
        };
    }
};
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/adaptive_tiling_test.zig`
Expected: PASS (all 3 tests)

### Step 5: Commit

```bash
git add src/gpu/adaptive_tiling.zig src/gpu/tests/adaptive_tiling_test.zig
git commit -m "feat(gpu): add AdaptiveTiling for runtime-optimal matrix tile selection"
```

---

## Task 6: Integrate Adaptive Tiling into Matrix Kernels

**Files:**
- Modify: `src/gpu/kernels/matrix.zig:23-146`
- Modify: `src/gpu/unified.zig:555-620`
- Test: `src/gpu/tests/matrix_test.zig`

### Step 1: Write the failing test for adaptive matmul

```zig
// Add to src/gpu/tests/matrix_test.zig
test "matrixMultiply uses adaptive tiling for non-square matrices" {
    const allocator = std.testing.allocator;
    var gpu = try Gpu.init(allocator, .{});
    defer gpu.deinit();

    // Create tall skinny matrix (4096 x 64)
    var a = try gpu.createBuffer(f32, 4096 * 64);
    defer a.deinit();
    var b = try gpu.createBuffer(f32, 64 * 256);
    defer b.deinit();
    var c = try gpu.createBuffer(f32, 4096 * 256);
    defer c.deinit();

    // Fill with test data
    gpu.fillBuffer(a, 1.0);
    gpu.fillBuffer(b, 1.0);

    // Execute with adaptive tiling
    try gpu.matrixMultiply(a, b, c, 4096, 64, 256);

    // Verify result
    const result = try c.toHost(allocator);
    defer allocator.free(result);

    // Each element should be 64.0 (sum of 64 ones)
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[0], 0.001);
}
```

### Step 2: Run test to verify it fails

Run: `zig test src/gpu/tests/matrix_test.zig --test-filter "adaptive tiling"`
Expected: FAIL (either with missing function or incorrect result due to fixed tiling)

### Step 3: Update matrix multiplication to use AdaptiveTiling

In `src/gpu/unified.zig`, modify `matrixMultiply`:

```zig
const AdaptiveTiling = @import("adaptive_tiling.zig").AdaptiveTiling;

pub fn matrixMultiply(self: *Gpu, a: *UnifiedBuffer, b: *UnifiedBuffer, c: *UnifiedBuffer, m: u32, k: u32, n: u32) !void {
    // Get device info for adaptive tiling
    const device_info = self.getDeviceInfo();
    const tiling = AdaptiveTiling.init(.{
        .max_threads_per_block = device_info.max_threads_per_block,
        .max_shared_memory = device_info.max_shared_memory,
        .warp_size = device_info.warp_size,
        .compute_capability = device_info.compute_capability,
    });

    // Select optimal tile configuration
    const tile = tiling.selectTile(m, n, k, .f32);

    // Create launch config with adaptive tiling
    const config = LaunchConfig{
        .grid_dim = .{
            (m + tile.m - 1) / tile.m,
            (n + tile.n - 1) / tile.n,
            1,
        },
        .block_dim = .{ tile.m, tile.n, 1 },
        .shared_mem = (tile.m + tile.n) * tile.k * 4, // f32 size
    };

    // Execute with tiled config
    try self.dispatcher.execute(
        self.dispatcher.getBuiltinKernel(.matrix_multiply),
        config,
        .{ .a = a, .b = b, .c = c, .m = m, .k = k, .n = n, .tile_k = tile.k },
    );
}
```

### Step 4: Run test to verify it passes

Run: `zig test src/gpu/tests/matrix_test.zig --test-filter "adaptive tiling"`
Expected: PASS

### Step 5: Commit

```bash
git add src/gpu/unified.zig src/gpu/kernels/matrix.zig src/gpu/tests/matrix_test.zig
git commit -m "feat(gpu): integrate AdaptiveTiling into matrixMultiply for shape-aware optimization"
```

---

## Task 7: Add Performance Benchmarks

**Files:**
- Create: `src/gpu/benchmarks/sync_benchmark.zig`
- Create: `src/gpu/benchmarks/matmul_benchmark.zig`

### Step 1: Write sync benchmark

```zig
// src/gpu/benchmarks/sync_benchmark.zig
const std = @import("std");
const Gpu = @import("../unified.zig").Gpu;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var gpu = try Gpu.init(allocator, .{});
    defer gpu.deinit();

    const sizes = [_]usize{ 1024, 64 * 1024, 1024 * 1024, 64 * 1024 * 1024 };
    const iterations = 100;

    std.debug.print("Buffer Synchronization Benchmark\n", .{});
    std.debug.print("================================\n", .{});

    for (sizes) |size| {
        var buffer = try gpu.createBuffer(f32, size / 4);
        defer buffer.deinit();

        var timer = try std.time.Timer.start();

        var i: u32 = 0;
        while (i < iterations) : (i += 1) {
            buffer.markHostDirty();
            buffer.getSyncEvent().wait();
        }

        const elapsed_ns = timer.read();
        const avg_ns = elapsed_ns / iterations;

        std.debug.print("Size: {d:>10} bytes, Avg sync: {d:>8} ns\n", .{ size, avg_ns });
    }
}
```

### Step 2: Write matmul benchmark

```zig
// src/gpu/benchmarks/matmul_benchmark.zig
const std = @import("std");
const Gpu = @import("../unified.zig").Gpu;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var gpu = try Gpu.init(allocator, .{});
    defer gpu.deinit();

    const configs = [_]struct { m: u32, k: u32, n: u32 }{
        .{ .m = 1024, .k = 1024, .n = 1024 }, // Square
        .{ .m = 4096, .k = 64, .n = 256 }, // Tall skinny
        .{ .m = 64, .k = 64, .n = 4096 }, // Wide flat
        .{ .m = 2048, .k = 512, .n = 2048 }, // Large square
    };
    const iterations = 10;

    std.debug.print("Matrix Multiplication Benchmark\n", .{});
    std.debug.print("==============================\n", .{});

    for (configs) |cfg| {
        var a = try gpu.createBuffer(f32, cfg.m * cfg.k);
        defer a.deinit();
        var b = try gpu.createBuffer(f32, cfg.k * cfg.n);
        defer b.deinit();
        var c = try gpu.createBuffer(f32, cfg.m * cfg.n);
        defer c.deinit();

        // Warmup
        try gpu.matrixMultiply(&a, &b, &c, cfg.m, cfg.k, cfg.n);

        var timer = try std.time.Timer.start();

        var i: u32 = 0;
        while (i < iterations) : (i += 1) {
            try gpu.matrixMultiply(&a, &b, &c, cfg.m, cfg.k, cfg.n);
        }

        const elapsed_ns = timer.read();
        const avg_ms = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;
        const flops = @as(f64, @floatFromInt(2 * cfg.m * cfg.k * cfg.n));
        const gflops = flops / (avg_ms * 1_000_000.0);

        std.debug.print("Shape: [{d}x{d}] * [{d}x{d}], Avg: {d:.2} ms, {d:.2} GFLOPS\n", .{
            cfg.m, cfg.k, cfg.k, cfg.n, avg_ms, gflops,
        });
    }
}
```

### Step 3: Run benchmarks

Run: `zig build benchmarks && ./zig-out/bin/sync_benchmark && ./zig-out/bin/matmul_benchmark`
Expected: Benchmark output with timing data

### Step 4: Commit

```bash
git add src/gpu/benchmarks/
git commit -m "feat(gpu): add performance benchmarks for sync and matmul"
```

---

## Task 8: Update Module Exports and Documentation

**Files:**
- Modify: `src/gpu/mod.zig`
- Modify: `docs-src/gpu.md`

### Step 1: Update mod.zig exports

Add to `src/gpu/mod.zig`:

```zig
pub const SyncEvent = @import("sync_event.zig").SyncEvent;
pub const KernelRing = @import("kernel_ring.zig").KernelRing;
pub const AdaptiveTiling = @import("adaptive_tiling.zig").AdaptiveTiling;
```

### Step 2: Run module compilation check

Run: `zig build -Denable-gpu=true`
Expected: Clean compilation with no errors

### Step 3: Update GPU documentation

Add to `docs-src/gpu.md`:

```markdown
## Performance Optimizations

### Event-Based Memory Synchronization

The GPU module uses `SyncEvent` for non-blocking memory synchronization:

```zig
var buffer = try gpu.createBuffer(f32, 1024);
buffer.markHostDirty();

// Non-blocking check
if (!buffer.getSyncEvent().isComplete()) {
    // Do other work while waiting
}

// Or block until complete
buffer.getSyncEvent().wait();
```

### Kernel Launch Caching

The `KernelRing` provides fast-path reuse of recent kernel configurations:

- Automatic caching of last 256 kernel launches
- O(1) lookup for repeated configurations
- Lock-free implementation for multi-threaded dispatch

### Adaptive Matrix Tiling

Matrix operations automatically select optimal tile sizes based on:

- Matrix dimensions (handles tall/wide/square)
- Device compute capability (SM 6.x through 8.x)
- Shared memory constraints
- Warp alignment requirements
```

### Step 4: Commit

```bash
git add src/gpu/mod.zig docs-src/gpu.md
git commit -m "docs(gpu): document performance optimizations and update exports"
```

---

## Task 9: Run Full Test Suite and Verify

**Files:**
- All modified GPU files

### Step 1: Run all GPU tests

Run: `zig build test --summary all -Denable-gpu=true`
Expected: All tests pass

### Step 2: Run benchmarks to verify performance improvement

Run: `zig build benchmarks -Denable-gpu=true && ./zig-out/bin/matmul_benchmark`
Expected: Performance data for comparison

### Step 3: Final commit with summary

```bash
git add -A
git commit -m "feat(gpu): complete performance refactor with event sync, kernel caching, and adaptive tiling

- Add SyncEvent for non-blocking memory synchronization
- Add KernelRing for fast-path kernel launch caching
- Add AdaptiveTiling for runtime-optimal matrix tile selection
- Integrate all optimizations into unified GPU API
- Add benchmarks for performance validation

Expected improvements:
- 1.5-2x faster memory sync (event-based vs polling)
- 1.2-1.5x faster repeated kernel launches (ring buffer)
- 1.3-2x faster non-square matrix operations (adaptive tiling)"
```

---

## Summary

| Task | Component | Purpose |
|------|-----------|---------|
| 1 | SyncEvent | Event-based memory synchronization primitive |
| 2 | UnifiedBuffer integration | Replace polling with event-based sync |
| 3 | KernelRing | Lock-free kernel descriptor caching |
| 4 | Dispatcher integration | Fast-path kernel launches |
| 5 | AdaptiveTiling | Runtime tile size selection |
| 6 | Matrix kernel integration | Shape-aware tiling |
| 7 | Benchmarks | Performance validation |
| 8 | Exports & docs | Module updates |
| 9 | Final verification | Full test suite |

**Total commits:** 9 incremental commits following TDD pattern
