// ============================================================================
// ABI Framework — Composable Allocator Combinators
// Adapted from abi-system-v2.0/alloc.zig
// ============================================================================
//
// Zero-cost allocator wrappers that compose via std.mem.Allocator interface.
// Each combinator adds a single concern:
//
//   TrackingAllocator  — Counts allocations, bytes, and peak usage
//   LimitingAllocator  — Enforces a hard memory ceiling
//   FallbackAllocator  — Primary → fallback chain
//   NullAllocator      — Always fails (sentinel / test stub)
//
// All vtables use the Zig 0.16 5-parameter signature:
//   alloc(ctx, len, alignment, ret_addr) -> ?[*]u8
//   resize(ctx, memory: []u8, alignment, new_len, ret_addr) -> bool
//   remap(ctx, memory: []u8, alignment, new_len, ret_addr) -> ?[*]u8
//   free(ctx, memory: []u8, alignment, ret_addr) -> void
// ============================================================================

const std = @import("std");

// ─── Tracking Allocator ────────────────────────────────────────────────────

/// Wraps an allocator and records allocation statistics.
/// Thread-safe via atomic counters.
pub const TrackingAllocator = struct {
    backing: std.mem.Allocator,

    alloc_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    free_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    bytes_allocated: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    bytes_freed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    peak_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn init(backing: std.mem.Allocator) TrackingAllocator {
        return .{ .backing = backing };
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawAlloc(len, alignment, ret_addr);
        if (result != null) {
            _ = self.alloc_count.fetchAdd(1, .monotonic);
            const new_bytes = self.bytes_allocated.fetchAdd(len, .monotonic) + len;
            const freed = self.bytes_freed.load(.monotonic);
            const current = new_bytes -| freed;
            // Update peak (monotonic CAS loop)
            var peak = self.peak_bytes.load(.monotonic);
            while (current > peak) {
                if (self.peak_bytes.cmpxchgWeak(peak, current, .monotonic, .monotonic)) |p| {
                    peak = p;
                } else break;
            }
        }
        return result;
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        _ = self.free_count.fetchAdd(1, .monotonic);
        _ = self.bytes_freed.fetchAdd(memory.len, .monotonic);
        self.backing.rawFree(memory, alignment, ret_addr);
    }

    // ── Statistics ──────────────────────────────────────────────

    pub const Stats = struct {
        alloc_count: u64,
        free_count: u64,
        bytes_allocated: u64,
        bytes_freed: u64,
        bytes_live: u64,
        peak_bytes: u64,
    };

    pub fn stats(self: *const TrackingAllocator) Stats {
        const allocated = self.bytes_allocated.load(.acquire);
        const freed = self.bytes_freed.load(.acquire);
        return .{
            .alloc_count = self.alloc_count.load(.acquire),
            .free_count = self.free_count.load(.acquire),
            .bytes_allocated = allocated,
            .bytes_freed = freed,
            .bytes_live = if (allocated > freed) allocated - freed else 0,
            .peak_bytes = self.peak_bytes.load(.acquire),
        };
    }

    pub fn reset(self: *TrackingAllocator) void {
        self.alloc_count.store(0, .release);
        self.free_count.store(0, .release);
        self.bytes_allocated.store(0, .release);
        self.bytes_freed.store(0, .release);
        self.peak_bytes.store(0, .release);
    }
};

// ─── Limiting Allocator ────────────────────────────────────────────────────

/// Enforces a maximum total allocation budget. Returns null when the
/// limit would be exceeded.
pub const LimitingAllocator = struct {
    backing: std.mem.Allocator,
    limit: usize,
    current: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn init(backing: std.mem.Allocator, limit: usize) LimitingAllocator {
        return .{ .backing = backing, .limit = limit };
    }

    pub fn allocator(self: *LimitingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *LimitingAllocator = @ptrCast(@alignCast(ctx));

        // Atomic check-and-reserve
        var cur = self.current.load(.acquire);
        while (true) {
            if (cur + len > self.limit) return null;
            if (self.current.cmpxchgWeak(cur, cur + len, .acq_rel, .acquire)) |c| {
                cur = c;
            } else break;
        }

        const result = self.backing.rawAlloc(len, alignment, ret_addr);
        if (result == null) {
            // Roll back reservation
            _ = self.current.fetchSub(len, .release);
        }
        return result;
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *LimitingAllocator = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len) {
            const diff = new_len - memory.len;
            const cur = self.current.load(.acquire);
            if (cur + diff > self.limit) return false;
        }
        return self.backing.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *LimitingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *LimitingAllocator = @ptrCast(@alignCast(ctx));
        _ = self.current.fetchSub(memory.len, .release);
        self.backing.rawFree(memory, alignment, ret_addr);
    }

    pub fn remaining(self: *const LimitingAllocator) usize {
        const cur = self.current.load(.acquire);
        return if (self.limit > cur) self.limit - cur else 0;
    }

    pub fn usage(self: *const LimitingAllocator) usize {
        return self.current.load(.acquire);
    }
};

// ─── Fallback Allocator ────────────────────────────────────────────────────

/// Tries the primary allocator first; falls back to the secondary on failure.
pub const FallbackAllocator = struct {
    primary: std.mem.Allocator,
    secondary: std.mem.Allocator,
    fallback_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn init(primary: std.mem.Allocator, secondary: std.mem.Allocator) FallbackAllocator {
        return .{ .primary = primary, .secondary = secondary };
    }

    pub fn allocator(self: *FallbackAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *FallbackAllocator = @ptrCast(@alignCast(ctx));

        if (self.primary.rawAlloc(len, alignment, ret_addr)) |result| {
            return result;
        }

        _ = self.fallback_count.fetchAdd(1, .monotonic);
        return self.secondary.rawAlloc(len, alignment, ret_addr);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        if (self.primary.rawResize(memory, alignment, new_len, ret_addr)) return true;
        return self.secondary.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        if (self.primary.rawRemap(memory, alignment, new_len, ret_addr)) |result| return result;
        return self.secondary.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *FallbackAllocator = @ptrCast(@alignCast(ctx));
        // Determine ownership: try resizing to 0 on primary. If it succeeds,
        // the primary owns the allocation; otherwise free from secondary.
        if (self.primary.rawResize(memory, alignment, 0, ret_addr)) {
            self.primary.rawFree(memory, alignment, ret_addr);
        } else {
            self.secondary.rawFree(memory, alignment, ret_addr);
        }
    }

    pub fn fallbackHits(self: *const FallbackAllocator) u64 {
        return self.fallback_count.load(.acquire);
    }
};

// ─── Null Allocator ────────────────────────────────────────────────────────

/// Always returns null / does nothing. Useful as a test sentinel or
/// terminal in an allocator chain.
pub const NullAllocator = struct {
    pub fn allocator() std.mem.Allocator {
        return .{
            .ptr = undefined,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(_: *anyopaque, _: usize, _: std.mem.Alignment, _: usize) ?[*]u8 {
        return null;
    }

    fn resizeFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remapFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn freeFn(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {}
};

// ─── Tests ──────────────────────────────────────────────────────────────────

test "TrackingAllocator counts allocs and frees" {
    var tracker = TrackingAllocator.init(std.testing.allocator);
    const alloc = tracker.allocator();

    const data = try alloc.alloc(u8, 100);
    const s1 = tracker.stats();
    try std.testing.expectEqual(@as(u64, 1), s1.alloc_count);
    try std.testing.expect(s1.bytes_allocated >= 100);
    try std.testing.expect(s1.bytes_live >= 100);

    alloc.free(data);
    const s2 = tracker.stats();
    try std.testing.expectEqual(@as(u64, 1), s2.free_count);
    try std.testing.expectEqual(@as(u64, 0), s2.bytes_live);
}

test "TrackingAllocator peak tracking" {
    var tracker = TrackingAllocator.init(std.testing.allocator);
    const alloc = tracker.allocator();

    const a = try alloc.alloc(u8, 200);
    const b = try alloc.alloc(u8, 300);
    // Peak should be >= 500
    try std.testing.expect(tracker.stats().peak_bytes >= 500);

    alloc.free(a);
    alloc.free(b);
    // Peak should still be >= 500 even after freeing
    try std.testing.expect(tracker.stats().peak_bytes >= 500);
}

test "LimitingAllocator enforces limit" {
    var limiter = LimitingAllocator.init(std.testing.allocator, 256);
    const alloc = limiter.allocator();

    // Should succeed — within limit
    const data = try alloc.alloc(u8, 100);
    defer alloc.free(data);

    try std.testing.expect(limiter.usage() >= 100);
    try std.testing.expect(limiter.remaining() <= 156);

    // Should fail — would exceed limit
    const result = alloc.alloc(u8, 200);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "LimitingAllocator frees restore capacity" {
    var limiter = LimitingAllocator.init(std.testing.allocator, 256);
    const alloc = limiter.allocator();

    const a = try alloc.alloc(u8, 128);
    try std.testing.expect(limiter.remaining() <= 128);
    alloc.free(a);
    try std.testing.expectEqual(@as(usize, 256), limiter.remaining());

    // Now we should be able to allocate the full limit again
    const b = try alloc.alloc(u8, 200);
    alloc.free(b);
}

test "NullAllocator always returns OutOfMemory" {
    const alloc = NullAllocator.allocator();
    try std.testing.expectError(error.OutOfMemory, alloc.alloc(u8, 1));
}

test "FallbackAllocator uses secondary on failure" {
    // Use NullAllocator as primary (always fails)
    var fb = FallbackAllocator.init(
        NullAllocator.allocator(),
        std.testing.allocator,
    );
    const alloc = fb.allocator();

    // Should succeed via secondary
    const data = try alloc.alloc(u8, 64);
    defer alloc.free(data);

    try std.testing.expect(fb.fallbackHits() >= 1);
}

test "TrackingAllocator reset" {
    var tracker = TrackingAllocator.init(std.testing.allocator);
    const alloc = tracker.allocator();

    const data = try alloc.alloc(u8, 50);
    alloc.free(data);

    tracker.reset();
    const s = tracker.stats();
    try std.testing.expectEqual(@as(u64, 0), s.alloc_count);
    try std.testing.expectEqual(@as(u64, 0), s.free_count);
    try std.testing.expectEqual(@as(u64, 0), s.peak_bytes);
}
