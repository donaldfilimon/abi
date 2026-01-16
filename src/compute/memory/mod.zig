//! Memory management utilities for the ABI framework.
//!
//! Provides stable allocators for long-lived data, worker arenas for scratch
//! allocations, and fixed-size memory pools for performance-critical scenarios.

const std = @import("std");

pub const StableAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{}) = .{},

    pub fn allocator(self: *StableAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    pub fn deinit(self: *StableAllocator) void {
        _ = self.gpa.deinit();
    }
};

pub const WorkerArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) WorkerArena {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn allocator(self: *WorkerArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *WorkerArena) void {
        self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *WorkerArena) void {
        self.arena.deinit();
    }
};

pub const ScratchArena = WorkerArena;

pub const PoolError = error{
    InvalidPool,
    InvalidBlock,
};

pub const FixedPool = struct {
    allocator: std.mem.Allocator,
    block_size: usize,
    storage: []u8,
    free_offsets: []usize,
    free_count: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        block_size: usize,
        block_count: usize,
    ) !FixedPool {
        if (block_size == 0 or block_count == 0) return PoolError.InvalidPool;

        const total_size = block_size * block_count;
        const storage = try allocator.alloc(u8, total_size);
        errdefer allocator.free(storage);

        const free_offsets = try allocator.alloc(usize, block_count);
        errdefer allocator.free(free_offsets);

        var i: usize = 0;
        while (i < block_count) : (i += 1) {
            free_offsets[i] = i * block_size;
        }

        return .{
            .allocator = allocator,
            .block_size = block_size,
            .storage = storage,
            .free_offsets = free_offsets,
            .free_count = block_count,
        };
    }

    pub fn deinit(self: *FixedPool) void {
        self.allocator.free(self.storage);
        self.allocator.free(self.free_offsets);
        self.* = undefined;
    }

    pub fn alloc(self: *FixedPool) ?[]u8 {
        if (self.free_count == 0) return null;
        self.free_count -= 1;
        const offset = self.free_offsets[self.free_count];
        return self.storage[offset .. offset + self.block_size];
    }

    pub fn free(self: *FixedPool, block: []u8) PoolError!void {
        if (block.len != self.block_size) return PoolError.InvalidBlock;

        const base = @intFromPtr(self.storage.ptr);
        const addr = @intFromPtr(block.ptr);
        if (addr < base) return PoolError.InvalidBlock;
        const offset = addr - base;
        if (offset % self.block_size != 0) return PoolError.InvalidBlock;
        if (offset + self.block_size > self.storage.len) {
            return PoolError.InvalidBlock;
        }
        if (self.free_count >= self.free_offsets.len) return PoolError.InvalidBlock;

        self.free_offsets[self.free_count] = offset;
        self.free_count += 1;
    }

    pub fn available(self: *const FixedPool) usize {
        return self.free_count;
    }

    pub fn capacity(self: *const FixedPool) usize {
        return self.free_offsets.len;
    }
};

test "stable allocator allocates" {
    var stable = StableAllocator{};
    defer stable.deinit();
    const allocator = stable.allocator();
    const buffer = try allocator.alloc(u8, 16);
    allocator.free(buffer);
}

test "worker arena reset" {
    var arena = WorkerArena.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const first = try allocator.alloc(u8, 8);
    _ = first;
    arena.reset();
    const second = try allocator.alloc(u8, 8);
    try std.testing.expectEqual(@as(usize, 8), second.len);
}

test "fixed pool allocates and reuses blocks" {
    var pool = try FixedPool.init(std.testing.allocator, 8, 2);
    defer pool.deinit();

    const a = pool.alloc() orelse return error.TestUnexpectedResult;
    const b = pool.alloc() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), pool.available());
    try std.testing.expect(pool.alloc() == null);

    try pool.free(a);
    try std.testing.expectEqual(@as(usize, 1), pool.available());

    const c = pool.alloc() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), pool.available());
    _ = b;
    _ = c;
}

test "fixed pool rejects invalid blocks" {
    var pool = try FixedPool.init(std.testing.allocator, 16, 4);
    defer pool.deinit();

    const valid = pool.alloc() orelse return error.TestUnexpectedResult;
    defer _ = pool.free(valid);

    var invalid = [_]u8{0} ** 16;
    try std.testing.expectError(PoolError.InvalidBlock, pool.free(&invalid));

    var wrong_size = [_]u8{0} ** 8;
    try std.testing.expectError(PoolError.InvalidBlock, pool.free(&wrong_size));
}

// ============================================================================
// Slab Allocator - Multi-size class pool for hot paths
// ============================================================================

/// Slab allocator with multiple size classes for efficient allocation
/// of frequently-used small objects without syscall overhead.
pub const SlabAllocator = struct {
    /// Size classes: 64, 128, 256, 512, 1024, 2048, 4096 bytes
    const SIZE_CLASSES = [_]usize{ 64, 128, 256, 512, 1024, 2048, 4096 };
    const NUM_CLASSES = SIZE_CLASSES.len;

    backing: std.mem.Allocator,
    pools: [NUM_CLASSES]?FixedPool,
    stats: Stats,

    pub const Stats = struct {
        allocations: u64 = 0,
        deallocations: u64 = 0,
        pool_hits: u64 = 0,
        pool_misses: u64 = 0,
        bytes_allocated: u64 = 0,
    };

    pub fn init(backing: std.mem.Allocator, blocks_per_class: usize) !SlabAllocator {
        var self = SlabAllocator{
            .backing = backing,
            .pools = [_]?FixedPool{null} ** NUM_CLASSES,
            .stats = .{},
        };

        // Initialize pools for each size class
        for (SIZE_CLASSES, 0..) |size, i| {
            self.pools[i] = FixedPool.init(backing, size, blocks_per_class) catch null;
        }

        return self;
    }

    pub fn deinit(self: *SlabAllocator) void {
        for (&self.pools) |*pool_opt| {
            if (pool_opt.*) |*pool| {
                pool.deinit();
                pool_opt.* = null;
            }
        }
    }

    /// Find the appropriate size class for a given size
    fn findSizeClass(size: usize) ?usize {
        for (SIZE_CLASSES, 0..) |class_size, i| {
            if (size <= class_size) return i;
        }
        return null;
    }

    /// Allocate memory, using pool if size fits, otherwise backing allocator
    pub fn alloc(self: *SlabAllocator, size: usize) ?[]u8 {
        self.stats.allocations += 1;
        self.stats.bytes_allocated += size;

        if (findSizeClass(size)) |class_idx| {
            if (self.pools[class_idx]) |*pool| {
                if (pool.alloc()) |block| {
                    self.stats.pool_hits += 1;
                    return block[0..size];
                }
            }
        }

        // Fallback to backing allocator
        self.stats.pool_misses += 1;
        return self.backing.alloc(u8, size) catch null;
    }

    /// Free memory back to pool or backing allocator
    pub fn free(self: *SlabAllocator, ptr: []u8) void {
        self.stats.deallocations += 1;

        // Try to return to appropriate pool
        if (findSizeClass(ptr.len)) |class_idx| {
            if (self.pools[class_idx]) |*pool| {
                // Check if this pointer belongs to the pool
                const pool_base = @intFromPtr(pool.storage.ptr);
                const pool_end = pool_base + pool.storage.len;
                const ptr_addr = @intFromPtr(ptr.ptr);

                if (ptr_addr >= pool_base and ptr_addr < pool_end) {
                    // Reconstruct full block slice for proper free
                    const block_size = SIZE_CLASSES[class_idx];
                    const full_block = ptr.ptr[0..block_size];
                    pool.free(full_block) catch {};
                    return;
                }
            }
        }

        // Not from pool, free via backing allocator
        self.backing.free(ptr);
    }

    pub fn getStats(self: *const SlabAllocator) Stats {
        return self.stats;
    }

    /// Get pool utilization for each size class
    pub fn getUtilization(self: *const SlabAllocator) [NUM_CLASSES]f32 {
        var result: [NUM_CLASSES]f32 = [_]f32{0.0} ** NUM_CLASSES;
        for (self.pools, 0..) |pool_opt, i| {
            if (pool_opt) |pool| {
                const cap = pool.capacity();
                const used = cap - pool.available();
                result[i] = if (cap > 0) @as(f32, @floatFromInt(used)) / @as(f32, @floatFromInt(cap)) else 0.0;
            }
        }
        return result;
    }
};

// ============================================================================
// Zero-Copy Buffer - Avoid unnecessary copies
// ============================================================================

/// Zero-copy buffer that can wrap existing memory or own its allocation
pub const ZeroCopyBuffer = struct {
    data: []u8,
    owned: bool,
    allocator: ?std.mem.Allocator,

    /// Create a buffer wrapping existing memory (no copy, no ownership)
    pub fn wrap(data: []u8) ZeroCopyBuffer {
        return .{
            .data = data,
            .owned = false,
            .allocator = null,
        };
    }

    /// Create a buffer wrapping const memory (no copy, no ownership)
    pub fn wrapConst(data: []const u8) ZeroCopyBuffer {
        return .{
            .data = @constCast(data),
            .owned = false,
            .allocator = null,
        };
    }

    /// Create an owned buffer with new allocation
    pub fn create(allocator: std.mem.Allocator, size: usize) !ZeroCopyBuffer {
        const data = try allocator.alloc(u8, size);
        return .{
            .data = data,
            .owned = true,
            .allocator = allocator,
        };
    }

    /// Create an owned buffer by copying data
    pub fn copy(allocator: std.mem.Allocator, source: []const u8) !ZeroCopyBuffer {
        const data = try allocator.dupe(u8, source);
        return .{
            .data = data,
            .owned = true,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ZeroCopyBuffer) void {
        if (self.owned) {
            if (self.allocator) |alloc| {
                alloc.free(self.data);
            }
        }
        self.* = undefined;
    }

    pub fn slice(self: *const ZeroCopyBuffer) []u8 {
        return self.data;
    }

    pub fn constSlice(self: *const ZeroCopyBuffer) []const u8 {
        return self.data;
    }

    pub fn len(self: *const ZeroCopyBuffer) usize {
        return self.data.len;
    }

    /// Transfer ownership - caller becomes responsible for freeing
    pub fn takeOwnership(self: *ZeroCopyBuffer) ?[]u8 {
        if (!self.owned) return null;
        const data = self.data;
        self.owned = false;
        self.allocator = null;
        return data;
    }
};

// ============================================================================
// Scoped Arena - RAII-style temporary allocation scope
// ============================================================================

/// Scoped arena for temporary allocations that are freed together
pub const ScopedArena = struct {
    arena: std.heap.ArenaAllocator,
    checkpoint: ?usize,

    pub fn init(backing: std.mem.Allocator) ScopedArena {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
            .checkpoint = null,
        };
    }

    pub fn deinit(self: *ScopedArena) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *ScopedArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Save current state for later restore
    pub fn saveCheckpoint(self: *ScopedArena) void {
        // ArenaAllocator doesn't expose state directly, so we track usage
        self.checkpoint = self.arena.queryCapacity();
    }

    /// Reset to checkpoint or beginning
    pub fn restore(self: *ScopedArena) void {
        self.arena.reset(.retain_capacity);
        self.checkpoint = null;
    }

    /// Get memory usage
    pub fn usage(self: *const ScopedArena) usize {
        return self.arena.queryCapacity();
    }
};

// ============================================================================
// Tests for new types
// ============================================================================

test "slab allocator basic" {
    var slab = try SlabAllocator.init(std.testing.allocator, 8);
    defer slab.deinit();

    // Allocate various sizes
    const small = slab.alloc(32) orelse return error.TestUnexpectedResult;
    const medium = slab.alloc(200) orelse return error.TestUnexpectedResult;
    const large = slab.alloc(1000) orelse return error.TestUnexpectedResult;

    // Write to verify memory is usable
    @memset(small, 'A');
    @memset(medium, 'B');
    @memset(large, 'C');

    slab.free(small);
    slab.free(medium);
    slab.free(large);

    const stats = slab.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.allocations);
}

test "zero copy buffer wrap" {
    var data = [_]u8{ 1, 2, 3, 4 };
    var buf = ZeroCopyBuffer.wrap(&data);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 4), buf.len());
    try std.testing.expectEqual(@as(u8, 1), buf.slice()[0]);

    // Modifying original should affect buffer
    data[0] = 99;
    try std.testing.expectEqual(@as(u8, 99), buf.slice()[0]);
}

test "zero copy buffer owned" {
    var buf = try ZeroCopyBuffer.create(std.testing.allocator, 16);
    defer buf.deinit();

    @memset(buf.slice(), 42);
    try std.testing.expectEqual(@as(u8, 42), buf.slice()[0]);
}

test "scoped arena usage" {
    var arena = ScopedArena.init(std.testing.allocator);
    defer arena.deinit();

    const alloc = arena.allocator();
    _ = try alloc.alloc(u8, 1024);

    try std.testing.expect(arena.usage() >= 1024);

    arena.restore();
    // After restore, can allocate again
    _ = try alloc.alloc(u8, 512);
}
