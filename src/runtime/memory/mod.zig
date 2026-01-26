//! Memory Management for Runtime
//!
//! This module provides memory management utilities:
//!
//! - `MemoryPool` / `FixedPool` - Fixed-size object pool
//! - `ArenaAllocator` / `WorkerArena` - Arena-style allocation
//! - `SlabAllocator` - Multi-size class pool for hot paths
//! - `ZeroCopyBuffer` - Avoid unnecessary copies
//! - `ScopedArena` - RAII-style temporary allocation scope
//! - Buffer management utilities

const std = @import("std");

// ============================================================================
// Stable Allocator - Long-lived allocations
// ============================================================================

pub const StableAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{}) = .{},

    pub fn allocator(self: *StableAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    pub fn deinit(self: *StableAllocator) void {
        _ = self.gpa.deinit();
    }
};

// ============================================================================
// Worker Arena - Scratch allocations
// ============================================================================

pub const WorkerArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) WorkerArena {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn allocator(self: *WorkerArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *WorkerArena) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *WorkerArena) void {
        self.arena.deinit();
    }
};

pub const ScratchArena = WorkerArena;
pub const ArenaAllocator = WorkerArena;

// ============================================================================
// Fixed Pool - Fixed-size block allocator
// ============================================================================

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

// Alias for compatibility
pub const MemoryPool = FixedPool;

// ============================================================================
// Slab Allocator - Multi-size class pool for hot paths
// ============================================================================

pub const SlabAllocator = struct {
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

    /// Binary search for optimal size class (O(log n) vs O(n) linear scan).
    fn findSizeClass(size: usize) ?usize {
        // Binary search for first size class >= size
        var left: usize = 0;
        var right: usize = SIZE_CLASSES.len;

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (SIZE_CLASSES[mid] < size) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return if (left < SIZE_CLASSES.len) left else null;
    }

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

        self.stats.pool_misses += 1;
        return self.backing.alloc(u8, size) catch null;
    }

    pub fn free(self: *SlabAllocator, ptr: []u8) void {
        self.stats.deallocations += 1;

        if (findSizeClass(ptr.len)) |class_idx| {
            if (self.pools[class_idx]) |*pool| {
                const pool_base = @intFromPtr(pool.storage.ptr);
                const pool_end = pool_base + pool.storage.len;
                const ptr_addr = @intFromPtr(ptr.ptr);

                if (ptr_addr >= pool_base and ptr_addr < pool_end) {
                    const block_size = SIZE_CLASSES[class_idx];
                    const full_block = ptr.ptr[0..block_size];
                    pool.free(full_block) catch {};
                    return;
                }
            }
        }

        self.backing.free(ptr);
    }

    pub fn getStats(self: *const SlabAllocator) Stats {
        return self.stats;
    }

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

// Alias for compatibility
pub const ObjectPool = SlabAllocator;

// ============================================================================
// Zero-Copy Buffer
// ============================================================================

pub const ZeroCopyBuffer = struct {
    data: []u8,
    owned: bool,
    allocator: ?std.mem.Allocator,

    pub fn wrap(data: []u8) ZeroCopyBuffer {
        return .{ .data = data, .owned = false, .allocator = null };
    }

    pub fn wrapConst(data: []const u8) ZeroCopyBuffer {
        return .{ .data = @constCast(data), .owned = false, .allocator = null };
    }

    pub fn create(alloc: std.mem.Allocator, size: usize) !ZeroCopyBuffer {
        const data = try alloc.alloc(u8, size);
        return .{ .data = data, .owned = true, .allocator = alloc };
    }

    pub fn copy(alloc: std.mem.Allocator, source: []const u8) !ZeroCopyBuffer {
        const data = try alloc.dupe(u8, source);
        return .{ .data = data, .owned = true, .allocator = alloc };
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

    pub fn takeOwnership(self: *ZeroCopyBuffer) ?[]u8 {
        if (!self.owned) return null;
        const data = self.data;
        self.owned = false;
        self.allocator = null;
        return data;
    }
};

// Aliases for compatibility
pub const RingBuffer = ZeroCopyBuffer;
pub const AlignedBuffer = ZeroCopyBuffer;
pub const MappedBuffer = ZeroCopyBuffer;

// ============================================================================
// Scoped Arena
// ============================================================================

pub const ScopedArena = struct {
    arena: std.heap.ArenaAllocator,
    checkpoint: ?usize,

    pub fn init(backing: std.mem.Allocator) ScopedArena {
        return .{ .arena = std.heap.ArenaAllocator.init(backing), .checkpoint = null };
    }

    pub fn deinit(self: *ScopedArena) void {
        self.arena.deinit();
    }

    pub fn allocator(self: *ScopedArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn saveCheckpoint(self: *ScopedArena) void {
        self.checkpoint = self.arena.queryCapacity();
    }

    pub fn restore(self: *ScopedArena) void {
        _ = self.arena.reset(.retain_capacity);
        self.checkpoint = null;
    }

    pub fn usage(self: *const ScopedArena) usize {
        return self.arena.queryCapacity();
    }
};

// ============================================================================
// Memory Stats and Tracking
// ============================================================================

pub const MemoryStats = struct {
    allocations: u64 = 0,
    deallocations: u64 = 0,
    bytes_allocated: u64 = 0,
    bytes_freed: u64 = 0,
    peak_usage: u64 = 0,
};

pub const AllocationTracker = struct {
    stats: MemoryStats = .{},

    pub fn recordAlloc(self: *AllocationTracker, size: usize) void {
        self.stats.allocations += 1;
        self.stats.bytes_allocated += size;
        const current = self.stats.bytes_allocated - self.stats.bytes_freed;
        if (current > self.stats.peak_usage) {
            self.stats.peak_usage = current;
        }
    }

    pub fn recordFree(self: *AllocationTracker, size: usize) void {
        self.stats.deallocations += 1;
        self.stats.bytes_freed += size;
    }

    pub fn getStats(self: *const AllocationTracker) MemoryStats {
        return self.stats;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MemoryPool basic allocation" {
    var pool = try MemoryPool.init(std.testing.allocator, 64, 8);
    defer pool.deinit();

    const ptr = pool.alloc();
    try std.testing.expect(ptr != null);
    try pool.free(ptr.?);
}

test "WorkerArena reset" {
    var arena = WorkerArena.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const first = try alloc.alloc(u8, 8);
    _ = first;
    arena.reset();
    const second = try alloc.alloc(u8, 8);
    try std.testing.expectEqual(@as(usize, 8), second.len);
}

test "SlabAllocator basic" {
    var slab = try SlabAllocator.init(std.testing.allocator, 8);
    defer slab.deinit();

    const small = slab.alloc(32) orelse return error.TestUnexpectedResult;
    @memset(small, 'A');
    slab.free(small);

    const stats = slab.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.allocations);
}

test "ZeroCopyBuffer wrap" {
    var data = [_]u8{ 1, 2, 3, 4 };
    var buf = ZeroCopyBuffer.wrap(&data);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 4), buf.len());
    data[0] = 99;
    try std.testing.expectEqual(@as(u8, 99), buf.slice()[0]);
}
