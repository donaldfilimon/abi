//! Core Allocators Module
//!
//! Memory allocation utilities and patterns for the framework

const std = @import("std");

/// Memory allocation strategies available in the framework
pub const AllocationStrategy = enum {
    /// General purpose allocator (default)
    general_purpose,
    /// Arena allocator for batch operations
    arena,
    /// C allocator for C interop
    c,
    /// Page allocator for large allocations
    page,
    /// Fixed buffer allocator for small, predictable allocations
    fixed_buffer,
};

/// Allocator configuration
pub const AllocatorConfig = struct {
    /// The allocation strategy to use
    strategy: AllocationStrategy = .general_purpose,
    /// Maximum memory usage (0 = unlimited)
    max_memory: usize = 0,
    /// Enable memory tracking
    enable_tracking: bool = false,
};

/// Memory tracking information
pub const MemoryStats = struct {
    /// Total bytes allocated
    bytes_allocated: usize = 0,
    /// Total bytes freed
    bytes_freed: usize = 0,
    /// Current active allocations
    active_allocations: usize = 0,
    /// Peak memory usage
    peak_usage: usize = 0,

    /// Gets current memory usage
    pub fn currentUsage(self: MemoryStats) usize {
        return self.bytes_allocated - self.bytes_freed;
    }

    /// Checks if memory limit is exceeded
    pub fn isOverLimit(self: MemoryStats, limit: usize) bool {
        return self.currentUsage() > limit;
    }
};

/// Tracked allocator wrapper that monitors memory usage
pub const TrackedAllocator = struct {
    parent_allocator: std.mem.Allocator,
    stats: MemoryStats,
    max_memory: usize,

    const Self = @This();

    pub fn init(parent_allocator: std.mem.Allocator, max_memory: usize) Self {
        return Self{
            .parent_allocator = parent_allocator,
            .stats = MemoryStats{},
            .max_memory = max_memory,
        };
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    pub fn getStats(self: *const Self) MemoryStats {
        return self.stats;
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        if (self.max_memory > 0 and self.stats.currentUsage() + len > self.max_memory) {
            return null;
        }

        const result = self.parent_allocator.alloc(len, alignment.toLog2Units(), ret_addr);
        if (result) |_| {
            self.stats.bytes_allocated += len;
            self.stats.active_allocations += 1;
            if (self.stats.currentUsage() > self.stats.peak_usage) {
                self.stats.peak_usage = self.stats.currentUsage();
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;

        const result = self.parent_allocator.resize(buf, alignment.toLog2Units(), new_len, ret_addr);
        if (result) {
            if (new_len > old_len) {
                self.stats.bytes_allocated += new_len - old_len;
            } else {
                self.stats.bytes_freed += old_len - new_len;
            }
        }
        return result;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        if (self.max_memory > 0 and self.stats.currentUsage() - memory.len + new_len > self.max_memory) {
            return null;
        }

        const result = self.parent_allocator.rawRemap(memory, alignment, new_len, ret_addr);
        if (result) |_| {
            const old_len = memory.len;
            if (new_len > old_len) {
                self.stats.bytes_allocated += new_len - old_len;
            } else {
                self.stats.bytes_freed += old_len - new_len;
            }
            if (self.stats.currentUsage() > self.stats.peak_usage) {
                self.stats.peak_usage = self.stats.currentUsage();
            }
        }
        return result;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        self.stats.bytes_freed += buf.len;
        if (self.stats.active_allocations > 0) {
            self.stats.active_allocations -= 1;
        }

        self.parent_allocator.free(buf, alignment.toLog2Units(), ret_addr);
    }
};

/// Owned allocator that manages its own lifetime
pub const OwnedAllocator = struct {
    allocator: std.mem.Allocator,
    backing_allocator: union(enum) {
        none,
        gpa: std.heap.GeneralPurposeAllocator(.{}),
        arena: std.heap.ArenaAllocator,
    },

    /// Create an owned allocator with the specified strategy
    pub fn create(config: AllocatorConfig, parent_allocator: ?std.mem.Allocator) !OwnedAllocator {
        const parent = parent_allocator orelse std.heap.page_allocator;

        return switch (config.strategy) {
            .general_purpose => {
                var gpa = std.heap.GeneralPurposeAllocator(.{}){};
                return OwnedAllocator{
                    .allocator = gpa.allocator(),
                    .backing_allocator = .{ .gpa = gpa },
                };
            },
            .arena => {
                var arena = std.heap.ArenaAllocator.init(parent);
                return OwnedAllocator{
                    .allocator = arena.allocator(),
                    .backing_allocator = .{ .arena = arena },
                };
            },
            .c => OwnedAllocator{
                .allocator = std.heap.c_allocator,
                .backing_allocator = .none,
            },
            .page => OwnedAllocator{
                .allocator = std.heap.page_allocator,
                .backing_allocator = .none,
            },
            .fixed_buffer => OwnedAllocator{
                .allocator = std.heap.FixedBufferAllocator.init(&[_]u8{}).allocator(),
                .backing_allocator = .none,
            }, // initialized with empty buffer - allocations will fail until buffer is provided
        };
    }

    /// Deinitialize the owned allocator
    pub fn deinit(self: *OwnedAllocator) void {
        switch (self.backing_allocator) {
            .gpa => |*gpa| _ = gpa.deinit(),
            .arena => |*arena| arena.deinit(),
            .none => {},
        }
    }
};

/// Allocator factory that creates allocators based on configuration
/// ⚠️  DEPRECATED: This factory causes memory leaks and will be removed in a future version
/// ❌ DO NOT USE IN PRODUCTION CODE
/// ✅ Use OwnedAllocator.create instead for safe memory management
pub const AllocatorFactory = struct {
    /// Creates a tracked allocator
    pub fn createTracked(parent_allocator: std.mem.Allocator, max_memory: usize) TrackedAllocator {
        return TrackedAllocator.init(parent_allocator, max_memory);
    }
};

test "allocators - tracked allocator" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracked = AllocatorFactory.createTracked(gpa.allocator(), 1024);
    const allocator = tracked.allocator();

    const memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);

    const stats = tracked.getStats();
    try std.testing.expectEqual(@as(usize, 100), stats.bytes_allocated);
    try std.testing.expectEqual(@as(usize, 1), stats.active_allocations);
    try std.testing.expectEqual(@as(usize, 100), stats.currentUsage());
}

test "allocators - owned allocator no memory leaks" {
    var owned = try OwnedAllocator.create(.{ .strategy = .general_purpose }, null);
    defer owned.deinit();

    const allocator = owned.allocator;
    const memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);

    try std.testing.expectEqual(@as(usize, 100), memory.len);
}

test "allocators - memory limit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracked = AllocatorFactory.createTracked(gpa.allocator(), 50);
    const allocator = tracked.allocator();

    // This should succeed
    const memory1 = try allocator.alloc(u8, 30);
    defer allocator.free(memory1);

    // This should fail due to memory limit
    const memory2 = allocator.alloc(u8, 30);
    try std.testing.expectError(error.OutOfMemory, memory2);
}
