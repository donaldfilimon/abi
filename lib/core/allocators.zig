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
        .free = free,
    };
    
    fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        
        if (self.max_memory > 0 and self.stats.currentUsage() + len > self.max_memory) {
            return null;
        }
        
        const result = self.parent_allocator.rawAlloc(len, log2_ptr_align, ret_addr);
        if (result) |ptr| {
            self.stats.bytes_allocated += len;
            self.stats.active_allocations += 1;
            if (self.stats.currentUsage() > self.stats.peak_usage) {
                self.stats.peak_usage = self.stats.currentUsage();
            }
        }
        return result;
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;
        
        const result = self.parent_allocator.rawResize(buf, log2_buf_align, new_len, ret_addr);
        if (result) {
            if (new_len > old_len) {
                self.stats.bytes_allocated += new_len - old_len;
            } else {
                self.stats.bytes_freed += old_len - new_len;
            }
        }
        return result;
    }
    
    fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        
        self.stats.bytes_freed += buf.len;
        if (self.stats.active_allocations > 0) {
            self.stats.active_allocations -= 1;
        }
        
        self.parent_allocator.rawFree(buf, log2_buf_align, ret_addr);
    }
};

/// Allocator factory that creates allocators based on configuration
pub const AllocatorFactory = struct {
    /// Creates an allocator based on the provided configuration
    pub fn create(config: AllocatorConfig, parent_allocator: ?std.mem.Allocator) std.mem.Allocator {
        const parent = parent_allocator orelse std.heap.page_allocator;
        
        return switch (config.strategy) {
            .general_purpose => std.heap.GeneralPurposeAllocator(.{}).allocator(),
            .arena => std.heap.ArenaAllocator.init(parent).allocator(),
            .c => std.heap.c_allocator,
            .page => std.heap.page_allocator,
            .fixed_buffer => std.heap.FixedBufferAllocator.init(&[_]u8{}).allocator(),
        };
    }
    
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
    try std.testing.expect(memory2 == null);
}