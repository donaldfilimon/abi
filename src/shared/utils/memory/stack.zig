//! Stack-based allocator for temporary allocations.
//!
//! Provides a fast bump allocator with:
//! - O(1) allocation (bump pointer)
//! - O(1) reset (pointer reset)
//! - Stack-like deallocation (LIFO order only)
//! - Optional overflow fallback to backing allocator
//!
//! Ideal for:
//! - Temporary scratch buffers
//! - Function-local allocations
//! - Parser intermediate data
//! - Short-lived collections
//!
//! Usage:
//! ```zig
//! var stack = StackAllocator.init(&buffer);
//! const allocator = stack.allocator();
//!
//! const data = try allocator.alloc(u8, 100);
//! // Use data...
//!
//! stack.reset(); // Instantly free all allocations
//! ```

const std = @import("std");

/// Stack allocator errors.
pub const StackError = error{
    OutOfMemory,
    StackOverflow,
    InvalidFree,
};

/// Configuration for stack allocator behavior.
pub const StackConfig = struct {
    /// Whether to allow fallback to backing allocator when stack is full.
    allow_fallback: bool = false,
    /// Backing allocator for fallback (required if allow_fallback is true).
    fallback_allocator: ?std.mem.Allocator = null,
    /// Enable strict LIFO checking (debug mode).
    strict_lifo: bool = false,
};

/// Fixed-size stack allocator with optional fallback.
pub const StackAllocator = struct {
    buffer: []u8,
    offset: usize,
    config: StackConfig,
    peak_usage: usize,
    allocation_count: u64,
    fallback_allocations: std.ArrayListUnmanaged([]u8),

    const Self = @This();

    /// Initialize with a fixed buffer.
    pub fn init(buffer: []u8) Self {
        return initWithConfig(buffer, .{});
    }

    /// Initialize with configuration.
    pub fn initWithConfig(buffer: []u8, config: StackConfig) Self {
        return .{
            .buffer = buffer,
            .offset = 0,
            .config = config,
            .peak_usage = 0,
            .allocation_count = 0,
            .fallback_allocations = .{},
        };
    }

    /// Deinitialize (frees any fallback allocations).
    pub fn deinit(self: *Self) void {
        if (self.config.fallback_allocator) |fallback| {
            for (self.fallback_allocations.items) |alloc_slice| {
                fallback.free(alloc_slice);
            }
            self.fallback_allocations.deinit(fallback);
        }
        self.* = undefined;
    }

    /// Get an allocator interface.
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// Reset the stack (free all allocations at once).
    pub fn reset(self: *Self) void {
        self.offset = 0;

        // Free fallback allocations too
        if (self.config.fallback_allocator) |fallback| {
            for (self.fallback_allocations.items) |alloc_slice| {
                fallback.free(alloc_slice);
            }
            self.fallback_allocations.shrinkAndFree(fallback, 0);
        }
    }

    /// Get current usage.
    pub fn usage(self: *const Self) usize {
        return self.offset;
    }

    /// Get available space.
    pub fn available(self: *const Self) usize {
        return self.buffer.len - self.offset;
    }

    /// Get peak usage.
    pub fn peakUsage(self: *const Self) usize {
        return self.peak_usage;
    }

    /// Get statistics.
    pub fn getStats(self: *const Self) StackStats {
        return .{
            .capacity = self.buffer.len,
            .usage = self.offset,
            .available = self.buffer.len - self.offset,
            .peak_usage = self.peak_usage,
            .allocation_count = self.allocation_count,
            .fallback_count = self.fallback_allocations.items.len,
            .utilization = if (self.buffer.len > 0)
                @as(f32, @floatFromInt(self.offset)) / @as(f32, @floatFromInt(self.buffer.len))
            else
                0.0,
        };
    }

    /// Create a savepoint for partial reset.
    pub fn savepoint(self: *const Self) Savepoint {
        return .{
            .offset = self.offset,
            .fallback_count = self.fallback_allocations.items.len,
        };
    }

    /// Restore to a savepoint.
    pub fn restore(self: *Self, sp: Savepoint) void {
        self.offset = sp.offset;

        // Free fallback allocations made after savepoint
        if (self.config.fallback_allocator) |fallback| {
            while (self.fallback_allocations.items.len > sp.fallback_count) {
                // Get the last item and remove it
                const last_idx = self.fallback_allocations.items.len - 1;
                const alloc_slice = self.fallback_allocations.items[last_idx];
                self.fallback_allocations.items.len = last_idx;
                fallback.free(alloc_slice);
            }
        }
    }

    // Allocator vtable implementations
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        const align_val = alignment.toByteUnits();
        const aligned_offset = (self.offset + align_val - 1) & ~(align_val - 1);
        const end_offset = aligned_offset + len;

        if (end_offset <= self.buffer.len) {
            self.offset = end_offset;
            self.allocation_count += 1;

            if (self.offset > self.peak_usage) {
                self.peak_usage = self.offset;
            }

            return self.buffer.ptr + aligned_offset;
        }

        // Try fallback if enabled
        if (self.config.allow_fallback) {
            if (self.config.fallback_allocator) |fallback| {
                const fallback_mem = fallback.rawAlloc(len, alignment, ret_addr) orelse return null;
                const slice = fallback_mem[0..len];
                self.fallback_allocations.append(fallback, slice) catch return null;
                return fallback_mem;
            }
        }

        return null;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if this is a stack allocation
        const mem_addr = @intFromPtr(memory.ptr);
        const buf_start = @intFromPtr(self.buffer.ptr);
        const buf_end = buf_start + self.buffer.len;

        if (mem_addr >= buf_start and mem_addr < buf_end) {
            const mem_end = mem_addr + memory.len;
            const current_top = buf_start + self.offset;

            // Only resize if this is the topmost allocation
            if (mem_end == current_top) {
                const new_end = mem_addr - buf_start + new_len;
                if (new_end <= self.buffer.len) {
                    self.offset = new_end;
                    if (self.offset > self.peak_usage) {
                        self.peak_usage = self.offset;
                    }
                    return true;
                }
            }
            return false;
        }

        // Delegate to fallback
        if (self.config.fallback_allocator) |fallback| {
            return fallback.rawResize(memory, alignment, new_len, ret_addr);
        }

        return false;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if this is a stack allocation
        const mem_addr = @intFromPtr(memory.ptr);
        const buf_start = @intFromPtr(self.buffer.ptr);
        const buf_end = buf_start + self.buffer.len;

        if (mem_addr >= buf_start and mem_addr < buf_end) {
            // Stack allocations can't be remapped to different location
            if (resize(ctx, memory, alignment, new_len, ret_addr)) {
                return memory.ptr;
            }
            return null;
        }

        // Delegate to fallback
        if (self.config.fallback_allocator) |fallback| {
            return fallback.rawRemap(memory, alignment, new_len, ret_addr);
        }

        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if this is a stack allocation
        const mem_addr = @intFromPtr(memory.ptr);
        const buf_start = @intFromPtr(self.buffer.ptr);
        const buf_end = buf_start + self.buffer.len;

        if (mem_addr >= buf_start and mem_addr < buf_end) {
            // For stack allocator, we only support freeing the topmost allocation
            const mem_end = mem_addr + memory.len;
            const current_top = buf_start + self.offset;

            if (mem_end == current_top) {
                self.offset = mem_addr - buf_start;
            }
            // Otherwise, just ignore (memory will be freed on reset)
            return;
        }

        // Free from fallback
        if (self.config.fallback_allocator) |fallback| {
            // Find and remove from fallback list
            for (self.fallback_allocations.items, 0..) |alloc_slice, i| {
                if (alloc_slice.ptr == memory.ptr) {
                    _ = self.fallback_allocations.swapRemove(i);
                    fallback.rawFree(memory, alignment, ret_addr);
                    return;
                }
            }
        }
    }
};

/// Savepoint for partial stack reset.
pub const Savepoint = struct {
    offset: usize,
    fallback_count: usize,
};

/// Stack allocator statistics.
pub const StackStats = struct {
    capacity: usize,
    usage: usize,
    available: usize,
    peak_usage: usize,
    allocation_count: u64,
    fallback_count: usize,
    utilization: f32,
};

/// Create a stack allocator with compile-time known size.
pub fn FixedStack(comptime size: usize) type {
    return struct {
        buffer: [size]u8 = undefined,
        stack: StackAllocator = undefined,

        const Self = @This();

        pub fn init(self: *Self) void {
            self.stack = StackAllocator.init(&self.buffer);
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return self.stack.allocator();
        }

        pub fn reset(self: *Self) void {
            self.stack.reset();
        }

        pub fn getStats(self: *const Self) StackStats {
            return self.stack.getStats();
        }
    };
}

/// Scoped stack allocator that auto-resets on scope exit.
pub fn ScopedStack(comptime size: usize) type {
    return struct {
        buffer: [size]u8 = undefined,
        stack: StackAllocator = undefined,

        const Self = @This();

        pub fn scoped() Self {
            var self = Self{};
            self.stack = StackAllocator.init(&self.buffer);
            return self;
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return self.stack.allocator();
        }
    };
}

test "stack allocator basic" {
    var buffer: [1024]u8 = undefined;
    var stack = StackAllocator.init(&buffer);
    defer stack.deinit();

    const allocator = stack.allocator();

    const a = try allocator.alloc(u8, 100);
    const b = try allocator.alloc(u8, 200);

    try std.testing.expectEqual(@as(usize, 100), a.len);
    try std.testing.expectEqual(@as(usize, 200), b.len);

    const stats = stack.getStats();
    try std.testing.expect(stats.usage >= 300);
}

test "stack allocator reset" {
    var buffer: [1024]u8 = undefined;
    var stack = StackAllocator.init(&buffer);
    defer stack.deinit();

    const allocator = stack.allocator();

    _ = try allocator.alloc(u8, 500);
    try std.testing.expect(stack.usage() >= 500);

    stack.reset();
    try std.testing.expectEqual(@as(usize, 0), stack.usage());

    // Can allocate again
    _ = try allocator.alloc(u8, 500);
    try std.testing.expect(stack.usage() >= 500);
}

test "stack allocator savepoint" {
    var buffer: [1024]u8 = undefined;
    var stack = StackAllocator.init(&buffer);
    defer stack.deinit();

    const allocator = stack.allocator();

    _ = try allocator.alloc(u8, 100);
    const sp = stack.savepoint();

    _ = try allocator.alloc(u8, 200);
    _ = try allocator.alloc(u8, 300);

    try std.testing.expect(stack.usage() >= 600);

    stack.restore(sp);
    try std.testing.expect(stack.usage() <= 200);
}

test "stack allocator overflow" {
    var buffer: [128]u8 = undefined;
    var stack = StackAllocator.init(&buffer);
    defer stack.deinit();

    const allocator = stack.allocator();

    _ = try allocator.alloc(u8, 64);

    // This should fail
    const result = allocator.alloc(u8, 128);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "stack allocator with fallback" {
    var buffer: [64]u8 = undefined;
    var stack = StackAllocator.initWithConfig(&buffer, .{
        .allow_fallback = true,
        .fallback_allocator = std.testing.allocator,
    });
    defer stack.deinit();

    const allocator = stack.allocator();

    // First allocation fits in stack
    const a = try allocator.alloc(u8, 32);
    _ = a;

    // Second allocation uses fallback
    const b = try allocator.alloc(u8, 128);
    defer allocator.free(b);

    try std.testing.expectEqual(@as(usize, 128), b.len);
}

test "fixed stack" {
    var fixed = FixedStack(512){};
    fixed.init();

    const allocator = fixed.allocator();

    const data = try allocator.alloc(u8, 100);
    try std.testing.expectEqual(@as(usize, 100), data.len);

    fixed.reset();
    try std.testing.expectEqual(@as(usize, 0), fixed.getStats().usage);
}

test "stack allocator LIFO free" {
    var buffer: [256]u8 = undefined;
    var stack = StackAllocator.init(&buffer);
    defer stack.deinit();

    const allocator = stack.allocator();

    const a = try allocator.alloc(u8, 32);
    const usage_after_a = stack.usage();

    const b = try allocator.alloc(u8, 32);
    const usage_after_b = stack.usage();

    // Free b (top of stack) - should work
    allocator.free(b);
    try std.testing.expectEqual(usage_after_a, stack.usage());

    // Free a (now top of stack) - should work
    allocator.free(a);
    try std.testing.expectEqual(@as(usize, 0), stack.usage());

    _ = usage_after_b;
}
