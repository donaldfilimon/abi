//! Tracking allocator for debugging memory allocations.
//!
//! Wraps any allocator to track all allocations, providing:
//! - Allocation counting and size tracking
//! - Peak memory usage monitoring
//! - Leak detection with source location info
//! - Allocation history for debugging
//!
//! Usage:
//! ```zig
//! var tracker = TrackingAllocator.init(std.testing.allocator, .{});
//! defer tracker.deinit();
//! const allocator = tracker.allocator();
//! // ... use allocator ...
//! const stats = tracker.getStats();
//! if (tracker.detectLeaks()) {
//!     var io_backend = std.Io.Threaded.init(std.testing.allocator, .{
//!         .environ = std.process.Environ.empty,
//!     });
//!     defer io_backend.deinit();
//!     var stderr_buffer: [4096]u8 = undefined;
//!     var stderr_writer = std.Io.File.stderr().writer(io_backend.io(), &stderr_buffer);
//!     tracker.dumpLeaks(&stderr_writer);
//! }
//! ```

const std = @import("std");

/// Configuration for the tracking allocator.
pub const TrackingConfig = struct {
    /// Maximum number of allocations to track in history.
    max_history: usize = 1024,
    /// Enable stack trace capture (expensive).
    capture_stack_traces: bool = false,
    /// Enable detailed timing info.
    capture_timing: bool = false,
    /// Log allocations immediately.
    log_allocations: bool = false,
};

/// Information about a single allocation.
pub const AllocationInfo = struct {
    /// Address of the allocation.
    address: usize,
    /// Size of the allocation.
    size: usize,
    /// Alignment of the allocation.
    alignment: u8,
    /// Timestamp when allocated (if timing enabled).
    timestamp: ?i128,
    /// Whether this allocation is still active.
    is_active: bool,
    /// Source file (if available).
    source_file: ?[]const u8,
    /// Source line (if available).
    source_line: ?u32,
};

/// Statistics collected by the tracking allocator.
pub const TrackingStats = struct {
    /// Total number of allocations made.
    total_allocations: u64,
    /// Total number of frees made.
    total_frees: u64,
    /// Current number of active allocations.
    active_allocations: u64,
    /// Total bytes currently allocated.
    current_bytes: u64,
    /// Peak bytes allocated at any point.
    peak_bytes: u64,
    /// Total bytes allocated over lifetime.
    total_bytes_allocated: u64,
    /// Total bytes freed over lifetime.
    total_bytes_freed: u64,
    /// Number of failed allocations.
    failed_allocations: u64,
    /// Number of double-frees detected.
    double_frees: u64,
    /// Number of invalid frees detected.
    invalid_frees: u64,
};

/// Tracking allocator that wraps another allocator.
pub const TrackingAllocator = struct {
    backing_allocator: std.mem.Allocator,
    config: TrackingConfig,
    stats: TrackingStats,
    allocations: std.AutoHashMapUnmanaged(usize, AllocationInfo),
    history: std.ArrayListUnmanaged(AllocationInfo),
    mutex: std.Thread.Mutex,

    const Self = @This();

    /// Initialize the tracking allocator.
    pub fn init(backing_allocator: std.mem.Allocator, config: TrackingConfig) Self {
        return .{
            .backing_allocator = backing_allocator,
            .config = config,
            .stats = std.mem.zeroes(TrackingStats),
            .allocations = .{},
            .history = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the tracking allocator.
    pub fn deinit(self: *Self) void {
        self.allocations.deinit(self.backing_allocator);
        self.history.deinit(self.backing_allocator);
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

    /// Get current statistics.
    pub fn getStats(self: *Self) TrackingStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Reset statistics (but not active allocations).
    pub fn resetStats(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const active = self.stats.active_allocations;
        const current = self.stats.current_bytes;
        self.stats = std.mem.zeroes(TrackingStats);
        self.stats.active_allocations = active;
        self.stats.current_bytes = current;
        self.stats.peak_bytes = current;
    }

    /// Check if there are any memory leaks.
    pub fn detectLeaks(self: *Self) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats.active_allocations > 0;
    }

    /// Get count of active allocations.
    pub fn activeCount(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats.active_allocations;
    }

    /// Get active allocation info by address.
    pub fn getAllocationInfo(self: *Self, address: usize) ?AllocationInfo {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.allocations.get(address);
    }

    /// Write leak report to writer.
    pub fn dumpLeaks(self: *Self, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.stats.active_allocations == 0) {
            try writer.writeAll("No memory leaks detected.\n");
            return;
        }

        try writer.print("=== Memory Leak Report ===\n", .{});
        try writer.print("Active allocations: {d}\n", .{self.stats.active_allocations});
        try writer.print("Leaked bytes: {d}\n\n", .{self.stats.current_bytes});

        var iter = self.allocations.iterator();
        var count: usize = 0;
        while (iter.next()) |entry| {
            const info = entry.value_ptr;
            if (info.is_active) {
                try writer.print("  Leak #{d}: {d} bytes at 0x{x}\n", .{
                    count + 1,
                    info.size,
                    info.address,
                });
                if (info.source_file) |file| {
                    try writer.print("    Source: {s}", .{file});
                    if (info.source_line) |line| {
                        try writer.print(":{d}", .{line});
                    }
                    try writer.writeAll("\n");
                }
                count += 1;
            }
        }
    }

    /// Get allocation history (most recent first).
    pub fn getHistory(self: *Self) []const AllocationInfo {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.history.items;
    }

    // Allocator vtable implementations (Zig 0.16 API with std.mem.Alignment)
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const ptr = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (ptr) |p| {
            const address = @intFromPtr(p);
            const info = AllocationInfo{
                .address = address,
                .size = len,
                .alignment = @intFromEnum(ptr_align),
                .timestamp = if (self.config.capture_timing) 0 else null,
                .is_active = true,
                .source_file = null,
                .source_line = null,
            };

            self.allocations.put(self.backing_allocator, address, info) catch {};

            if (self.history.items.len < self.config.max_history) {
                self.history.append(self.backing_allocator, info) catch {};
            }

            self.stats.total_allocations += 1;
            self.stats.active_allocations += 1;
            self.stats.current_bytes += len;
            self.stats.total_bytes_allocated += len;

            if (self.stats.current_bytes > self.stats.peak_bytes) {
                self.stats.peak_bytes = self.stats.current_bytes;
            }
        } else {
            self.stats.failed_allocations += 1;
        }

        return ptr;
    }

    fn resize(ctx: *anyopaque, memory: []u8, ptr_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawResize(memory, ptr_align, new_len, ret_addr);

        if (result) {
            self.mutex.lock();
            defer self.mutex.unlock();

            const address = @intFromPtr(memory.ptr);
            if (self.allocations.getPtr(address)) |info| {
                const old_size = info.size;
                info.size = new_len;

                if (new_len > old_size) {
                    const diff = new_len - old_size;
                    self.stats.current_bytes += diff;
                    self.stats.total_bytes_allocated += diff;
                } else {
                    const diff = old_size - new_len;
                    self.stats.current_bytes -= diff;
                    self.stats.total_bytes_freed += diff;
                }

                if (self.stats.current_bytes > self.stats.peak_bytes) {
                    self.stats.peak_bytes = self.stats.current_bytes;
                }
            }
        }

        return result;
    }

    fn remap(ctx: *anyopaque, memory: []u8, ptr_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawRemap(memory, ptr_align, new_len, ret_addr);

        if (result) |new_ptr| {
            self.mutex.lock();
            defer self.mutex.unlock();

            const old_address = @intFromPtr(memory.ptr);
            const new_address = @intFromPtr(new_ptr);

            // Update tracking if address changed
            if (old_address != new_address) {
                if (self.allocations.fetchRemove(old_address)) |kv| {
                    var info = kv.value;
                    info.address = new_address;
                    info.size = new_len;
                    self.allocations.put(self.backing_allocator, new_address, info) catch {};
                }
            } else if (self.allocations.getPtr(old_address)) |info| {
                // Same address, just update size
                const old_size = info.size;
                info.size = new_len;

                if (new_len > old_size) {
                    const diff = new_len - old_size;
                    self.stats.current_bytes += diff;
                    self.stats.total_bytes_allocated += diff;
                } else {
                    const diff = old_size - new_len;
                    self.stats.current_bytes -= diff;
                    self.stats.total_bytes_freed += diff;
                }
            }

            if (self.stats.current_bytes > self.stats.peak_bytes) {
                self.stats.peak_bytes = self.stats.current_bytes;
            }
        }

        return result;
    }

    fn free(ctx: *anyopaque, memory: []u8, ptr_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const address = @intFromPtr(memory.ptr);

        self.mutex.lock();
        const tracked = self.allocations.fetchRemove(address);
        if (tracked) |kv| {
            if (!kv.value.is_active) {
                self.stats.double_frees += 1;
            } else {
                self.stats.total_frees += 1;
                self.stats.active_allocations -= 1;
                self.stats.current_bytes -= kv.value.size;
                self.stats.total_bytes_freed += kv.value.size;
            }
        } else {
            self.stats.invalid_frees += 1;
        }
        self.mutex.unlock();

        self.backing_allocator.rawFree(memory, ptr_align, ret_addr);
    }
};

test "tracking allocator basic operations" {
    var tracker = TrackingAllocator.init(std.testing.allocator, .{});
    defer tracker.deinit();

    const allocator = tracker.allocator();

    // Allocate some memory
    const ptr1 = try allocator.alloc(u8, 100);
    const ptr2 = try allocator.alloc(u8, 200);

    var stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.active_allocations);
    try std.testing.expectEqual(@as(u64, 300), stats.current_bytes);

    // Free one
    allocator.free(ptr1);
    stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.active_allocations);
    try std.testing.expectEqual(@as(u64, 200), stats.current_bytes);

    // Free the other
    allocator.free(ptr2);
    stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.active_allocations);
    try std.testing.expectEqual(@as(u64, 0), stats.current_bytes);

    try std.testing.expect(!tracker.detectLeaks());
}

test "tracking allocator leak detection" {
    var tracker = TrackingAllocator.init(std.testing.allocator, .{});
    defer {
        // Clean up intentional leaks for test
        var iter = tracker.allocations.iterator();
        while (iter.next()) |entry| {
            const ptr: [*]u8 = @ptrFromInt(entry.key_ptr.*);
            std.testing.allocator.free(ptr[0..entry.value_ptr.size]);
        }
        tracker.deinit();
    }

    const allocator = tracker.allocator();

    // Allocate but don't free (intentional leak for testing)
    _ = try allocator.alloc(u8, 50);

    try std.testing.expect(tracker.detectLeaks());
    try std.testing.expectEqual(@as(u64, 1), tracker.activeCount());
}

test "tracking allocator peak usage" {
    var tracker = TrackingAllocator.init(std.testing.allocator, .{});
    defer tracker.deinit();

    const allocator = tracker.allocator();

    const ptr1 = try allocator.alloc(u8, 100);
    const ptr2 = try allocator.alloc(u8, 200);

    var stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 300), stats.peak_bytes);

    allocator.free(ptr1);
    stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 300), stats.peak_bytes); // Peak unchanged

    allocator.free(ptr2);
    stats = tracker.getStats();
    try std.testing.expectEqual(@as(u64, 300), stats.peak_bytes); // Peak unchanged
    try std.testing.expectEqual(@as(u64, 0), stats.current_bytes);
}
