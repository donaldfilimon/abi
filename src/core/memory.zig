//! Memory Management and Tracking utilities
const std = @import("std");
const time = @import("../foundation/time.zig");
const errors = @import("../foundation/errors.zig");

pub const AllocationRecord = struct {
    ptr: usize,
    size: usize,
    timestamp: i64,
    tag: []const u8,
};

pub const MemoryTracker = struct {
    allocator: std.mem.Allocator,
    records: std.ArrayListUnmanaged(AllocationRecord),
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    current_usage: usize,

    pub fn init(allocator: std.mem.Allocator) MemoryTracker {
        return .{
            .allocator = allocator,
            .records = std.ArrayListUnmanaged(AllocationRecord).empty,
            .total_allocated = 0,
            .total_freed = 0,
            .peak_usage = 0,
            .current_usage = 0,
        };
    }

    pub fn deinit(self: *MemoryTracker) void {
        for (self.records.items) |*record| {
            self.allocator.free(record.tag);
        }
        self.records.deinit(self.allocator);
    }

    pub fn trackAlloc(self: *MemoryTracker, ptr: [*]const u8, size: usize, tag: []const u8) !void {
        const tag_copy = try self.allocator.dupe(u8, tag);
        errdefer self.allocator.free(tag_copy);

        const record = AllocationRecord{
            .ptr = @intFromPtr(ptr),
            .size = size,
            .timestamp = time.unixMs(),
            .tag = tag_copy,
        };

        try self.records.append(self.allocator, record);

        self.total_allocated += size;
        self.current_usage += size;
        if (self.current_usage > self.peak_usage) {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn trackFree(self: *MemoryTracker, ptr: [*]const u8, size: usize) void {
        _ = ptr;
        self.total_freed += size;
        if (self.current_usage >= size) {
            self.current_usage -= size;
        } else {
            self.current_usage = 0;
        }
    }

    pub fn getRecordCount(self: *const MemoryTracker) usize {
        return self.records.items.len;
    }

    pub fn getRecords(self: *const MemoryTracker) []const AllocationRecord {
        return self.records.items;
    }

    pub fn getPeakUsage(self: *const MemoryTracker) usize {
        return self.peak_usage;
    }

    pub fn getCurrentUsage(self: *const MemoryTracker) usize {
        return self.current_usage;
    }

    pub fn getLeakedBytes(self: *const MemoryTracker) usize {
        return self.total_allocated - self.total_freed;
    }

    /// Non-fallible tracking for hot-path use (no tag string allocation).
    pub fn trackAllocNoTag(self: *MemoryTracker, size: usize) void {
        self.total_allocated += size;
        self.current_usage += size;
        if (self.current_usage > self.peak_usage) {
            self.peak_usage = self.current_usage;
        }
    }

    /// Non-fallible free tracking for hot-path use.
    pub fn trackFreeNoTag(self: *MemoryTracker, size: usize) void {
        self.total_freed += size;
        if (self.current_usage >= size) {
            self.current_usage -= size;
        } else {
            self.current_usage = 0;
        }
    }
};

pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    offset: usize,
    block_size: usize,
    allocation_count: usize,

    pub fn init(allocator: std.mem.Allocator, block_size: usize, block_count: usize) !MemoryPool {
        const total_size = block_size * block_count;
        const buffer = try allocator.alloc(u8, total_size);
        errdefer allocator.free(buffer);

        @memset(buffer, 0);

        return .{
            .allocator = allocator,
            .buffer = buffer,
            .offset = 0,
            .block_size = block_size,
            .allocation_count = 0,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        self.allocator.free(self.buffer);
        self.buffer = &[_]u8{};
        self.offset = 0;
    }

    pub fn alloc(self: *MemoryPool, size: usize) ![]u8 {
        if (size > self.block_size) {
            return errors.AbiError.InvalidConfig;
        }

        if (self.offset + size > self.buffer.len) {
            return errors.AbiError.OutOfMemory;
        }

        const slice = self.buffer[self.offset .. self.offset + size];
        self.offset += size;
        self.allocation_count += 1;

        return slice;
    }

    pub fn reset(self: *MemoryPool) void {
        self.offset = 0;
        self.allocation_count = 0;
        @memset(self.buffer, 0);
    }

    pub fn getRemaining(self: *const MemoryPool) usize {
        return self.buffer.len - self.offset;
    }

    pub fn getUsed(self: *const MemoryPool) usize {
        return self.offset;
    }

    pub fn getCapacity(self: *const MemoryPool) usize {
        return self.buffer.len;
    }
};

pub const TrackingAllocator = struct {
    parent: std.mem.Allocator,
    tracker: *MemoryTracker,

    pub fn init(parent: std.mem.Allocator, tracker: *MemoryTracker) TrackingAllocator {
        return .{
            .parent = parent,
            .tracker = tracker,
        };
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
        const result = self.parent.rawAlloc(len, alignment, ret_addr) orelse return null;

        const tag = std.fmt.allocPrint(self.parent, "alloc@{x}", .{ret_addr}) catch blk: {
            break :blk "unknown";
        };
        defer if (tag.len > 7) self.parent.free(tag);

        self.tracker.trackAlloc(result, len, tag) catch |err| {
            std.log.warn("failed to track allocation: {s}", .{@errorName(err)});
        };

        return result;
    }

    fn resizeFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        return self.parent.rawResize(buf, alignment, new_len, ret_addr);
    }

    fn remapFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        return self.parent.rawRemap(buf, alignment, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.tracker.trackFree(buf.ptr, buf.len);
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "MemoryTracker init and deinit" {
    var tracker = MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();

    try std.testing.expectEqual(@as(usize, 0), tracker.getRecordCount());
    try std.testing.expectEqual(@as(usize, 0), tracker.getCurrentUsage());
}

test "MemoryTracker track allocations" {
    var tracker = MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();

    const buf = try std.testing.allocator.alloc(u8, 100);
    defer std.testing.allocator.free(buf);

    try tracker.trackAlloc(buf.ptr, 100, "test_buf");
    try std.testing.expectEqual(@as(usize, 1), tracker.getRecordCount());
    try std.testing.expectEqual(@as(usize, 100), tracker.getCurrentUsage());
    try std.testing.expectEqual(@as(usize, 100), tracker.getPeakUsage());
}

test "MemoryPool alloc and reset" {
    var pool = try MemoryPool.init(std.testing.allocator, 64, 10);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 640), pool.getCapacity());

    const block = try pool.alloc(32);
    try std.testing.expectEqual(@as(usize, 32), block.len);
    try std.testing.expectEqual(@as(usize, 608), pool.getRemaining());

    pool.reset();
    try std.testing.expectEqual(@as(usize, 640), pool.getRemaining());
    try std.testing.expectEqual(@as(usize, 0), pool.getUsed());
}

test "MemoryPool overflow" {
    var pool = try MemoryPool.init(std.testing.allocator, 16, 2);
    defer pool.deinit();

    _ = try pool.alloc(16);
    _ = try pool.alloc(16);

    const err = pool.alloc(1) catch |e| e;
    try std.testing.expectEqual(errors.AbiError.OutOfMemory, err);
}

test "MemoryPool block size limit" {
    var pool = try MemoryPool.init(std.testing.allocator, 8, 10);
    defer pool.deinit();

    const err = pool.alloc(16) catch |e| e;
    try std.testing.expectEqual(errors.AbiError.InvalidConfig, err);
}

test "MemoryTracker trackAllocNoTag and trackFreeNoTag" {
    var tracker = MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();

    tracker.trackAllocNoTag(64);
    try std.testing.expectEqual(@as(usize, 64), tracker.getCurrentUsage());
    try std.testing.expectEqual(@as(usize, 64), tracker.getPeakUsage());

    tracker.trackAllocNoTag(32);
    try std.testing.expectEqual(@as(usize, 96), tracker.getCurrentUsage());
    try std.testing.expectEqual(@as(usize, 96), tracker.getPeakUsage());

    tracker.trackFreeNoTag(48);
    try std.testing.expectEqual(@as(usize, 48), tracker.getCurrentUsage());
    // peak unchanged
    try std.testing.expectEqual(@as(usize, 96), tracker.getPeakUsage());

    // over-free clamps to zero
    tracker.trackFreeNoTag(999);
    try std.testing.expectEqual(@as(usize, 0), tracker.getCurrentUsage());
}

test "TrackingAllocator" {
    var tracker = MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();

    var tracking = TrackingAllocator.init(std.testing.allocator, &tracker);
    const alloc = tracking.allocator();

    const buf = try alloc.alloc(u8, 50);
    defer alloc.free(buf);

    try std.testing.expect(tracker.getRecordCount() > 0);
    try std.testing.expect(tracker.getCurrentUsage() > 0);
}
