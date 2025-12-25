const std = @import("std");

pub const WorkloadHints = struct {
    cpu_affinity: ?u32 = null,
    estimated_duration_us: ?u64 = null,
    prefers_gpu: bool = false,
    requires_gpu: bool = false,
};

pub const ExecutionContext = struct {
    allocator: std.mem.Allocator,
    worker_index: u32 = 0,
    start_ns: u64 = 0,
};

pub const ResultHandle = struct {
    bytes: []const u8,
    owned: bool = false,
    allocator: ?std.mem.Allocator = null,

    pub fn fromSlice(bytes: []const u8) ResultHandle {
        return .{ .bytes = bytes, .owned = false, .allocator = null };
    }

    pub fn fromOwned(allocator: std.mem.Allocator, bytes: []u8) ResultHandle {
        return .{ .bytes = bytes, .owned = true, .allocator = allocator };
    }

    pub fn deinit(self: *ResultHandle) void {
        if (self.owned) {
            if (self.allocator) |allocator| {
                allocator.free(self.bytes);
            }
        }
        self.* = undefined;
    }
};

pub const WorkloadVTable = struct {
    execute: *const fn (ctx: *ExecutionContext, user: *anyopaque) anyerror!ResultHandle,
};

pub const ResultVTable = struct {
    release: *const fn (user: *anyopaque, allocator: std.mem.Allocator) void,
};

pub const GPUWorkloadVTable = struct {
    execute: *const fn (ctx: *ExecutionContext, user: *anyopaque) anyerror!ResultHandle,
};

pub const WorkItem = struct {
    id: u64,
    user: *anyopaque,
    vtable: *const WorkloadVTable,
    priority: i32 = 0,
    hints: WorkloadHints = .{},
    gpu_vtable: ?*const GPUWorkloadVTable = null,
};

pub fn runWorkItem(ctx: *ExecutionContext, item: *const WorkItem) !ResultHandle {
    return item.vtable.execute(ctx, item.user);
}

test "work item executes vtable" {
    const allocator = std.testing.allocator;
    var value: u32 = 0;
    const ctx = ExecutionContext{ .allocator = allocator };
    const vtable = WorkloadVTable{ .execute = runSample };
    const item = WorkItem{
        .id = 1,
        .user = &value,
        .vtable = &vtable,
        .priority = 0,
        .hints = .{},
    };

    const result = try runWorkItem(&ctx, &item);
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 7), value);
    try std.testing.expectEqual(@as(usize, 0), result.bytes.len);
}

fn runSample(_: *ExecutionContext, user: *anyopaque) !ResultHandle {
    const ptr: *u32 = @ptrCast(@alignCast(user));
    ptr.* = 7;
    return ResultHandle.fromSlice(&.{});
}
