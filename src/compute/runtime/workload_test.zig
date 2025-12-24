const std = @import("std");
const workload = @import("workload.zig");
const runtime = @import("mod.zig");

const WorkloadVTable = workload.WorkloadVTable;
const WorkItem = workload.WorkItem;
const ExecutionContext = workload.ExecutionContext;
const ResultHandle = workload.ResultHandle;
const ResultVTable = workload.ResultVTable;

const TestCounter = struct {
    exec_count: std.atomic.Value(u32),
    destroy_count: std.atomic.Value(u32),

    fn exec(user: *anyopaque, ctx: *ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
        _ = ctx;
        const self = @as(*const TestCounter, @ptrCast(@alignCast(user)));
        self.exec_count.fetchAdd(1, .monotonic);

        const result_ptr = try a.create(u32);
        result_ptr.* = 42;
        return @ptrCast(result_ptr);
    }

    fn destroy(user: *anyopaque, a: std.mem.Allocator) void {
        _ = a;
        const self = @as(*const TestCounter, @ptrCast(@alignCast(user)));
        self.destroy_count.fetchAdd(1, .monotonic);
    }
};

test "workload vtable exec and destroy called exactly once" {
    var counter = TestCounter{
        .exec_count = std.atomic.Value(u32).init(0),
        .destroy_count = std.atomic.Value(u32).init(0),
    };

    const vtable = WorkloadVTable{
        .exec = TestCounter.exec,
        .destroy = TestCounter.destroy,
        .name = "test_workload",
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const item = WorkItem{
        .id = 1,
        .user = &counter,
        .vtable = &vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var ctx = ExecutionContext{
        .worker_id = 0,
        .arena = &arena,
    };

    const result_ptr = try item.vtable.exec(item.user, &ctx, allocator);

    try std.testing.expect(@as(*u32, @ptrCast(@alignCast(result_ptr))).* == 42);
    try std.testing.expect(counter.exec_count.load(.monotonic) == 1);

    item.vtable.destroy(item.user, allocator);
    try std.testing.expect(counter.destroy_count.load(.monotonic) == 1);

    allocator.destroy(@as(*u32, @ptrCast(@alignCast(result_ptr))));
}

fn dummyDestroy(ptr: *anyopaque, a: std.mem.Allocator) void {
    _ = ptr;
    _ = a;
}

test "result handle as and deinit" {
    const vtable = ResultVTable{
        .destroy = dummyDestroy,
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data_ptr = try allocator.create(u32);
    data_ptr.* = 99;

    const handle = ResultHandle{
        .ptr = @ptrCast(data_ptr),
        .vtable = &vtable,
    };

    const typed: *u32 = handle.as(u32);
    try std.testing.expect(typed.* == 99);

    handle.deinit(allocator);
    allocator.destroy(data_ptr);
}

test "workload hints" {
    const hints = workload.DEFAULT_HINTS;
    try std.testing.expect(hints.cpu_affinity == null);
    try std.testing.expect(hints.estimated_duration_us == null);

    const custom_hints = workload.WorkloadHints{
        .cpu_affinity = 0,
        .estimated_duration_us = 1000,
    };
    try std.testing.expect(custom_hints.cpu_affinity.? == 0);
    try std.testing.expect(custom_hints.estimated_duration_us.? == 1000);
}

test "execution context" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const ctx = ExecutionContext{
        .worker_id = 5,
        .arena = &arena,
    };

    try std.testing.expect(ctx.worker_id == 5);
    try std.testing.expect(ctx.arena == &arena);
}
