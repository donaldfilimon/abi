const std = @import("std");
const engine_mod = @import("engine.zig");
const workload = @import("workload.zig");
const config = @import("config.zig");

const Engine = engine_mod.Engine;
const WorkItem = workload.WorkItem;
const ResultHandle = workload.ResultHandle;
const ResultMetadata = engine_mod.ResultMetadata;

fn dummyDestroy(ptr: *anyopaque, a: std.mem.Allocator) void {
    _ = ptr;
    _ = a;
}

test "engine init and deinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config.EngineConfig{
        .worker_count = 2,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    const engine = try Engine.init(allocator, cfg);
    try std.testing.expect(engine.workers.len == 2);
    engine.deinit();
}

test "engine submit generates unique ids" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config.EngineConfig{
        .worker_count = 2,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    var engine = try Engine.init(allocator, cfg);
    defer engine.deinit();

    const dummy_vtable = workload.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *workload.ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
                _ = user;
                _ = ctx;
                const result_ptr = try a.create(u32);
                result_ptr.* = 42;
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = dummyDestroy,
        .name = "dummy",
    };

    var user_data: u32 = 123;

    const id1 = try engine.submit(WorkItem{
        .id = 0,
        .user = &user_data,
        .vtable = &dummy_vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    });

    const id2 = try engine.submit(WorkItem{
        .id = 0,
        .user = &user_data,
        .vtable = &dummy_vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    });

    try std.testing.expect(id1 != id2);
    try std.testing.expect(id2 > id1);
}

test "engine take transfers ownership" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config.EngineConfig{
        .worker_count = 2,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    var engine = try Engine.init(allocator, cfg);
    defer engine.deinit();

    const dummy_vtable = workload.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *workload.ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
                _ = user;
                _ = ctx;
                const result_ptr = try a.create(u32);
                result_ptr.* = 42;
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = dummyDestroy,
        .name = "dummy",
    };

    var user_data: u32 = 123;

    const id1 = try engine.submit(WorkItem{
        .id = 0,
        .user = &user_data,
        .vtable = &dummy_vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    });

    _ = id1;
}

test "engine complete result" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config.EngineConfig{
        .worker_count = 2,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    var engine = try Engine.init(allocator, cfg);
    defer engine.deinit();

    const result_vtable = workload.ResultVTable{
        .destroy = dummyDestroy,
    };

    var result_data: u32 = 99;

    const handle = ResultHandle{
        .ptr = @ptrCast(&result_data),
        .vtable = &result_vtable,
    };

    const dummy_vtable = workload.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *workload.ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
                _ = user;
                _ = ctx;
                const result_ptr = try a.create(u32);
                result_ptr.* = 42;
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = dummyDestroy,
        .name = "dummy",
    };

    var user_data: u32 = 123;

    const id1 = try engine.submit(WorkItem{
        .id = 0,
        .user = &user_data,
        .vtable = &dummy_vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    });

    try engine.completeResultWithMetadata(id1, handle, 0, 1000, 2000);

    const metadata = engine.getResultMetadata(id1);
    try std.testing.expect(metadata != null);
    try std.testing.expect(metadata.?.worker_id == 0);
    try std.testing.expect(metadata.?.execution_duration_ns == 1000);
}
