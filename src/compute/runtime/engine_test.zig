const std = @import("std");
const engine_mod = @import("engine.zig");
const workload = @import("workload.zig");
const config = @import("config.zig");
const builtin = @import("builtin");

fn sleep(ns: u64) void {
    if (builtin.os.tag == .windows) {
        _ = std.os.windows.kernel32.SleepEx(@intCast(ns / std.time.ns_per_ms), 0);
    } else {
        const s = ns / std.time.ns_per_s;
        const n = ns % std.time.ns_per_s;
        _ = std.posix.nanosleep(&.{ .sec = @intCast(s), .nsec = @intCast(n) }, null);
    }
}

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

    // Spin until result is ready
    var result_handle: ?ResultHandle = null;
    while (result_handle == null) {
        result_handle = engine.take(id1);
        if (result_handle == null) {
            sleep(1 * std.time.ns_per_ms);
        }
    }

    const result = result_handle.?;
    try std.testing.expect(result.as(u32).* == 42);
    
    // Handle owns memory, so we deinit it
    var res_handle_copy = result;
    res_handle_copy.deinit(allocator);
}

test "engine poll returns any completed result" {
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
    _ = try engine.submit(WorkItem{
        .id = 0,
        .user = &user_data,
        .vtable = &dummy_vtable,
        .priority = 1.0,
        .hints = workload.DEFAULT_HINTS,
    });

    var result_handle: ?ResultHandle = null;
    while (result_handle == null) {
        result_handle = engine.poll();
        if (result_handle == null) {
            sleep(1 * std.time.ns_per_ms);
        }
    }

    const result = result_handle.?;
    try std.testing.expect(result.as(u32).* == 42);

    var res_handle_copy = result;
    res_handle_copy.deinit(allocator);
}

test "engine submit multiple tasks" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config.EngineConfig{
        .worker_count = 4,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    var engine = try Engine.init(allocator, cfg);
    defer engine.deinit();

    const TaskData = struct {
        value: u32,
    };

    const task_vtable = workload.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *workload.ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
                const data: *TaskData = @ptrCast(@alignCast(user));
                _ = ctx;
                const result_ptr = try a.create(u32);
                result_ptr.* = data.value * 2;
                // Simulate some work
                // sleep(1); // Removed to avoid dependency, just computation
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = dummyDestroy,
        .name = "multiply",
    };

    const task_count = 50;
    var task_ids: [task_count]u64 = undefined;
    var task_data: [task_count]TaskData = undefined;

    for (0..task_count) |i| {
        task_data[i] = TaskData{ .value = @intCast(i) };
        task_ids[i] = try engine.submit(WorkItem{
            .id = 0,
            .user = &task_data[i],
            .vtable = &task_vtable,
            .priority = 1.0,
            .hints = workload.DEFAULT_HINTS,
        });
    }

    // Collect all results
    var completed: usize = 0;
    for (0..task_count) |i| {
        const id = task_ids[i];
        var handle: ?ResultHandle = null;
        while (handle == null) {
            handle = engine.take(id);
            if (handle == null) {
                sleep(100 * std.time.ns_per_us); // 100us
            }
        }
        
        const result = handle.?;
        const value = result.as(u32).*;
        try std.testing.expectEqual(@as(u32, @intCast(i * 2)), value);
        
        var handle_copy = result;
        handle_copy.deinit(allocator);
        completed += 1;
    }

    try std.testing.expectEqual(task_count, completed);
}

test "engine worker stress" {
    // This test submits enough work to likely force work stealing
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Use fewer workers to force queue buildup
    const cfg = config.EngineConfig{
        .worker_count = 2, 
        .drain_mode = .drain,
        .queue_capacity = 256,
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
                result_ptr.* = 1;
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = dummyDestroy,
        .name = "dummy",
    };

    var user_data: u32 = 0;
    const count = 100;
    var ids: [count]u64 = undefined;

    for (0..count) |i| {
        ids[i] = try engine.submit(WorkItem{
            .id = 0,
            .user = &user_data,
            .vtable = &dummy_vtable,
            .priority = 1.0,
            .hints = workload.DEFAULT_HINTS,
        });
    }

    for (0..count) |i| {
        const id = ids[i];
        while (true) {
            if (engine.take(id)) |handle| {
                var h = handle;
                h.deinit(allocator);
                break;
            }
            sleep(100 * std.time.ns_per_us);
        }
    }
}