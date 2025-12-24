//! Compute demo
//!
//! Minimal program demonstrating the compute framework
//! with submit/poll/take workflow and one workload.

const std = @import("std");
const runtime = @import("runtime/mod.zig");

fn demoDestroy(ptr: *anyopaque, a: std.mem.Allocator) void {
    _ = ptr;
    _ = a;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const info = runtime.topology.detect();
    std.debug.print("Compute Framework Demo\n", .{});
    std.debug.print("Cores: {d}, Cache Line: {d} bytes\n", .{ info.core_count, info.cache_line_size });

    const cfg = runtime.config.DEFAULT_CONFIG;

    var engine = try runtime.Engine.init(allocator, cfg);
    defer engine.deinit();

    const vtable = runtime.WorkloadVTable{
        .exec = struct {
            fn exec(user: *anyopaque, ctx: *runtime.ExecutionContext, a: std.mem.Allocator) anyerror!*anyopaque {
                _ = user;
                _ = ctx;
                const result_ptr = try a.create(u32);
                result_ptr.* = 42;
                return @ptrCast(result_ptr);
            }
        }.exec,
        .destroy = demoDestroy,
        .name = "demo_workload",
    };

    var counter = std.atomic.Value(u32).init(0);

    for (0..4) |i| {
        var user_data: u32 = @intCast(i);
        _ = try engine.submit(runtime.WorkItem{
            .id = 0,
            .user = &user_data,
            .vtable = &vtable,
            .priority = 1.0,
            .hints = runtime.WorkloadHints{ .cpu_affinity = null, .estimated_duration_us = null },
        });
        _ = counter.fetchAdd(1, .monotonic);
    }

    std.debug.print("Submitted tasks: {d}\n", .{counter.load(.acquire)});
}
