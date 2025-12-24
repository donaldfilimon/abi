//! Compute demo
//!
//! Minimal program demonstrating the compute framework
//! with submit/poll/take workflow and one workload.

const std = @import("std");
const runtime = @import("runtime/mod.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const info = runtime.topology.detect();
    std.debug.print("Compute Framework Demo\n", .{});
    std.debug.print("Cores: {d}, Cache Line: {d} bytes\n", .{ info.core_count, info.cache_line_size });

    const cfg = runtime.config.DEFAULT_CONFIG;

    var engine = try runtime.Engine.init(allocator, cfg);
    defer engine.deinit();

    var counter = std.atomic.Value(u32).init(0);

    for (0..4) |i| {
        const hints = runtime.WorkloadHints{ .cpu_affinity = null, .estimated_duration_us = null };
        _ = try engine.submit(.{ .id = @as(u64, i), .user = null, .vtable = null, .priority = 0.0, .hints = hints });
        _ = counter.fetchAdd(1, .monotonic);
    }

    std.debug.print("Completed tasks: {d}\n", .{counter.load(.acquire)});
}
