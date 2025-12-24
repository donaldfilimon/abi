const std = @import("std");
const abbey = @import("main.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const info = abbey.core.detectCpuInfo();
    std.debug.print("Abbey Framework Demo\n", .{});
    std.debug.print("Cores: {d}, SIMD width: {d}, Cache line: {d} bytes\n", .{
        info.core_count,
        info.simd_width,
        info.cache_line_size,
    });

    var framework = try abbey.Framework.init(allocator, .{ .worker_count = 2, .enable_profiler = true });
    defer framework.deinit();

    var a = try abbey.workload.Matrix.init(allocator, 2, 2);
    defer a.deinit();
    var b = try abbey.workload.Matrix.init(allocator, 2, 2);
    defer b.deinit();
    var out = try abbey.workload.Matrix.init(allocator, 2, 2);
    defer out.deinit();

    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(1, 0, 3.0);
    a.set(1, 1, 4.0);

    b.set(0, 0, 5.0);
    b.set(0, 1, 6.0);
    b.set(1, 0, 7.0);
    b.set(1, 1, 8.0);

    abbey.workload.matmul(&out, a, b);
    std.debug.print("Matmul result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{
        out.get(0, 0),
        out.get(0, 1),
        out.get(1, 0),
        out.get(1, 1),
    });

    var counter = std.atomic.Value(u32).init(0);
    const Context = struct {
        fn task(ctx: ?*anyopaque) void {
            const ptr: *std.atomic.Value(u32) = @ptrCast(@alignCast(ctx.?));
            _ = ptr.fetchAdd(1, .monotonic);
        }
    };

    for (0..4) |_| {
        try framework.engine.submit(.{ .func = Context.task, .ctx = &counter });
    }
    framework.engine.waitIdle();
    std.debug.print("Completed tasks: {d}\n", .{counter.load(.acquire)});
}
