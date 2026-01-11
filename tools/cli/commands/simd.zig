//! SIMD performance demo command.

const std = @import("std");
const abi = @import("abi");

/// Run the SIMD performance demonstration.
pub fn run(allocator: std.mem.Allocator) !void {
    const has_simd = abi.hasSimdSupport();
    std.debug.print("SIMD Support: {s}\n", .{if (has_simd) "available" else "unavailable"});

    if (!has_simd) {
        std.debug.print("SIMD not available on this platform.\n", .{});
        return;
    }

    const size = 1000;
    const total_size = size * 3;
    var data = try allocator.alloc(f32, total_size);
    defer allocator.free(data);

    var a = data[0..size];
    var b = data[size .. size * 2];
    var result = data[size * 2 .. total_size];

    var i: usize = 0;
    while (i < size) : (i += 1) {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i * 2);
    }

    var timer = try std.time.Timer.start();
    _ = timer.lap();

    abi.simd.vectorAdd(a, b, result);

    const add_time = timer.lap();

    i = 0;
    while (i < size) : (i += 1) {
        const expected = a[i] + b[i];
        if (@abs(result[i] - expected) > 1e-6) {
            std.debug.print("SIMD verification failed at index {d}\n", .{i});
            return;
        }
    }

    _ = timer.lap();
    const dot_result = abi.simd.vectorDot(a, b);
    const dot_time = timer.lap();

    _ = timer.lap();
    const norm_result = abi.simd.vectorL2Norm(a);
    const norm_time = timer.lap();

    _ = timer.lap();
    const cos_result = abi.simd.cosineSimilarity(a, b);
    const cos_time = timer.lap();

    std.debug.print("SIMD Operations Performance ({} elements):\n", .{size});
    std.debug.print("  Vector Addition: {d} ns ({d:.2} ops/sec)\n", .{
        add_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(add_time)) / std.time.ns_per_s),
    });
    std.debug.print("  Dot Product: {d} ns ({d:.2} ops/sec)\n", .{
        dot_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(dot_time)) / std.time.ns_per_s),
    });
    std.debug.print("  L2 Norm: {d} ns ({d:.2} ops/sec)\n", .{
        norm_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(norm_time)) / std.time.ns_per_s),
    });
    std.debug.print("  Cosine Similarity: {d} ns ({d:.2} ops/sec)\n", .{
        cos_time,
        @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(cos_time)) / std.time.ns_per_s),
    });

    std.debug.print("Results:\n", .{});
    std.debug.print("  Dot Product: {d:.6}\n", .{dot_result});
    std.debug.print("  L2 Norm: {d:.6}\n", .{norm_result});
    std.debug.print("  Cosine Similarity: {d:.6}\n", .{cos_result});
}
