const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = abi.initWithConfig(allocator, abi.Config.minimal()) catch |err| {
        std.debug.print("Failed to initialize framework: {}\n", .{err});
        return err;
    };
    defer framework.deinit();

    std.debug.print("Compute runtime initialized successfully\n", .{});

    // Check SIMD support
    const has_simd = abi.hasSimdSupport();
    std.debug.print("SIMD support: {s}\n", .{if (has_simd) "available" else "not available"});

    if (!has_simd) {
        std.debug.print("Warning: SIMD operations may not be optimized\n", .{});
    }

    // Test SIMD operations
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const dot_result = abi.vectorDot(&vec_a, &vec_b);
    std.debug.print("SIMD dot product: {d:.3}\n", .{dot_result});

    var vec_sum = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    abi.vectorAdd(&vec_a, &vec_b, &vec_sum);
    std.debug.print("SIMD vector addition result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ vec_sum[0], vec_sum[1], vec_sum[2], vec_sum[3] });
}
