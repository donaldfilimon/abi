//! SIMD Compute Example
//!
//! Demonstrates SIMD vector operations including dot product,
//! cosine similarity, and L2 distance calculations.
//!
//! Run with: `zig build run-compute`

const std = @import("std");
const abi = @import("abi");
const v2 = abi.shared.utils.v2_primitives;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = abi.Framework.initMinimal(allocator) catch |err| {
        std.debug.print("Failed to initialize framework: {t}\n", .{err});
        return err;
    };
    defer framework.deinit();

    std.debug.print("Compute runtime initialized successfully\n", .{});
    std.debug.print("Platform: {s}\n", .{v2.Platform.description()});

    // Check SIMD support
    const has_simd = abi.simd.hasSimdSupport();
    std.debug.print("SIMD support: {s}\n", .{if (has_simd) "available" else "not available"});

    if (!has_simd) {
        std.debug.print("Warning: SIMD operations may not be optimized\n", .{});
    }

    // Test SIMD operations
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const dot_result = abi.simd.vectorDot(&vec_a, &vec_b);
    std.debug.print("SIMD dot product: {d:.3}\n", .{dot_result});

    var vec_sum = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    abi.simd.vectorAdd(&vec_a, &vec_b, &vec_sum);
    std.debug.print("SIMD vector addition result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ vec_sum[0], vec_sum[1], vec_sum[2], vec_sum[3] });

    const bounded_dot = v2.Math.clamp(f32, dot_result, -1_000_000.0, 1_000_000.0);
    std.debug.print("Bounded dot product (v2 clamp): {d:.3}\n", .{bounded_dot});
}
