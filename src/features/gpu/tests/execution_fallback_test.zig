const std = @import("std");
const exec = @import("../execution_coordinator.zig");

test "GPU to SIMD fallback on GPU unavailable" {
    const allocator = std.testing.allocator;

    var coordinator = try exec.ExecutionCoordinator.init(allocator, .{
        .prefer_gpu = true,
        .fallback_chain = &.{ .simd, .scalar },
    });
    defer coordinator.deinit();

    const input_a = [_]f32{ 1, 2, 3, 4 };
    const input_b = [_]f32{ 5, 6, 7, 8 };
    var result = [_]f32{ 0, 0, 0, 0 };

    const exec_method = try coordinator.vectorAdd(&input_a, &input_b, &result);

    // Should use best available method
    try std.testing.expect(exec_method != .failed);
    try std.testing.expectEqual(@as(f32, 6), result[0]);
    try std.testing.expectEqual(@as(f32, 8), result[1]);
    try std.testing.expectEqual(@as(f32, 10), result[2]);
    try std.testing.expectEqual(@as(f32, 12), result[3]);
}

test "automatic method selection based on size" {
    const allocator = std.testing.allocator;

    var coordinator = try exec.ExecutionCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    // Small vectors use SIMD/scalar
    const small_a = [_]f32{1} ** 10;
    const small_b = [_]f32{2} ** 10;
    var small_result = [_]f32{0} ** 10;

    const small_method = try coordinator.vectorAdd(&small_a, &small_b, &small_result);

    // Should not use GPU for tiny vectors
    try std.testing.expect(small_method != .gpu);
    try std.testing.expectEqual(@as(f32, 3), small_result[0]);
}

test "explicit method override" {
    const allocator = std.testing.allocator;

    var coordinator = try exec.ExecutionCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    const input_a = [_]f32{ 1, 2, 3, 4 };
    const input_b = [_]f32{ 5, 6, 7, 8 };
    var result = [_]f32{ 0, 0, 0, 0 };

    const exec_method = try coordinator.vectorAddWithMethod(
        &input_a,
        &input_b,
        &result,
        .simd,
    );

    try std.testing.expect(exec_method == .simd or exec_method == .scalar);
    try std.testing.expectEqual(@as(f32, 6), result[0]);
}
