//! Integration tests for end-to-end functionality.
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    std.debug.print("Running integration tests...\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
        .enable_database = true,
        .enable_gpu = false,
    });
    defer abi.shutdown(&framework);

    std.debug.print("Framework initialized\n", .{});

    try testDatabaseOperations(allocator);
    try testVectorOperations(allocator);

    std.debug.print("\n✅ All integration tests passed!\n", .{});
}

fn testDatabaseOperations(allocator: std.mem.Allocator) !void {
    std.debug.print("Test: Database operations\n", .{});

    const handle = try abi.database.createDatabase(allocator, "integration_test");
    defer abi.database.closeDatabase(&handle);

    const vec1 = &.{ 1.0, 0.0, 0.0 };
    const vec2 = &.{ 0.0, 1.0, 0.0 };
    const vec3 = &.{ 0.0, 0.0, 1.0 };

    try abi.database.insertVector(handle, 1, vec1, null);
    try abi.database.insertVector(handle, 2, vec2, null);
    try abi.database.insertVector(handle, 3, vec3, null);

    const query = &.{ 1.0, 0.0, 0.0 };
    const results = try abi.database.searchVectors(handle, query, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expect(results[0].score > 0.9);

    const stats = abi.database.getStats(&handle);
    try std.testing.expectEqual(@as(usize, 3), stats.count);
    try std.testing.expectEqual(@as(usize, 3), stats.dimension);

    std.debug.print("  ✅ Database operations passed\n", .{});
}

fn testVectorOperations() !void {
    std.debug.print("Test: Vector operations\n", .{});
    const simd = abi.simd;

    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var result: [4]f32 = undefined;

    simd.vectorAdd(&a, &b, &result);
    try std.testing.expectEqual(@as(f32, 3.0), result[0]);
    try std.testing.expectEqual(@as(f32, 5.0), result[1]);
    try std.testing.expectEqual(@as(f32, 7.0), result[2]);
    try std.testing.expectEqual(@as(f32, 9.0), result[3]);

    const dot = simd.vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), dot, 1e-6);

    const norm_a = simd.vectorL2Norm(&a);
    try std.testing.expectApproxEqAbs(@as(f32, 5.4772), norm_a, 1e-4);

    const similarity = simd.cosineSimilarity(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), similarity, 1e-6);

    std.debug.print("  ✅ Vector operations passed\n", .{});
}
