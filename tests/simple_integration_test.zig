const std = @import("std");
const abi = @import("abi");

/// Simple integration test to isolate issues
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Running Simple Integration Test", .{});

    // Test basic database functionality
    const test_file = "test_simple_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Add a simple vector
    const test_vector = [_]f32{1.0} ** 128;
    const id = try db.addEmbedding(&test_vector);
    std.log.info("  Added vector with ID: {}", .{id});

    // Search for similar vectors
    const results = try db.search(&test_vector, 5, allocator);
    defer allocator.free(results);

    // Format the results count safely
    const result_count = results.len;
    std.log.info("  Found {} similar vectors", .{result_count});

    std.log.info("✅ Simple integration test passed", .{});
}
