//! Nested integration tests demonstrating hierarchical artifacts.

const std = @import("std");

test "basic nested test output generation" {
    // Determine the path to the integration-tests output directory
    const test_dir = "tools/cli/tests/.integration-tests/run-001/golden";

    // Create a dummy output artifact
    var out_file = try std.fs.cwd().createFile(
        try std.fs.path.join(std.testing.allocator, &.{ test_dir, "output.txt" }),
        .{},
    );
    defer out_file.close();

    try out_file.writer().writeAll("OK\n");

    // In a real test, we would compare this against a golden file or assert on it.
    try std.testing.expect(true);
}
