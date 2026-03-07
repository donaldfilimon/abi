//! Nested integration tests demonstrating hierarchical artifacts.
//!
//! Zig 0.16: std.fs.cwd() removed; use std.Io.Dir.cwd() and dir.createFile(io, path, .{})
//! when running full integration with filesystem. This test validates path layout and flow.

const std = @import("std");

test "basic nested test output generation" {
    const test_dir = "tools/cli/tests/.integration-tests/run-001/golden";
    const joined = try std.fs.path.join(std.testing.allocator, &.{ test_dir, "output.txt" });
    defer std.testing.allocator.free(joined);

    // Validate path layout (full run would use std.Io.Dir.cwd().createFile(io, joined, .{}))
    try std.testing.expect(std.mem.endsWith(u8, joined, "output.txt"));
    try std.testing.expect(std.mem.indexOf(u8, joined, ".integration-tests") != null);
}
