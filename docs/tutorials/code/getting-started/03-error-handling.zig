//! Getting Started Tutorial - Example 3: Error Handling
//!
//! Run with: zig run docs/tutorials/code/getting-started/03-error-handling.zig

const std = @import("std");
// In a real project, you would use: const abi = @import("abi");
// For tutorial purposes, we use a relative path.
const abi = @import("../../../../src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    if (abi.database.openOrCreate(allocator, "test_db")) |*db| {
        defer abi.database.close(db);
        std.debug.print("Database opened successfully\n", .{});
    } else |err| {
        switch (err) {
            error.DatabaseDisabled => {
                std.debug.print("Database feature is disabled.\n", .{});
                std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
            },
            else => {
                std.debug.print("Database error: {t}\n", .{err});
            },
        }
    }
}
