//! Database CLI command.
//!
//! Delegates to the database feature's built-in CLI handler.

const std = @import("std");
const abi = @import("abi");

/// Run the database command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    try abi.database.cli.run(allocator, args);
}
