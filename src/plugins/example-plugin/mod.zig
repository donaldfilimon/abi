const std = @import("std");

pub const name = "example-plugin";
pub const description = "Minimal example plugin used by registry generation tests.";
pub const version = "0.1.0";
pub const target_feature = "plugins";

pub fn register() void {}

/// Future execution entry point (currently called via the manager's run hook).
pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "example-plugin received input (len={d})", .{input.len});
}
