const std = @import("std");

pub const name = "example-wdbx-plugin";
pub const description = "Example WDBX plugin used by multi-plugin registry contract tests.";
pub const version = "0.1.0";
pub const target_feature = "wdbx";

pub fn register() void {}

/// Future execution entry point (WDBX-aware plugins will receive context in later iterations).
pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "example-wdbx-plugin executed (input len={d})", .{input.len});
}
