const std = @import("std");

pub const name = "example-plugin";
pub const description = "Minimal example plugin used by registry generation tests.";
pub const version = "0.1.0";
pub const target_feature = "plugins";

pub fn register() void {}

/// Future execution entry point (currently called via the manager's run hook).
/// Handles `__context__:<name>` inputs for context providers declared in the
/// plugin manifest (`context_providers` array).
pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, input, "__context__:")) {
        const provider = input["__context__:".len..];
        if (std.mem.eql(u8, provider, "example-info")) {
            return try allocator.dupe(u8, "example-plugin v0.1.0 (target: plugins)");
        }
        return try std.fmt.allocPrint(allocator, "unknown context provider: {s}", .{provider});
    }
    return try std.fmt.allocPrint(allocator, "example-plugin received input (len={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}
