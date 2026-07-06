const std = @import("std");

pub const name = "example-wdbx-plugin";
pub const description = "Example WDBX plugin used by multi-plugin registry contract tests.";
pub const version = "0.1.0";
pub const target_feature = "wdbx";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
