const std = @import("std");

pub const name = "metrics-plugin";
pub const description = "Example reference plugin targeting the feat-metrics gate.";
pub const version = "0.1.0";
pub const target_feature = "metrics";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
