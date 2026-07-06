const std = @import("std");

pub const name = "sea-plugin";
pub const description = "Example reference plugin targeting the feat-sea gate.";
pub const version = "0.1.0";
pub const target_feature = "sea";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
