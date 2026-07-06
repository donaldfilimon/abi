const std = @import("std");

pub const name = "shader-plugin";
pub const description = "Example reference plugin targeting the feat-shader gate.";
pub const version = "0.1.0";
pub const target_feature = "shader";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
