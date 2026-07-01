const std = @import("std");

pub const name = "os-control-plugin";
pub const description = "Example reference plugin targeting the feat-os-control gate.";
pub const version = "0.1.0";
pub const target_feature = "os-control";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}
