const std = @import("std");

pub const name = "foundationmodels-plugin";
pub const description = "Example reference plugin targeting the feat-foundationmodels gate.";
pub const version = "0.1.0";
pub const target_feature = "foundationmodels";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "foundationmodels-plugin event (bytes={d})", .{input.len});
}

test {
    std.testing.refAllDecls(@This());
}
