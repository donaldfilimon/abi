const std = @import("std");

pub const name = "nn-plugin";
pub const description = "Example reference plugin targeting the feat-nn gate.";
pub const version = "0.1.0";
pub const target_feature = "nn";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "nn-plugin event (bytes={d})", .{input.len});
}
