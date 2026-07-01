const std = @import("std");

pub const name = "accelerator-plugin";
pub const description = "Example reference plugin targeting the feat-accelerator gate.";
pub const version = "0.1.0";
pub const target_feature = "accelerator";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "accelerator-plugin event (bytes={d})", .{input.len});
}
