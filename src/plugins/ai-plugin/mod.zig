const std = @import("std");

pub const name = "ai-plugin";
pub const description = "Example reference plugin targeting the feat-ai gate.";
pub const version = "0.1.0";
pub const target_feature = "ai";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "ai-plugin event (bytes={d})", .{input.len});
}
