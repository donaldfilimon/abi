const std = @import("std");
pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    std.log.info("Abbey processing: {s}", .{input});
    return try std.fmt.allocPrint(allocator, "Abbey analyzed: {s}", .{input});
}
