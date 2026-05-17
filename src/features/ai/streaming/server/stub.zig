const std = @import("std");
pub fn handleOpenAIChatCompletions(allocator: std.mem.Allocator, request: []const u8, writer: anytype) !void {
    _ = allocator;
    _ = request;
    try writer.writeAll("{\"error\":\"AI streaming feature is disabled\"}");
}
