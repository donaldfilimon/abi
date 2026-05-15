const std = @import("std");

pub const OpenAIRequest = struct {
    model: []const u8,
    stream: bool,
};

pub fn handleOpenAIChatCompletions(allocator: std.mem.Allocator, request_body: []const u8) !void {
    const parsed = try std.json.parseFromSlice(OpenAIRequest, allocator, request_body, .{});
    defer parsed.deinit();

    if (parsed.value.stream) {
        std.log.info("Streaming requested for model: {s}", .{parsed.value.model});
    } else {
        std.log.info("Non-streaming requested for model: {s}", .{parsed.value.model});
    }
}
