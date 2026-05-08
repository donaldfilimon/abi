const std = @import("std");
const escape_json = @import("../../../../foundation/mod.zig").utils.json.escapeJsonContent;
const types = @import("../types.zig");

pub fn allocJsonMessages(
    allocator: std.mem.Allocator,
    history: []const types.Message,
) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();

    try aw.writer.writeAll("[");
    for (history, 0..) |message, index| {
        if (index > 0) try aw.writer.writeAll(",");

        const escaped_content = try escape_json(allocator, message.content);
        defer allocator.free(escaped_content);

        try aw.writer.print(
            "{{\"role\":\"{s}\",\"content\":\"{s}\"}}",
            .{ roleString(message.role), escaped_content },
        );
    }
    try aw.writer.writeAll("]");

    return aw.toOwnedSlice();
}

pub fn allocTranscript(
    allocator: std.mem.Allocator,
    history: []const types.Message,
) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    errdefer aw.deinit();

    for (history) |message| {
        try aw.writer.print("{s}{s}\n", .{
            rolePrefix(message.role),
            message.content,
        });
    }
    try aw.writer.writeAll("Assistant: ");

    return aw.toOwnedSlice();
}

fn roleString(role: types.Message.Role) []const u8 {
    return switch (role) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
    };
}

fn rolePrefix(role: types.Message.Role) []const u8 {
    return switch (role) {
        .system => "System: ",
        .user => "User: ",
        .assistant => "Assistant: ",
    };
}

test {
    std.testing.refAllDecls(@This());
}
