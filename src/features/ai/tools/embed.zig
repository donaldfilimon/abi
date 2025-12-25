const std = @import("std");
const Conn = @import("../../connectors/mod.zig");

pub fn embed(conn: Conn.Connector, allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    _ = conn;

    var result = try allocator.alloc(f32, 768);
    errdefer allocator.free(result);

    var i: usize = 0;
    for (text) |byte| {
        result[i] = @as(f32, @floatFromInt(byte)) / 255.0;
        i += 1;
        if (i >= result.len) break;
    }

    while (i < result.len) : (i += 1) {
        result[i] = 0.0;
    }

    return result;
}
