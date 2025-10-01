const std = @import("std");
const Conn = @import("../../../connectors/mod.zig");

pub fn embed(conn: Conn.Connector, allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    _ = conn;
    _ = allocator;
    _ = text;
    return error.NotImplemented;
}
