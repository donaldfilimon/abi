const std = @import("std");

pub fn ensureNonEmpty(name: []const u8, value: []const u8) !void {
    if (value.len == 0) {
        std.log.err("{s} must not be empty", .{name});
        return error.Empty;
    }
}
