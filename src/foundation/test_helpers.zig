const std = @import("std");

pub const TestWriter = struct {
    allocator: std.mem.Allocator,
    buffer: *std.ArrayListUnmanaged(u8),

    pub fn writeAll(self: *const @This(), bytes: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, bytes);
    }

    pub fn print(self: *const @This(), comptime format: []const u8, args: anytype) !void {
        const rendered = try std.fmt.allocPrint(self.allocator, format, args);
        defer self.allocator.free(rendered);
        try self.buffer.appendSlice(self.allocator, rendered);
    }
};

test {
    std.testing.refAllDecls(@This());
}