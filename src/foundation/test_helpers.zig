const std = @import("std");
const temp_path = @import("temp_path.zig");

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

pub fn deleteTestFileIfExists(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch |err| {
        std.debug.print("failed to delete test file '{s}': {s}\n", .{ path, @errorName(err) });
    };
}

test {
    std.testing.refAllDecls(@This());
}