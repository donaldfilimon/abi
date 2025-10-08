const std = @import("std");

pub fn getDocument(allocator: std.mem.Allocator, id: []const u8) ![]u8 {
    _ = id;
    return std.mem.dupe(allocator, u8, "Example document text") catch return error.Alloc;
}

pub fn persistSummary(allocator: std.mem.Allocator, id: []const u8, summary: []const u8) !void {
    _ = allocator;
    _ = id;
    _ = summary;
}
