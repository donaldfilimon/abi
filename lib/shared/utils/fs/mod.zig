//! File System Utilities Module
//!
//! File system operations and utilities

const std = @import("std");

/// Read entire file content into a buffer
pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
}

/// Write content to file
pub fn writeFile(path: []const u8, content: []const u8) !void {
    try std.fs.cwd().writeFile(.{ .sub_path = path, .data = content });
}

/// Check if file exists
pub fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

/// Get file size
pub fn getFileSize(path: []const u8) !u64 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.getEndPos();
}

test {
    std.testing.refAllDecls(@This());
}

test "readFile and writeFile" {
    const allocator = std.testing.allocator;
    const test_path = "test_file.txt";
    const content = "Hello, World!";

    // Write file
    try writeFile(test_path, content);
    defer std.fs.cwd().deleteFile(test_path) catch {};

    // Read file
    const read_content = try readFile(allocator, test_path);
    defer allocator.free(read_content);

    try std.testing.expectEqualStrings(content, read_content);
}
