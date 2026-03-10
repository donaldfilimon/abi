///! CEL source file management.
///! Provides line/column tracking and source text access.
const std = @import("std");

pub const SourceFile = struct {
    name: []const u8,
    contents: []const u8,

    /// Compute the 1-indexed line and column for a byte offset.
    pub fn lineColFromOffset(self: SourceFile, offset: u32) struct { line: u32, col: u32 } {
        const off: usize = @min(offset, @as(u32, @intCast(self.contents.len)));
        var line: u32 = 1;
        var last_nl: usize = 0; // position after last newline (or 0)
        for (self.contents[0..off], 0..) |c, i| {
            if (c == '\n') {
                line += 1;
                last_nl = i + 1;
            }
        }
        return .{ .line = line, .col = @as(u32, @intCast(off - last_nl)) + 1 };
    }

    /// Return the contents of a 1-indexed line, without the trailing newline.
    /// Returns null if the line number is out of range.
    pub fn getLine(self: SourceFile, line_number: u32) ?[]const u8 {
        if (line_number == 0) return null;

        var current_line: u32 = 1;
        var line_start: usize = 0;

        for (self.contents, 0..) |c, i| {
            if (current_line == line_number) {
                if (c == '\n') {
                    return self.contents[line_start..i];
                }
            } else {
                if (c == '\n') {
                    current_line += 1;
                    line_start = i + 1;
                }
            }
        }

        // Handle last line (no trailing newline).
        if (current_line == line_number and line_start <= self.contents.len) {
            return self.contents[line_start..];
        }

        return null;
    }

    /// Return the total number of lines in the source.
    pub fn lineCount(self: SourceFile) u32 {
        if (self.contents.len == 0) return 0;
        var count: u32 = 1;
        for (self.contents) |c| {
            if (c == '\n') count += 1;
        }
        // If the file ends with a newline, the last "line" is empty;
        // we still count it (consistent with most editors).
        return count;
    }
};

// ── Tests ────────────────────────────────────────────────────────────

test "lineColFromOffset basic" {
    const src = SourceFile{
        .name = "test.cel",
        .contents = "ab\ncd\nef",
    };
    // 'a' is at offset 0 -> line 1, col 1
    const lc0 = src.lineColFromOffset(0);
    try std.testing.expectEqual(@as(u32, 1), lc0.line);
    try std.testing.expectEqual(@as(u32, 1), lc0.col);

    // 'd' is at offset 4 -> line 2, col 2
    const lc4 = src.lineColFromOffset(4);
    try std.testing.expectEqual(@as(u32, 2), lc4.line);
    try std.testing.expectEqual(@as(u32, 2), lc4.col);

    // 'e' is at offset 6 -> line 3, col 1
    const lc6 = src.lineColFromOffset(6);
    try std.testing.expectEqual(@as(u32, 3), lc6.line);
    try std.testing.expectEqual(@as(u32, 1), lc6.col);
}

test "getLine" {
    const src = SourceFile{
        .name = "test.cel",
        .contents = "hello\nworld\nfoo",
    };
    try std.testing.expectEqualStrings("hello", src.getLine(1).?);
    try std.testing.expectEqualStrings("world", src.getLine(2).?);
    try std.testing.expectEqualStrings("foo", src.getLine(3).?);
    try std.testing.expect(src.getLine(0) == null);
    try std.testing.expect(src.getLine(4) == null);
}

test "lineCount" {
    const s1 = SourceFile{ .name = "a", .contents = "one\ntwo\nthree" };
    try std.testing.expectEqual(@as(u32, 3), s1.lineCount());

    const s2 = SourceFile{ .name = "b", .contents = "one\ntwo\n" };
    try std.testing.expectEqual(@as(u32, 3), s2.lineCount());

    const s3 = SourceFile{ .name = "c", .contents = "" };
    try std.testing.expectEqual(@as(u32, 0), s3.lineCount());
}
