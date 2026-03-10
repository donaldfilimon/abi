const std = @import("std");

// ── Source location types ──────────────────────────────────────────────

pub const Loc = struct {
    start: u32 = 0,
    end: u32 = 0,
};

pub const LineCol = struct {
    line: u32,
    col: u32,
};

// ── Minimal source-file helper (will move to source.zig) ──────────────

pub const SourceFile = struct {
    name: []const u8,
    contents: []const u8,

    pub fn lineColFromOffset(self: SourceFile, offset: u32) LineCol {
        var line: u32 = 1;
        var col: u32 = 1;
        for (self.contents[0..@min(offset, @as(u32, @intCast(self.contents.len)))]) |c| {
            if (c == '\n') {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        return .{ .line = line, .col = col };
    }

    pub fn getLine(self: SourceFile, line_number: u32) ?[]const u8 {
        var current_line: u32 = 1;
        var start: usize = 0;
        for (self.contents, 0..) |c, i| {
            if (current_line == line_number) {
                if (c == '\n') {
                    return self.contents[start..i];
                }
            } else {
                if (c == '\n') {
                    current_line += 1;
                    start = i + 1;
                }
            }
        }
        if (current_line == line_number and start < self.contents.len) {
            return self.contents[start..];
        }
        return null;
    }
};

// ── Diagnostic types ──────────────────────────────────────────────────

pub const Level = enum {
    err,
    warn,
    note,
};

pub const Diagnostic = struct {
    level: Level,
    message: []const u8,
    loc: Loc,

    pub fn render(self: Diagnostic, src: SourceFile, writer: anytype) !void {
        const pos = src.lineColFromOffset(self.loc.start);
        try writer.print("{s}:{d}:{d}: {s}: {s}\n", .{
            src.name,
            pos.line,
            pos.col,
            @tagName(self.level),
            self.message,
        });
        if (src.getLine(pos.line)) |line| {
            try writer.print("  {s}\n", .{line});
            // Caret: 2 spaces indent + (col-1) spaces + caret
            try writer.writeByteNTimes(' ', pos.col + 1);
            try writer.writeAll("^\n");
        }
    }
};

pub const DiagnosticList = struct {
    items: std.ArrayListUnmanaged(Diagnostic),
    has_errors: bool,

    pub const empty: DiagnosticList = .{
        .items = .{},
        .has_errors = false,
    };

    pub fn addError(
        self: *DiagnosticList,
        allocator: std.mem.Allocator,
        loc: Loc,
        message: []const u8,
    ) !void {
        try self.items.append(allocator, .{
            .level = .err,
            .message = message,
            .loc = loc,
        });
        self.has_errors = true;
    }

    pub fn addWarning(
        self: *DiagnosticList,
        allocator: std.mem.Allocator,
        loc: Loc,
        message: []const u8,
    ) !void {
        try self.items.append(allocator, .{
            .level = .warn,
            .message = message,
            .loc = loc,
        });
    }

    pub fn addNote(
        self: *DiagnosticList,
        allocator: std.mem.Allocator,
        loc: Loc,
        message: []const u8,
    ) !void {
        try self.items.append(allocator, .{
            .level = .note,
            .message = message,
            .loc = loc,
        });
    }

    pub fn deinit(self: *DiagnosticList, allocator: std.mem.Allocator) void {
        self.items.deinit(allocator);
    }

    pub fn errorCount(self: DiagnosticList) usize {
        var count: usize = 0;
        for (self.items.items) |item| {
            if (item.level == .err) count += 1;
        }
        return count;
    }

    pub fn renderAll(self: DiagnosticList, src: SourceFile, writer: anytype) !void {
        for (self.items.items) |item| {
            try item.render(src, writer);
        }
    }
};

// ── Tests ─────────────────────────────────────────────────────────────

test "lineColFromOffset basic" {
    const src = SourceFile{ .name = "test.cel", .contents = "abc\ndef\nghi" };
    const pos = src.lineColFromOffset(5); // 'd' is offset 4, 'e' is 5
    try std.testing.expectEqual(@as(u32, 2), pos.line);
    try std.testing.expectEqual(@as(u32, 2), pos.col);
}

test "getLine returns correct line" {
    const src = SourceFile{ .name = "test.cel", .contents = "line1\nline2\nline3" };
    try std.testing.expectEqualStrings("line1", src.getLine(1).?);
    try std.testing.expectEqualStrings("line2", src.getLine(2).?);
    try std.testing.expectEqualStrings("line3", src.getLine(3).?);
    try std.testing.expect(src.getLine(4) == null);
}

test "render error diagnostic" {
    const src = SourceFile{ .name = "file.cel", .contents = "const x = @;" };
    const diag = Diagnostic{
        .level = .err,
        .message = "unexpected token",
        .loc = .{ .start = 10, .end = 11 },
    };
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try diag.render(src, fbs.writer());
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "file.cel:1:11: err: unexpected token") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "^") != null);
}

test "render warning diagnostic" {
    const src = SourceFile{ .name = "test.cel", .contents = "var unused = 1;" };
    const diag = Diagnostic{
        .level = .warn,
        .message = "unused variable",
        .loc = .{ .start = 4, .end = 10 },
    };
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try diag.render(src, fbs.writer());
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "warn: unused variable") != null);
}

test "DiagnosticList tracks errors" {
    const allocator = std.testing.allocator;
    var diags = DiagnosticList.empty;
    defer diags.deinit(allocator);

    try diags.addWarning(allocator, .{ .start = 0, .end = 1 }, "w1");
    try std.testing.expect(!diags.has_errors);

    try diags.addError(allocator, .{ .start = 5, .end = 6 }, "e1");
    try std.testing.expect(diags.has_errors);
    try std.testing.expectEqual(@as(usize, 1), diags.errorCount());
    try std.testing.expectEqual(@as(usize, 2), diags.items.items.len);
}

test "DiagnosticList renderAll" {
    const allocator = std.testing.allocator;
    const src = SourceFile{ .name = "t.cel", .contents = "a + b" };
    var diags = DiagnosticList.empty;
    defer diags.deinit(allocator);

    try diags.addError(allocator, .{ .start = 0, .end = 1 }, "err1");
    try diags.addWarning(allocator, .{ .start = 4, .end = 5 }, "warn1");

    var buf: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try diags.renderAll(src, fbs.writer());
    const output = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "err1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "warn1") != null);
}
