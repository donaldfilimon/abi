//! Pure, bounded line editing for the interactive agent REPL.
//!
//! This module deliberately knows nothing about terminals, WDBX, or command
//! dispatch. Keeping state transitions here lets the raw-terminal path stay
//! small and makes editing behavior testable without a TTY.

const std = @import("std");

pub const MAX_LINE_BYTES = 4096;
pub const MAX_HISTORY_ENTRIES = 100;

pub const Key = union(enum) {
    printable: u8,
    left,
    right,
    home,
    end,
    up,
    down,
    delete,
    backspace,
    tab,
    enter,
    eof,
    ignore,
};

const EscapeState = enum { normal, escape, csi, ss3 };

/// Incremental ANSI key decoder. Unsupported and incomplete sequences are
/// ignored rather than forwarded into the editor as prompt text.
pub const KeyDecoder = struct {
    state: EscapeState = .normal,
    csi_params: [8]u8 = undefined,
    csi_len: usize = 0,

    pub fn pending(self: *const KeyDecoder) bool {
        return self.state != .normal;
    }

    pub fn cancelPending(self: *KeyDecoder) void {
        self.state = .normal;
    }

    pub fn feed(self: *KeyDecoder, byte: u8) ?Key {
        switch (self.state) {
            .normal => return self.feedNormal(byte),
            .escape => return self.feedEscape(byte),
            .csi => return self.feedCsi(byte),
            .ss3 => return self.feedSs3(byte),
        }
    }

    fn feedNormal(self: *KeyDecoder, byte: u8) ?Key {
        return switch (byte) {
            0x1b => blk: {
                self.state = .escape;
                break :blk null;
            },
            '\r', '\n' => .enter,
            0x01 => .home, // Ctrl-A
            0x04 => .eof,
            0x05 => .end, // Ctrl-E
            0x08, 0x7f => .backspace,
            '\t' => .tab,
            else => if (byte >= 0x20 and byte < 0x7f) .{ .printable = byte } else .ignore,
        };
    }

    fn feedEscape(self: *KeyDecoder, byte: u8) ?Key {
        self.state = switch (byte) {
            '[' => blk: {
                self.csi_len = 0;
                break :blk .csi;
            },
            'O' => .ss3,
            else => .normal,
        };
        return null;
    }

    fn feedCsi(self: *KeyDecoder, byte: u8) ?Key {
        if (byte >= 0x40 and byte <= 0x7e) {
            const params = self.csi_params[0..self.csi_len];
            self.state = .normal;
            if (params.len == 0) return switch (byte) {
                'A' => .up,
                'B' => .down,
                'C' => .right,
                'D' => .left,
                'H' => .home,
                'F' => .end,
                else => .ignore,
            };
            if (byte != '~') return .ignore;
            if (std.mem.eql(u8, params, "3")) return .delete;
            if (std.mem.eql(u8, params, "1") or std.mem.eql(u8, params, "7")) return .home;
            if (std.mem.eql(u8, params, "4") or std.mem.eql(u8, params, "8")) return .end;
            return .ignore;
        }
        if (self.csi_len == self.csi_params.len) {
            self.state = .normal;
            return .ignore;
        }
        self.csi_params[self.csi_len] = byte;
        self.csi_len += 1;
        return null;
    }

    fn feedSs3(self: *KeyDecoder, byte: u8) ?Key {
        self.state = .normal;
        return switch (byte) {
            'H' => .home,
            'F' => .end,
            else => .ignore,
        };
    }
};

pub const LineEditor = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8) = .empty,
    draft: std.ArrayListUnmanaged(u8) = .empty,
    history: std.ArrayListUnmanaged([]u8) = .empty,
    cursor: usize = 0,
    history_index: ?usize = null,

    pub fn init(allocator: std.mem.Allocator) LineEditor {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LineEditor) void {
        self.buffer.deinit(self.allocator);
        self.draft.deinit(self.allocator);
        for (self.history.items) |item| self.allocator.free(item);
        self.history.deinit(self.allocator);
    }

    pub fn text(self: *const LineEditor) []const u8 {
        return self.buffer.items;
    }

    pub fn insertPrintable(self: *LineEditor, byte: u8) !bool {
        if (byte < 0x20 or byte >= 0x7f or self.buffer.items.len >= MAX_LINE_BYTES) return false;
        try self.buffer.insert(self.allocator, self.cursor, byte);
        self.cursor += 1;
        self.history_index = null;
        return true;
    }

    pub fn moveLeft(self: *LineEditor) bool {
        if (self.cursor == 0) return false;
        self.cursor -= 1;
        return true;
    }

    pub fn moveRight(self: *LineEditor) bool {
        if (self.cursor == self.buffer.items.len) return false;
        self.cursor += 1;
        return true;
    }

    pub fn moveHome(self: *LineEditor) bool {
        if (self.cursor == 0) return false;
        self.cursor = 0;
        return true;
    }

    pub fn moveEnd(self: *LineEditor) bool {
        if (self.cursor == self.buffer.items.len) return false;
        self.cursor = self.buffer.items.len;
        return true;
    }

    pub fn deleteBackward(self: *LineEditor) bool {
        if (self.cursor == 0) return false;
        _ = self.buffer.orderedRemove(self.cursor - 1);
        self.cursor -= 1;
        self.history_index = null;
        return true;
    }

    pub fn deleteForward(self: *LineEditor) bool {
        if (self.cursor == self.buffer.items.len) return false;
        _ = self.buffer.orderedRemove(self.cursor);
        self.history_index = null;
        return true;
    }

    pub fn replace(self: *LineEditor, value: []const u8) !void {
        self.buffer.clearRetainingCapacity();
        try self.buffer.appendSlice(self.allocator, value);
        self.cursor = value.len;
    }

    pub fn clear(self: *LineEditor) void {
        self.buffer.clearRetainingCapacity();
        self.draft.clearRetainingCapacity();
        self.cursor = 0;
        self.history_index = null;
    }

    /// Record the submitted line in ephemeral history. Existing consecutive
    /// duplicates are not re-added, and retaining the newest 100 entries keeps
    /// memory bounded for long-lived sessions.
    pub fn recordSubmitted(self: *LineEditor) !void {
        const value = self.text();
        if (value.len == 0) return;
        if (self.history.items.len > 0 and std.mem.eql(u8, self.history.items[self.history.items.len - 1], value)) return;

        if (self.history.items.len == MAX_HISTORY_ENTRIES) {
            const oldest = self.history.orderedRemove(0);
            self.allocator.free(oldest);
        }
        try self.history.append(self.allocator, try self.allocator.dupe(u8, value));
    }

    pub fn historyUp(self: *LineEditor) !bool {
        if (self.history.items.len == 0) return false;
        if (self.history_index) |index| {
            if (index == 0) return false;
            self.history_index = index - 1;
        } else {
            self.draft.clearRetainingCapacity();
            try self.draft.appendSlice(self.allocator, self.buffer.items);
            self.history_index = self.history.items.len - 1;
        }
        try self.replace(self.history.items[self.history_index.?]);
        return true;
    }

    pub fn historyDown(self: *LineEditor) !bool {
        const index = self.history_index orelse return false;
        if (index + 1 < self.history.items.len) {
            self.history_index = index + 1;
            try self.replace(self.history.items[self.history_index.?]);
            return true;
        }
        self.history_index = null;
        try self.replace(self.draft.items);
        return true;
    }
};

test "line editor supports cursor movement and deletion" {
    var editor = LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    for ("ac") |byte| _ = try editor.insertPrintable(byte);
    try std.testing.expect(editor.moveLeft());
    _ = try editor.insertPrintable('b');
    try std.testing.expectEqualStrings("abc", editor.text());
    try std.testing.expectEqual(@as(usize, 2), editor.cursor);
    try std.testing.expect(editor.deleteForward());
    try std.testing.expectEqualStrings("ab", editor.text());
    try std.testing.expect(editor.moveHome());
    try std.testing.expect(editor.deleteForward());
    try std.testing.expectEqualStrings("b", editor.text());
    try std.testing.expect(editor.moveEnd());
    try std.testing.expect(editor.deleteBackward());
    try std.testing.expectEqualStrings("", editor.text());
}

test "line editor history is bounded deduplicated and restores drafts" {
    var editor = LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    try editor.replace("first");
    try editor.recordSubmitted();
    try editor.recordSubmitted();
    try editor.replace("second");
    try editor.recordSubmitted();
    try std.testing.expectEqual(@as(usize, 2), editor.history.items.len);

    try editor.replace("draft");
    try std.testing.expect(try editor.historyUp());
    try std.testing.expectEqualStrings("second", editor.text());
    try std.testing.expect(try editor.historyUp());
    try std.testing.expectEqualStrings("first", editor.text());
    try std.testing.expect(try editor.historyDown());
    try std.testing.expectEqualStrings("second", editor.text());
    try std.testing.expect(try editor.historyDown());
    try std.testing.expectEqualStrings("draft", editor.text());
}

test "line editor evicts the oldest entry after one hundred submissions" {
    var editor = LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    var buf: [16]u8 = undefined;
    for (0..MAX_HISTORY_ENTRIES + 1) |index| {
        const value = try std.fmt.bufPrint(&buf, "entry-{d}", .{index});
        try editor.replace(value);
        try editor.recordSubmitted();
    }
    try std.testing.expectEqual(@as(usize, MAX_HISTORY_ENTRIES), editor.history.items.len);
    try std.testing.expectEqualStrings("entry-1", editor.history.items[0]);
    try std.testing.expectEqualStrings("entry-100", editor.history.items[editor.history.items.len - 1]);
}

test "line editor key decoder handles ANSI navigation and ignores malformed sequences" {
    var decoder = KeyDecoder{};
    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    try std.testing.expect(decoder.pending());
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('['));
    try std.testing.expectEqual(Key.up, decoder.feed('A').?);

    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('['));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('3'));
    try std.testing.expectEqual(Key.delete, decoder.feed('~').?);

    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('['));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('1'));
    try std.testing.expectEqual(Key.home, decoder.feed('~').?);
    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('['));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('4'));
    try std.testing.expectEqual(Key.end, decoder.feed('~').?);

    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('['));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('1'));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed(';'));
    try std.testing.expectEqual(@as(?Key, null), decoder.feed('5'));
    try std.testing.expectEqual(Key.ignore, decoder.feed('D').?);
    try std.testing.expectEqual(@as(?Key, null), decoder.feed(0x1b));
    decoder.cancelPending();
    try std.testing.expect(!decoder.pending());
    try std.testing.expectEqual(Key.home, decoder.feed(0x01).?);
    try std.testing.expectEqual(Key.end, decoder.feed(0x05).?);
    try std.testing.expectEqual(Key.ignore, decoder.feed(0x00).?);
}

test "line editor unsupported escape sequence tails never enter the editor buffer" {
    var decoder = KeyDecoder{};
    var editor = LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    const input = "a\x1b[1;5Db";
    for (input) |byte| {
        if (decoder.feed(byte)) |key| switch (key) {
            .printable => |printable| _ = try editor.insertPrintable(printable),
            else => {},
        };
    }
    try std.testing.expectEqualStrings("ab", editor.text());
}

test {
    std.testing.refAllDecls(@This());
}
