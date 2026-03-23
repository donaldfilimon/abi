//! Terminal input event parsing.
//!
//! Reads and decodes keyboard input from stdin, including
//! multi-byte escape sequences for arrow keys and function keys.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const Key = types.Key;
const Event = types.Event;

/// Input event reader for the terminal.
pub const EventReader = struct {
    fd: if (is_posix) std.posix.fd_t else void,

    const is_posix = switch (builtin.os.tag) {
        .macos, .linux, .freebsd, .netbsd, .openbsd => true,
        else => false,
    };

    pub fn init() EventReader {
        if (comptime is_posix) {
            return .{ .fd = std.posix.STDIN_FILENO };
        }
        return .{ .fd = {} };
    }

    /// Parse a single byte into a Key event.
    pub fn parseKey(byte: u8) Key {
        return switch (byte) {
            '\r', '\n' => .enter,
            '\t' => .tab,
            127 => .backspace,
            0x1b => .escape,
            1...8, 11...12, 14...26 => .{ .ctrl = byte + 'a' - 1 }, // Ctrl+A through Ctrl+Z (excluding \t, \n, \r)
            else => .{ .char = byte },
        };
    }

    /// Parse an escape sequence into a Key event.
    /// Expects the leading ESC has already been consumed.
    pub fn parseEscapeSequence(seq: []const u8) Key {
        if (seq.len == 0) return .escape;
        if (seq[0] != '[') return .{ .alt = seq[0] };

        if (seq.len < 2) return .escape;

        return switch (seq[1]) {
            'A' => .up,
            'B' => .down,
            'C' => .right,
            'D' => .left,
            'H' => .home,
            'F' => .end,
            '3' => if (seq.len > 2 and seq[2] == '~') .delete else .escape,
            '5' => if (seq.len > 2 and seq[2] == '~') .page_up else .escape,
            '6' => if (seq.len > 2 and seq[2] == '~') .page_down else .escape,
            else => .escape,
        };
    }
};

test "parseKey basic characters" {
    try std.testing.expectEqual(Key{ .char = 'a' }, EventReader.parseKey('a'));
    try std.testing.expectEqual(Key{ .char = 'Z' }, EventReader.parseKey('Z'));
    try std.testing.expectEqual(Key{ .char = '0' }, EventReader.parseKey('0'));
}

test "parseKey control characters" {
    try std.testing.expectEqual(Key.enter, EventReader.parseKey('\r'));
    try std.testing.expectEqual(Key.tab, EventReader.parseKey('\t'));
    try std.testing.expectEqual(Key.backspace, EventReader.parseKey(127));
    try std.testing.expectEqual(Key.escape, EventReader.parseKey(0x1b));
}

test "parseKey ctrl sequences" {
    try std.testing.expectEqual(Key{ .ctrl = 'a' }, EventReader.parseKey(1));
    try std.testing.expectEqual(Key{ .ctrl = 'c' }, EventReader.parseKey(3));
    try std.testing.expectEqual(Key{ .ctrl = 'z' }, EventReader.parseKey(26));
}

test "parseEscapeSequence arrows" {
    try std.testing.expectEqual(Key.up, EventReader.parseEscapeSequence("[A"));
    try std.testing.expectEqual(Key.down, EventReader.parseEscapeSequence("[B"));
    try std.testing.expectEqual(Key.right, EventReader.parseEscapeSequence("[C"));
    try std.testing.expectEqual(Key.left, EventReader.parseEscapeSequence("[D"));
}

test "parseEscapeSequence special keys" {
    try std.testing.expectEqual(Key.home, EventReader.parseEscapeSequence("[H"));
    try std.testing.expectEqual(Key.end, EventReader.parseEscapeSequence("[F"));
    try std.testing.expectEqual(Key.delete, EventReader.parseEscapeSequence("[3~"));
    try std.testing.expectEqual(Key.page_up, EventReader.parseEscapeSequence("[5~"));
    try std.testing.expectEqual(Key.page_down, EventReader.parseEscapeSequence("[6~"));
}

test "parseEscapeSequence empty" {
    try std.testing.expectEqual(Key.escape, EventReader.parseEscapeSequence(""));
}

test {
    std.testing.refAllDecls(@This());
}
