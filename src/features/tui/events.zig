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

<<<<<<< Updated upstream
    /// Read an event from the file descriptor, blocking until one is available.
    pub fn readEvent(self: *EventReader) !?Event {
        if (comptime !is_posix) return null;

        var buf: [16]u8 = undefined;
        const bytes_read = try std.posix.read(self.fd, &buf);
        if (bytes_read == 0) return null;

        if (buf[0] == 0x1b) {
            if (bytes_read == 1) {
                return .{ .key = .escape };
            }
            return .{ .key = parseEscapeSequence(buf[1..bytes_read]) };
        }

        return .{ .key = parseKey(buf[0]) };
=======
    /// Read a single event from the terminal, handling multi-byte escape sequences.
    pub fn readEvent(self: *EventReader) !Key {
        if (comptime !is_posix) return error.Unsupported;
        var buf: [1]u8 = undefined;
        const bytes_read = std.posix.read(self.fd, &buf) catch |err| return err;
        if (bytes_read == 0) return error.EndOfStream;

        if (buf[0] != 0x1b) {
            return parseKey(buf[0]);
        }

        // It's an escape sequence, try to read more
        var seq: [16]u8 = undefined;
        var seq_len: usize = 0;

        // Use poll to non-blockingly read the rest of the sequence
        var fds: [1]std.posix.pollfd = undefined;
        fds[0] = .{ .fd = self.fd, .events = std.posix.POLL.IN, .revents = 0 };

        while (seq_len < seq.len) {
            // Wait up to 10ms for next byte (escape sequences are sent quickly)
            const num_events = std.posix.poll(&fds, 10) catch break;
            if (num_events == 0) break;

            var b: [1]u8 = undefined;
            const n = std.posix.read(self.fd, &b) catch break;
            if (n == 0) break;
            
            seq[seq_len] = b[0];
            seq_len += 1;
            // Stop parsing if we reach a letter or tilde, typical ends of ANSI sequences
            if ((b[0] >= 'A' and b[0] <= 'Z') or (b[0] >= 'a' and b[0] <= 'z') or b[0] == '~') {
                break;
            }
        }

        return parseEscapeSequence(seq[0..seq_len]);
>>>>>>> Stashed changes
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

        if (seq[0] == 'O' and seq.len == 2) {
            return switch (seq[1]) {
                'P' => .f1,
                'Q' => .f2,
                'R' => .f3,
                'S' => .f4,
                else => .{ .alt = seq[0] },
            };
        }

        if (seq[0] != '[') return .{ .alt = seq[0] };
        if (seq.len < 2) return .escape;

        if (std.mem.eql(u8, seq, "[A")) return .up;
        if (std.mem.eql(u8, seq, "[B")) return .down;
        if (std.mem.eql(u8, seq, "[C")) return .right;
        if (std.mem.eql(u8, seq, "[D")) return .left;
        if (std.mem.eql(u8, seq, "[H")) return .home;
        if (std.mem.eql(u8, seq, "[F")) return .end;

        if (std.mem.eql(u8, seq, "[1~") or std.mem.eql(u8, seq, "[7~")) return .home;
        if (std.mem.eql(u8, seq, "[4~") or std.mem.eql(u8, seq, "[8~")) return .end;

        if (std.mem.eql(u8, seq, "[3~")) return .delete;
        if (std.mem.eql(u8, seq, "[5~")) return .page_up;
        if (std.mem.eql(u8, seq, "[6~")) return .page_down;

        if (std.mem.eql(u8, seq, "[11~")) return .f1;
        if (std.mem.eql(u8, seq, "[12~")) return .f2;
        if (std.mem.eql(u8, seq, "[13~")) return .f3;
        if (std.mem.eql(u8, seq, "[14~")) return .f4;
        if (std.mem.eql(u8, seq, "[15~")) return .f5;
        if (std.mem.eql(u8, seq, "[17~")) return .f6;
        if (std.mem.eql(u8, seq, "[18~")) return .f7;
        if (std.mem.eql(u8, seq, "[19~")) return .f8;
        if (std.mem.eql(u8, seq, "[20~")) return .f9;
        if (std.mem.eql(u8, seq, "[21~")) return .f10;
        if (std.mem.eql(u8, seq, "[23~")) return .f11;
        if (std.mem.eql(u8, seq, "[24~")) return .f12;

        return .escape;
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
    try std.testing.expectEqual(Key.f1, EventReader.parseEscapeSequence("OP"));
    try std.testing.expectEqual(Key.f5, EventReader.parseEscapeSequence("[15~"));
}

test "parseEscapeSequence empty" {
    try std.testing.expectEqual(Key.escape, EventReader.parseEscapeSequence(""));
}

test {
    std.testing.refAllDecls(@This());
}
