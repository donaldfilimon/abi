//! POSIX terminal abstraction.
//!
//! Provides raw mode control, terminal size detection, and
//! cleanup on exit. Uses POSIX termios for terminal control.

const std = @import("std");
const builtin = @import("builtin");

pub const TerminalSize = struct {
    width: u16,
    height: u16,
};

/// Terminal controller for raw mode and size detection.
pub const Terminal = struct {
    original_termios: if (is_posix) std.posix.termios else void,
    fd: if (is_posix) std.posix.fd_t else void,
    raw_mode_active: bool = false,

    const is_posix = builtin.os.tag == .macos or builtin.os.tag == .linux or builtin.os.tag == .freebsd or builtin.os.tag == .netbsd or builtin.os.tag == .openbsd;

    pub fn init() !Terminal {
        if (comptime !is_posix) {
            return .{
                .original_termios = {},
                .fd = {},
            };
        }
        const fd = std.posix.STDOUT_FILENO;
        if (std.c.isatty(fd) == 0) {
            return error.UnsupportedTerminal;
        }
        const termios = try std.posix.tcgetattr(fd);
        return .{
            .original_termios = termios,
            .fd = fd,
        };
    }

    pub fn enableRawMode(self: *Terminal) !void {
        if (comptime !is_posix) return;
        var raw = self.original_termios;

        // Zig 0.17 models termios flags as typed bitfields instead of integer masks.
        raw.iflag.BRKINT = false;
        raw.iflag.ICRNL = false;
        raw.iflag.INPCK = false;
        raw.iflag.ISTRIP = false;
        raw.iflag.IXON = false;

        raw.oflag.OPOST = false;

        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.IEXTEN = false;
        raw.lflag.ISIG = false;

        raw.cflag.CSIZE = .CS8;

        // Read: minimum 0 chars, timeout 100ms
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 1;

        try std.posix.tcsetattr(self.fd, .FLUSH, raw);
        self.raw_mode_active = true;
    }

    pub fn disableRawMode(self: *Terminal) void {
        if (comptime !is_posix) return;
        if (!self.raw_mode_active) return;
        _ = std.posix.tcsetattr(self.fd, .FLUSH, self.original_termios) catch {};
        self.raw_mode_active = false;
    }

    const default_width: u16 = 80;
    const default_height: u16 = 24;

    pub fn getSize(self: *const Terminal) !TerminalSize {
        if (comptime !is_posix) {
            return .{ .width = default_width, .height = default_height };
        }

        var ws = std.posix.winsize{
            .row = 0,
            .col = 0,
            .xpixel = 0,
            .ypixel = 0,
        };
        const result = std.posix.system.ioctl(self.fd, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
        if (result != 0) {
            return .{ .width = default_width, .height = default_height };
        }
        return .{
            .width = if (ws.col > 0) ws.col else default_width,
            .height = if (ws.row > 0) ws.row else default_height,
        };
    }

    pub fn deinit(self: *Terminal) void {
        self.disableRawMode();
    }
};
