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
    /// Original terminal state for restoration.
    original_termios: if (is_posix) std.posix.termios else void,
    /// File descriptor for the terminal.
    fd: if (is_posix) std.posix.fd_t else void,
    /// Whether raw mode is currently active.
    raw_mode_active: bool = false,

    const is_posix = switch (builtin.os.tag) {
        .macos, .linux, .freebsd, .netbsd, .openbsd => true,
        else => false,
    };

    /// Initialize the terminal controller.
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

    /// Enable raw terminal mode (disable echo, canonical, signals).
    /// Manipulates individual struct fields of Zig 0.16's packed termios flags.
    pub fn enableRawMode(self: *Terminal) !void {
        if (comptime !is_posix) return;
        var raw = self.original_termios;

        // Input: disable break, CR→NL, parity, strip, flow control
        raw.iflag.BRKINT = false;
        raw.iflag.ICRNL = false;
        raw.iflag.INPCK = false;
        raw.iflag.ISTRIP = false;
        raw.iflag.IXON = false;

        // Output: disable post-processing
        raw.oflag.OPOST = false;

        // Local: disable echo, canonical, signals, extended
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.IEXTEN = false;
        raw.lflag.ISIG = false;

        // Control: set 8-bit chars via CSIZE field
        raw.cflag.CSIZE = .CS8;

        // Read: minimum 0 chars, timeout 100ms
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 1;

        try std.posix.tcsetattr(self.fd, .FLUSH, raw);
        self.raw_mode_active = true;
    }

    /// Restore original terminal mode.
    pub fn disableRawMode(self: *Terminal) void {
        if (comptime !is_posix) return;
        if (!self.raw_mode_active) return;
        std.posix.tcsetattr(self.fd, .FLUSH, self.original_termios) catch {};
        self.raw_mode_active = false;
    }

    /// Get the terminal size via ioctl.
    pub fn getSize(self: *const Terminal) !TerminalSize {
        if (comptime !is_posix) {
            return .{ .width = 80, .height = 24 };
        }
        const TIOCGWINSZ = 0x5413;
        const macos_TIOCGWINSZ = 0x40087468;

        const ioctl_val = if (builtin.os.tag == .macos) macos_TIOCGWINSZ else TIOCGWINSZ;

        const Winsize = extern struct {
            ws_row: u16,
            ws_col: u16,
            ws_xpixel: u16,
            ws_ypixel: u16,
        };
        var ws: Winsize = undefined;
        const result = std.posix.system.ioctl(self.fd, ioctl_val, @intFromPtr(&ws));
        if (result != 0) {
            return .{ .width = 80, .height = 24 }; // fallback
        }
        return .{
            .width = if (ws.ws_col > 0) ws.ws_col else 80,
            .height = if (ws.ws_row > 0) ws.ws_row else 24,
        };
    }

    /// Cleanup: restore terminal state.
    pub fn deinit(self: *Terminal) void {
        self.disableRawMode();
    }
};

test "Terminal struct fields exist" {
    // Verify the Terminal struct can be instantiated (compile-time check)
    const T = Terminal;
    try std.testing.expect(@sizeOf(T) > 0);
}

test {
    std.testing.refAllDecls(@This());
}
