const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const test_helpers = @import("../../foundation/test_helpers.zig");

pub const ScreenState = types.ScreenState;

/// Portable stdin file descriptor/handle. `std.Io.File.stdin().handle` is a
/// `std.posix.fd_t` on every target (fd 0 on POSIX, the console HANDLE on
/// Windows), avoiding the `STDIN_FILENO` comptime-int vs HANDLE mismatch.
pub fn stdinFd() std.posix.fd_t {
    return std.Io.File.stdin().handle;
}

/// Raw-mode interactive terminal. Comptime-selected per platform so non-POSIX
/// targets never instantiate the termios/poll path.
pub const InteractiveTerminal = if (builtin.os.tag == .windows)
    WindowsInteractiveTerminal
else
    PosixInteractiveTerminal;

pub const ScreenSession = struct {
    term: InteractiveTerminal,
    screen_active: bool = false,

    pub fn init(fd: std.posix.fd_t) !ScreenSession {
        var term = try InteractiveTerminal.init(fd);
        errdefer term.deinit();
        try initScreen();
        return .{ .term = term, .screen_active = true };
    }

    pub fn deinit(self: *ScreenSession) void {
        if (self.screen_active) {
            deinitScreen();
            self.screen_active = false;
        }
        self.term.deinit();
    }
};

const PosixInteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !PosixInteractiveTerminal {
        if (@hasDecl(std.posix.system, "isatty") and std.posix.system.isatty(fd) == 0) return error.NotATerminal;

        const original = std.posix.tcgetattr(fd) catch return error.NotATerminal;
        var raw = original;
        raw.lflag.ICANON = false;
        raw.lflag.ECHO = false;
        if (@hasField(@TypeOf(raw.lflag), "ISIG")) raw.lflag.ISIG = false;

        const vmin = if (@hasDecl(std.posix, "VMIN")) std.posix.VMIN else std.posix.system.V.MIN;
        const vtime = if (@hasDecl(std.posix, "VTIME")) std.posix.VTIME else std.posix.system.V.TIME;

        raw.cc[@intFromEnum(vmin)] = 1;
        raw.cc[@intFromEnum(vtime)] = 0;

        try std.posix.tcsetattr(fd, .FLUSH, raw);
        return .{ .fd = fd, .original = original, .is_tty = true };
    }

    pub fn deinit(self: *PosixInteractiveTerminal) void {
        std.posix.tcsetattr(self.fd, .FLUSH, self.original) catch |err| {
            std.log.warn("failed to restore terminal: {s}", .{@errorName(err)});
        };
    }

    pub fn readKey(self: *PosixInteractiveTerminal) ?u8 {
        var buf: [1]u8 = undefined;
        const n = std.posix.read(self.fd, &buf) catch |err| {
            std.log.warn("read stdin failed: {s}", .{@errorName(err)});
            return null;
        };
        if (n == 0) return null;
        return buf[0];
    }

    pub fn pollInput(self: *PosixInteractiveTerminal, timeout_ms: i32) bool {
        var fds = [_]std.posix.pollfd{.{ .fd = self.fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const n = std.posix.poll(&fds, timeout_ms) catch return false;
        return n > 0 and (fds[0].revents & std.posix.POLL.IN) != 0;
    }
};

const WindowsInteractiveTerminal = struct {
    fd: std.posix.fd_t,
    is_tty: bool = false,

    pub fn init(fd: std.posix.fd_t) !WindowsInteractiveTerminal {
        _ = fd;
        return error.NotATerminal;
    }

    pub fn deinit(self: *WindowsInteractiveTerminal) void {
        _ = self;
    }

    pub fn readKey(self: *WindowsInteractiveTerminal) ?u8 {
        _ = self;
        return null;
    }

    pub fn pollInput(self: *WindowsInteractiveTerminal, timeout_ms: i32) bool {
        _ = self;
        _ = timeout_ms;
        return false;
    }
};

pub fn isQuitKey(byte: u8) bool {
    return byte == 'q' or byte == 'Q' or byte == 0x1b or byte == 0x03;
}

pub fn isRefreshKey(byte: u8) bool {
    return byte == 'r' or byte == 'R';
}

pub fn isTabKey(byte: u8) bool {
    return byte == 0x09;
}

pub fn isScrollUpKey(byte: u8) bool {
    return byte == 'k' or byte == 'K';
}

pub fn isScrollDownKey(byte: u8) bool {
    return byte == 'j' or byte == 'J';
}

pub fn initScreen() !void {
    std.debug.print("\x1b[?1049h\x1b[H", .{});
}

pub fn initScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049h\x1b[H");
}

pub fn clearScreen() !void {
    std.debug.print("\x1b[2J\x1b[H", .{});
}

pub fn homeScreen() void {
    std.debug.print("\x1b[H", .{});
}

pub fn homeScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[H");
}

pub fn clearToEnd() void {
    std.debug.print("\x1b[0J", .{});
}

pub fn clearToEndWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[0J");
}

pub fn clearScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[2J\x1b[H");
}

pub fn render(state: ScreenState) !void {
    std.debug.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    std.debug.print("Agents: abbey, aviva, abi | WDBX: in-memory training records\n", .{});
}

pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    try writer.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    try writer.writeAll("Agents: abbey, aviva, abi | WDBX: in-memory training records\n");
}

pub fn deinitScreen() void {
    std.debug.print("\x1b[?1049l", .{});
}

pub fn deinitScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049l");
}

test "writer render functions are testable" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const writer = test_helpers.TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };

    try renderWriter(&writer, .{ .width = 80, .height = 24 });
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "80x24") != null);
}

test "InteractiveTerminal struct layout" {
    const term = if (builtin.os.tag == .windows) InteractiveTerminal{
        .fd = 0,
        .is_tty = false,
    } else InteractiveTerminal{
        .fd = 0,
        .original = undefined,
        .is_tty = false,
    };
    try std.testing.expect(!term.is_tty);
    try std.testing.expectEqual(@as(std.posix.fd_t, 0), term.fd);
}

test "ScreenSession struct layout" {
    const term = if (builtin.os.tag == .windows) InteractiveTerminal{
        .fd = 0,
        .is_tty = false,
    } else InteractiveTerminal{
        .fd = 0,
        .original = undefined,
        .is_tty = false,
    };
    const session = ScreenSession{ .term = term };
    try std.testing.expect(!session.screen_active);
    try std.testing.expectEqual(@as(std.posix.fd_t, 0), session.term.fd);
}

test "alternate screen writer helpers are paired" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const writer = test_helpers.TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };
    try initScreenWriter(&writer);
    try deinitScreenWriter(&writer);
    try std.testing.expectEqualStrings("\x1b[?1049h\x1b[H\x1b[?1049l", buf.items);
}

test "redraw writer helpers emit home and clear-to-end controls" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const writer = test_helpers.TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };
    try homeScreenWriter(&writer);
    try clearToEndWriter(&writer);
    try std.testing.expectEqualStrings("\x1b[H\x1b[0J", buf.items);
}

test "quit and refresh key detection" {
    try std.testing.expect(isQuitKey('q'));
    try std.testing.expect(isQuitKey('Q'));
    try std.testing.expect(isQuitKey(0x1b));
    try std.testing.expect(isQuitKey(0x03));
    try std.testing.expect(!isQuitKey('r'));
    try std.testing.expect(isRefreshKey('r'));
    try std.testing.expect(isRefreshKey('R'));
    try std.testing.expect(!isRefreshKey('q'));
}

test "tab key detection" {
    try std.testing.expect(isTabKey(0x09));
    try std.testing.expect(!isTabKey(' '));
    try std.testing.expect(!isTabKey('t'));
}

test "scroll key detection" {
    try std.testing.expect(isScrollUpKey('k'));
    try std.testing.expect(isScrollUpKey('K'));
    try std.testing.expect(!isScrollUpKey('j'));
    try std.testing.expect(isScrollDownKey('j'));
    try std.testing.expect(isScrollDownKey('J'));
    try std.testing.expect(!isScrollDownKey('k'));
}

test {
    std.testing.refAllDecls(@This());
}
