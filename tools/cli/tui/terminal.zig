const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const windows = std.os.windows;
const events = @import("events.zig");

// Platform detection for cross-platform compatibility
const is_windows = builtin.os.tag == .windows;
const is_wasm = builtin.os.tag == .wasi or builtin.os.tag == .emscripten or
    builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
const is_posix = switch (builtin.os.tag) {
    .linux, .macos, .freebsd, .openbsd, .netbsd, .dragonfly, .solaris, .illumos, .haiku => true,
    else => !is_windows and !is_wasm,
};

/// Platform capabilities for feature detection
pub const PlatformCapabilities = struct {
    /// Whether the platform supports terminal UI
    supports_tui: bool,
    /// Whether mouse input is available
    supports_mouse: bool,
    /// Whether 256-color mode is available
    supports_256_colors: bool,
    /// Whether true-color (24-bit) is available
    supports_true_color: bool,
    /// Whether alternate screen buffer is supported
    supports_alt_screen: bool,
    /// Platform description for display
    platform_name: []const u8,

    /// Detect capabilities for the current platform
    pub fn detect() PlatformCapabilities {
        if (comptime is_wasm) {
            return .{
                .supports_tui = false,
                .supports_mouse = false,
                .supports_256_colors = false,
                .supports_true_color = false,
                .supports_alt_screen = false,
                .platform_name = "WebAssembly",
            };
        }

        if (comptime is_windows) {
            // Windows 10+ supports VT100 escape sequences
            return .{
                .supports_tui = true,
                .supports_mouse = true,
                .supports_256_colors = true,
                .supports_true_color = true,
                .supports_alt_screen = true,
                .platform_name = "Windows",
            };
        }

        if (comptime is_posix) {
            const platform_name = switch (builtin.os.tag) {
                .linux => "Linux",
                .macos => "macOS",
                .freebsd => "FreeBSD",
                .openbsd => "OpenBSD",
                .netbsd => "NetBSD",
                .dragonfly => "DragonFlyBSD",
                .solaris, .illumos => "Solaris/illumos",
                .haiku => "Haiku",
                else => "POSIX",
            };
            return .{
                .supports_tui = true,
                .supports_mouse = true,
                .supports_256_colors = true,
                .supports_true_color = true,
                .supports_alt_screen = true,
                .platform_name = platform_name,
            };
        }

        // Unknown platform - assume minimal support
        return .{
            .supports_tui = false,
            .supports_mouse = false,
            .supports_256_colors = false,
            .supports_true_color = false,
            .supports_alt_screen = false,
            .platform_name = "Unknown",
        };
    }
};

pub const TerminalSize = struct {
    rows: u16,
    cols: u16,
};

const RawState = union(enum) {
    none,
    posix: struct {
        termios: posix.termios,
    },
    windows: struct {
        input_handle: windows.HANDLE,
        output_handle: windows.HANDLE,
        input_mode: windows.DWORD,
        output_mode: windows.DWORD,
    },
};

const ENABLE_ECHO_INPUT: windows.DWORD = 0x0004;
const ENABLE_LINE_INPUT: windows.DWORD = 0x0002;
const ENABLE_PROCESSED_INPUT: windows.DWORD = 0x0001;
const ENABLE_VIRTUAL_TERMINAL_INPUT: windows.DWORD = 0x0200;

pub const Terminal = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    stdin_file: std.Io.File,
    stdout_file: std.Io.File,
    read_storage: [256]u8,
    input_buf: [32]u8,
    input_len: usize,
    input_pos: usize,
    raw_state: RawState,
    active: bool,

    /// Check if TUI is supported on the current platform
    pub fn isSupported() bool {
        return comptime PlatformCapabilities.detect().supports_tui;
    }

    /// Get platform capabilities
    pub fn capabilities() PlatformCapabilities {
        return comptime PlatformCapabilities.detect();
    }

    /// Get the platform name for display
    pub fn platformName() []const u8 {
        return comptime PlatformCapabilities.detect().platform_name;
    }

    pub fn init(allocator: std.mem.Allocator) Terminal {
        var terminal: Terminal = undefined;
        terminal.allocator = allocator;
        terminal.io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        terminal.stdin_file = std.Io.File.stdin();
        terminal.stdout_file = std.Io.File.stdout();
        terminal.read_storage = undefined;
        terminal.input_buf = undefined;
        terminal.input_len = 0;
        terminal.input_pos = 0;
        terminal.raw_state = .none;
        terminal.active = false;
        return terminal;
    }

    pub fn deinit(self: *Terminal) void {
        if (self.active) {
            self.exit() catch {};
        }
        self.io_backend.deinit();
    }

    pub fn enter(self: *Terminal) !void {
        if (self.active) return;

        // Check platform support at comptime
        if (comptime !isSupported()) {
            return error.PlatformNotSupported;
        }

        try self.enterRawMode();
        errdefer self.leaveRawMode() catch {};
        try self.enterAltScreen();
        try self.hideCursor();
        try self.enableMouse();
        try self.clear();
        self.active = true;
    }

    pub fn exit(self: *Terminal) !void {
        if (!self.active) return;
        try self.disableMouse();
        try self.showCursor();
        try self.exitAltScreen();
        try self.leaveRawMode();
        self.active = false;
    }

    pub fn clear(self: *Terminal) !void {
        try self.write("\x1b[2J\x1b[H");
    }

    pub fn enterAltScreen(self: *Terminal) !void {
        try self.write("\x1b[?1049h");
    }

    pub fn exitAltScreen(self: *Terminal) !void {
        try self.write("\x1b[?1049l");
    }

    pub fn hideCursor(self: *Terminal) !void {
        try self.write("\x1b[?25l");
    }

    pub fn showCursor(self: *Terminal) !void {
        try self.write("\x1b[?25h");
    }

    fn enableMouse(self: *Terminal) !void {
        try self.write("\x1b[?1000h\x1b[?1006h");
    }

    fn disableMouse(self: *Terminal) !void {
        try self.write("\x1b[?1000l\x1b[?1006l");
    }

    pub fn write(self: *Terminal, text: []const u8) !void {
        const io = self.io_backend.io();
        // Use writeStreamingAll for Zig 0.16 compatibility
        try self.stdout_file.writeStreamingAll(io, text);
    }

    pub fn readEvent(self: *Terminal) !events.Event {
        const byte_opt = try self.readByte();
        const byte = byte_opt orelse return error.EndOfStream;

        switch (byte) {
            0x03 => return .{ .key = .{ .code = .ctrl_c } },
            0x1b => return try self.readEscapeEvent(),
            0x7f, 0x08 => return .{ .key = .{ .code = .backspace } },
            '\r', '\n' => return .{ .key = .{ .code = .enter } },
            '\t' => return .{ .key = .{ .code = .tab } },
            else => return .{ .key = .{ .code = .character, .char = byte } },
        }
    }

    pub fn readKey(self: *Terminal) !events.Key {
        while (true) {
            const event = try self.readEvent();
            switch (event) {
                .key => |key| return key,
                .mouse => continue,
            }
        }
    }

    pub fn resetInput(self: *Terminal) void {
        self.input_len = 0;
        self.input_pos = 0;
    }

    pub fn size(self: *Terminal) TerminalSize {
        // WASM/WASI - no terminal support
        if (comptime is_wasm) {
            return .{ .rows = 24, .cols = 80 };
        }

        if (comptime builtin.os.tag == .windows) {
            var info: windows.CONSOLE_SCREEN_BUFFER_INFO = undefined;
            if (windows.kernel32.GetConsoleScreenBufferInfo(self.stdout_file.handle, &info) != windows.FALSE) {
                // Window coordinates are 0-indexed, so add 1 for actual row count
                const screen_height: i16 = info.srWindow.Bottom - info.srWindow.Top + 1;
                const screen_width: i16 = info.srWindow.Right - info.srWindow.Left + 1;
                return .{
                    .rows = @intCast(@max(1, screen_height)),
                    .cols = @intCast(@max(1, screen_width)),
                };
            }
            return .{ .rows = 24, .cols = 80 };
        }

        // POSIX-compatible systems (Linux, macOS, BSDs, etc.)
        if (comptime is_posix) {
            var winsize: posix.winsize = .{
                .row = 0,
                .col = 0,
                .xpixel = 0,
                .ypixel = 0,
            };
            const fd = self.stdout_file.handle;
            const err = posix.system.ioctl(fd, posix.T.IOCGWINSZ, @intFromPtr(&winsize));
            if (posix.errno(err) == .SUCCESS and winsize.row != 0 and winsize.col != 0) {
                return .{ .rows = winsize.row, .cols = winsize.col };
            }
            return .{ .rows = 24, .cols = 80 };
        }

        // Fallback for unknown platforms
        return .{ .rows = 24, .cols = 80 };
    }

    fn enterRawMode(self: *Terminal) !void {
        if (comptime is_wasm) {
            return error.PlatformNotSupported;
        }

        if (comptime is_windows) {
            try self.enterRawModeWindows();
        } else if (comptime is_posix) {
            try self.enterRawModePosix();
        } else {
            return error.PlatformNotSupported;
        }
    }

    fn leaveRawMode(self: *Terminal) !void {
        if (comptime is_wasm) {
            return;
        }

        if (comptime is_windows) {
            switch (self.raw_state) {
                .none => return,
                .posix => return,
                .windows => |state| {
                    _ = windows.kernel32.SetConsoleMode(state.input_handle, state.input_mode);
                    _ = windows.kernel32.SetConsoleMode(state.output_handle, state.output_mode);
                },
            }
        } else if (comptime is_posix) {
            switch (self.raw_state) {
                .none => return,
                .posix => |state| try posix.tcsetattr(self.stdin_file.handle, .FLUSH, state.termios),
                .windows => return,
            }
        }
    }

    fn enterRawModePosix(self: *Terminal) !void {
        if (comptime !is_posix) return;

        if (self.raw_state == .none) {
            const original = try posix.tcgetattr(self.stdin_file.handle);
            self.raw_state = .{ .posix = .{ .termios = original } };
        }

        var raw = switch (self.raw_state) {
            .posix => |state| state.termios,
            else => return,
        };

        raw.iflag.BRKINT = false;
        raw.iflag.INPCK = false;
        raw.iflag.ISTRIP = false;
        raw.iflag.ICRNL = false;
        raw.iflag.IXON = false;
        raw.oflag.OPOST = false;
        raw.cflag.CSIZE = .CS8;
        raw.cflag.CREAD = true;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.IEXTEN = false;
        raw.lflag.ISIG = false;

        raw.cc[@intFromEnum(posix.V.MIN)] = 1;
        raw.cc[@intFromEnum(posix.V.TIME)] = 0;

        try posix.tcsetattr(self.stdin_file.handle, .FLUSH, raw);
    }

    fn enterRawModeWindows(self: *Terminal) !void {
        if (self.raw_state == .none) {
            const input_handle = windows.kernel32.GetStdHandle(windows.STD_INPUT_HANDLE) orelse {
                return error.ConsoleUnavailable;
            };
            const output_handle = windows.kernel32.GetStdHandle(windows.STD_OUTPUT_HANDLE) orelse {
                return error.ConsoleUnavailable;
            };

            var input_mode: windows.DWORD = 0;
            var output_mode: windows.DWORD = 0;
            if (windows.kernel32.GetConsoleMode(input_handle, &input_mode) == windows.FALSE) {
                return error.ConsoleModeFailed;
            }
            if (windows.kernel32.GetConsoleMode(output_handle, &output_mode) == windows.FALSE) {
                return error.ConsoleModeFailed;
            }

            self.raw_state = .{ .windows = .{
                .input_handle = input_handle,
                .output_handle = output_handle,
                .input_mode = input_mode,
                .output_mode = output_mode,
            } };
        }

        const state = switch (self.raw_state) {
            .windows => |state| state,
            else => return,
        };

        const input_mode = (state.input_mode | ENABLE_VIRTUAL_TERMINAL_INPUT) &
            ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT);
        const output_mode = state.output_mode | windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING;

        _ = windows.kernel32.SetConsoleMode(state.input_handle, input_mode);
        _ = windows.kernel32.SetConsoleMode(state.output_handle, output_mode);
    }

    fn readEscapeEvent(self: *Terminal) !events.Event {
        const next_opt = try self.readByte();
        const next = next_opt orelse return .{ .key = .{ .code = .escape } };

        if (next != '[' and next != 'O') {
            self.unread(1);
            return .{ .key = .{ .code = .escape } };
        }

        const third_opt = try self.readByte();
        const third = third_opt orelse return .{ .key = .{ .code = .escape } };

        if (next == 'O') {
            return switch (third) {
                'H' => .{ .key = .{ .code = .home } },
                'F' => .{ .key = .{ .code = .end } },
                else => .{ .key = .{ .code = .escape } },
            };
        }

        if (third == '<') {
            if (try self.readMouseSgr()) |mouse| {
                return .{ .mouse = mouse };
            }
            return .{ .key = .{ .code = .escape } };
        }

        switch (third) {
            'A' => return .{ .key = .{ .code = .up } },
            'B' => return .{ .key = .{ .code = .down } },
            'C' => return .{ .key = .{ .code = .right } },
            'D' => return .{ .key = .{ .code = .left } },
            'H' => return .{ .key = .{ .code = .home } },
            'F' => return .{ .key = .{ .code = .end } },
            else => {},
        }

        if (third < '0' or third > '9') {
            return .{ .key = .{ .code = .escape } };
        }

        var digits: [3]u8 = undefined;
        var len: usize = 0;
        digits[len] = third;
        len += 1;

        var terminator: ?u8 = null;
        while (len < digits.len) {
            const more_opt = try self.readByte();
            const more = more_opt orelse break;
            if (more == '~') {
                terminator = more;
                break;
            }
            if (more < '0' or more > '9') {
                break;
            }
            digits[len] = more;
            len += 1;
        }

        if (terminator != '~') {
            return .{ .key = .{ .code = .escape } };
        }

        const code = digits[0..len];
        if (code.len == 1) {
            return switch (code[0]) {
                '1', '7' => .{ .key = .{ .code = .home } },
                '4', '8' => .{ .key = .{ .code = .end } },
                '3' => .{ .key = .{ .code = .delete } },
                '5' => .{ .key = .{ .code = .page_up } },
                '6' => .{ .key = .{ .code = .page_down } },
                else => .{ .key = .{ .code = .escape } },
            };
        }

        return .{ .key = .{ .code = .escape } };
    }

    fn readMouseSgr(self: *Terminal) !?events.Mouse {
        var params: [3]u16 = undefined;
        var param_index: usize = 0;
        var current: u16 = 0;
        var has_digit = false;
        var pressed: bool = false;

        while (true) {
            const next_opt = try self.readByte();
            const next = next_opt orelse return null;
            if (next >= '0' and next <= '9') {
                current = current * 10 + @as(u16, next - '0');
                has_digit = true;
                continue;
            }
            if (next == ';') {
                if (!has_digit or param_index >= params.len) return null;
                params[param_index] = current;
                param_index += 1;
                current = 0;
                has_digit = false;
                continue;
            }
            if (next == 'M' or next == 'm') {
                pressed = next == 'M';
                if (!has_digit or param_index >= params.len) return null;
                params[param_index] = current;
                param_index += 1;
                break;
            }
            return null;
        }

        if (param_index < 3) return null;

        const code = params[0];
        const col = params[1];
        const row = params[2];

        var button: events.MouseButton = .none;
        if (code == 64) {
            button = .wheel_up;
        } else if (code == 65) {
            button = .wheel_down;
        } else switch (code & 3) {
            0 => button = .left,
            1 => button = .middle,
            2 => button = .right,
            else => button = .none,
        }

        return events.Mouse{
            .row = row,
            .col = col,
            .button = button,
            .pressed = pressed,
        };
    }

    fn readByte(self: *Terminal) !?u8 {
        if (self.input_pos >= self.input_len) {
            try self.fillInput();
        }
        if (self.input_len == 0) return null;
        const byte = self.input_buf[self.input_pos];
        self.input_pos += 1;
        return byte;
    }

    fn fillInput(self: *Terminal) !void {
        if (comptime is_wasm) {
            // WASM has no stdin support in the traditional sense
            self.input_len = 0;
            self.input_pos = 0;
            return;
        }

        if (comptime is_windows) {
            var bytes_read: windows.DWORD = 0;
            const result = windows.kernel32.ReadFile(
                self.stdin_file.handle,
                &self.input_buf,
                @intCast(self.input_buf.len),
                &bytes_read,
                null,
            );
            if (result == windows.FALSE) {
                self.input_len = 0;
            } else {
                self.input_len = @intCast(bytes_read);
            }
        } else if (comptime is_posix) {
            const result = posix.system.read(
                self.stdin_file.handle,
                &self.input_buf,
                self.input_buf.len,
            );
            if (result < 0) {
                self.input_len = 0;
            } else {
                self.input_len = @intCast(result);
            }
        } else {
            // Unsupported platform
            self.input_len = 0;
        }
        self.input_pos = 0;
    }

    fn unread(self: *Terminal, count: usize) void {
        if (count <= self.input_pos) {
            self.input_pos -= count;
        }
    }
};
