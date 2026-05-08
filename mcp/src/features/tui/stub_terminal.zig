const std = @import("std");

pub const TerminalSize = struct {
    width: u16,
    height: u16,
};

pub const Terminal = struct {
    original_termios: void = {},
    fd: void = {},
    raw_mode_active: bool = false,

    pub fn init() !Terminal {
        return .{};
    }

    pub fn enableRawMode(self: *Terminal) !void {
        _ = self;
    }

    pub fn disableRawMode(self: *Terminal) void {
        _ = self;
    }

    pub fn getSize(self: *const Terminal) !TerminalSize {
        _ = self;
        return .{ .width = 80, .height = 24 };
    }

    pub fn deinit(self: *Terminal) void {
        _ = self;
    }
};
