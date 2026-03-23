//! TUI stub — disabled at compile time.

const std = @import("std");
pub const types = @import("types.zig");

pub const terminal = struct {};
pub const ansi = struct {};
pub const render = struct {};
pub const layout = struct {};
pub const widgets = struct {};
pub const events = struct {};
pub const dashboard = struct {};

pub const TuiError = types.TuiError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
