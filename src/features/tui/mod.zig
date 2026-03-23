//! Terminal User Interface
//!
//! Provides terminal rendering, input handling, widgets, and
//! an interactive dashboard for the ABI framework.

pub const types = @import("types.zig");
pub const terminal = @import("terminal.zig");
pub const ansi = @import("ansi.zig");
pub const render = @import("render.zig");
pub const layout = @import("layout.zig");
pub const widgets = @import("widgets.zig");
pub const events = @import("events.zig");
pub const dashboard = @import("dashboard.zig");

const std = @import("std");

pub const TuiError = types.TuiError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = true };
    }

    pub fn deinit(self: *Context) void {
        self.initialized = false;
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return true;
}

test {
    std.testing.refAllDecls(@This());
}

test "Context init and deinit" {
    var ctx = Context.init(std.testing.allocator);
    try std.testing.expect(ctx.initialized);
    ctx.deinit();
    try std.testing.expect(!ctx.initialized);
}

test "isEnabled returns true" {
    try std.testing.expect(isEnabled());
}
