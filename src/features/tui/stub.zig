//! TUI stub — disabled at compile time.

const std = @import("std");

pub const types = @import("types.zig");

pub const terminal = @import("stub_terminal.zig");
pub const ansi = @import("stub_ansi.zig");
pub const render = @import("stub_render.zig");
pub const layout = @import("stub_layout.zig");
pub const widgets = @import("stub_widgets.zig");
pub const events = @import("stub_events.zig");
pub const dashboard = @import("stub_dashboard.zig");

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

// Module-level lifecycle — inline for clarity, no helper needed.
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
